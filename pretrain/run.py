import argparse
import math
import os
import warnings

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch
import wandb

from datasets import build_combined_concat_dataloader
from trainers import OursTopKWrapper
from utils import load_config, save_wandb_run_id, setup_experiment, print_rank_zero


warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = argparse.ArgumentParser(description="Training with YAML Configuration")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Simple overrides
    parser.add_argument("--base_lr", type=float, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--output_dim", type=int, help="Override output dimension")
    parser.add_argument("--max_epochs", type=int, help="Override max epochs")
    parser.add_argument("--warmup_epochs", type=int, help="Override warmup epochs")
    parser.add_argument("--use_siglip", action="store_true", help="Override use siglip")
    parser.add_argument("--use_clip", action="store_true", help="Override use clip")
    parser.add_argument(
        "--segmentation_weight", type=float, help="Override segmentation loss weight"
    )
    parser.add_argument("--mlm_weight", type=float, help="Override mlm loss weight")
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Limit train/val to smoke_test_max_* samples per epoch (quick pipeline check)",
    )
    parser.add_argument(
        "--smoke_test_max_train_samples",
        type=int,
        default=None,
        help="With --smoke_test, max training samples per epoch (default: config or 100)",
    )
    parser.add_argument(
        "--smoke_test_max_val_samples",
        type=int,
        default=None,
        help="With --smoke_test, max val samples per epoch (default: config or 100)",
    )
    parser.add_argument(
        "--smoke_test_max_epochs",
        type=int,
        default=None,
        help="With --smoke_test, override max_epochs (e.g. 2 to hit topk projection warmup boundary)",
    )

    return parser.parse_args()


def main():
    # Initialize DDP if running with torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import torch.distributed as dist
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            print(f"[Rank {dist.get_rank()}/{dist.get_world_size()}] DDP initialized")

    args = parse_args()
    cfg = load_config(args.config)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if args.smoke_test:
        cfg["smoke_test"] = True
    if args.smoke_test_max_train_samples is not None:
        cfg["smoke_test_max_train_samples"] = args.smoke_test_max_train_samples
    if args.smoke_test_max_val_samples is not None:
        cfg["smoke_test_max_val_samples"] = args.smoke_test_max_val_samples
    if args.smoke_test_max_epochs is not None:
        cfg["smoke_test_max_epochs"] = args.smoke_test_max_epochs

    # Apply command line overrides
    if args.base_lr is not None:
        cfg["base_lr"] = args.base_lr
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.output_dim is not None:
        cfg["output_dim"] = args.output_dim
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    if args.warmup_epochs is not None:
        cfg["warmup_epochs"] = args.warmup_epochs
    if args.segmentation_weight is not None:
        cfg["segmentation_weight"] = args.segmentation_weight
    if args.mlm_weight is not None:
        cfg["mlm_weight"] = args.mlm_weight
    if args.use_siglip:
        cfg["use_siglip"] = True
    if args.use_clip:
        cfg["use_siglip"] = False
    if cfg.get("smoke_test") and cfg.get("smoke_test_max_epochs") is not None:
        cfg["max_epochs"] = int(cfg["smoke_test_max_epochs"])
        print_rank_zero(f"SMOKE TEST: max_epochs overridden to {cfg['max_epochs']}")
    seed_everything(cfg["seed"], workers=True, verbose=False)
    # torch.use_deterministic_algorithms(True)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Create experiment name
    vision_model_name = cfg["vision_encoder_name"].replace("/", "")
    text_model_name = cfg["text_encoder_name"].replace("/", "_")
    # Allow ablation configs to set an explicit `run_name` so each variant lands in its own
    # checkpoint subdirectory; falls back to the model name when unset (legacy behavior).
    run_name = cfg.get("run_name") or f"{cfg['model']}"
    exp_setup = setup_experiment(
        run_name,
        cfg,
        base_dir=cfg.get("base_dir", "experiments"),
        resume_from_checkpoint=args.resume_from,
    )

    # Use existing W&B run ID if resuming, otherwise use provided ID or create new
    wandb_run_id = (
        getattr(args, "id", None)
        if hasattr(args, "id")
        else exp_setup.get("wandb_run_id")
    )

    wandb_logger = WandbLogger(
        entity="smartxray",
        project="xray-vlp-stage2",
        name=f"{run_name}_v{exp_setup['version']}",
        id=wandb_run_id,
        log_model="best",
        save_dir=str(exp_setup["experiment_dir"]),
        settings=wandb.Settings(silent=True),
    )

    tb_logger = TensorBoardLogger(
        save_dir=str(exp_setup["tb_dir"]), name="", version=""
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(exp_setup["checkpoint_dir"]),
        filename="{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val/total_loss",
        mode="min",
        save_last=True,
        every_n_epochs=1,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # early_stop_cb = EarlyStopping(
    #     monitor="train/total_loss",
    #     mode="min",
    #     patience=10,
    #     verbose=True,
    #     min_delta=0.001,
    # )

    # ---- data ----
    num_workers = cfg.get("num_workers", 12)
    prefetch_factor = cfg.get("prefetch_factor", 2)
    smoke_test = bool(cfg.get("smoke_test", False))
    smoke_train_n = cfg.get("smoke_test_max_train_samples", 100) if smoke_test else None
    smoke_val_n = cfg.get("smoke_test_max_val_samples", 100) if smoke_test else None
    if smoke_test:
        print_rank_zero(
            f"SMOKE TEST: train max_samples/epoch={smoke_train_n}, val max_samples/epoch={smoke_val_n}"
        )

    train_dataloader = build_combined_concat_dataloader(
        chexpert_data_dir=cfg.get("chexpert_data_dir", None),
        padchest_data_dir=cfg.get("padchest_data_dir", None),
        mimic_data_dir=cfg.get("mimic_data_dir", None),
        private_data_dir=cfg.get("private_data_dir", None),
        rexgradient_data_dir=cfg.get("rexgradient_data_dir", None),
        split="train",
        region=cfg["region"],
        text=True,
        img_size=cfg["img_size"],
        in_channels=cfg["in_chans"],
        batch_size=cfg["batch_size"],
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        max_views=cfg["max_views"],
        max_regions=cfg["max_regions"],
        expand_samples=cfg.get("expand_samples", False),
        vision_encoder_name=cfg.get("vision_encoder_name", None),
        filter_top_num_image_pct=cfg.get("filter_top_num_image_pct", None),
        max_num_image=cfg.get("max_num_image", 10),
        max_samples=smoke_train_n,
        data_fraction=cfg.get("data_fraction", 1.0),
        data_fraction_seed=cfg.get("data_fraction_seed", 42),
    )
    # train_dataloader = build_weighted_combined_dataloader_each(
    #     chexpert_data_dir=cfg.get("chexpert_data_dir", None),
    #     padchest_data_dir=cfg.get("padchest_data_dir", None),
    #     mimic_data_dir=cfg.get("mimic_data_dir", None),
    #     private_data_dir=cfg.get("private_data_dir", None),
    #     rexgradient_data_dir=cfg.get("rexgradient_data_dir", None),
    #     region=cfg["region"].copy(),
    #     weights="uniform",
    #     split="train",
    #     text=True,
    #     img_size=cfg["img_size"],
    #     batch_size=cfg["batch_size"],
    #     num_workers=8,
    #     max_views=cfg["max_views"],
    #     max_regions=cfg["max_regions"],
    # )
    
    # val_dataloader = build_weighted_combined_dataloader_each(
    #     chexpert_data_dir=cfg.get("chexpert_data_dir", None),
    #     padchest_data_dir=cfg.get("padchest_data_dir", None),
    #     mimic_data_dir=cfg.get("mimic_data_dir", None),
    #     private_data_dir=cfg.get("private_data_dir", None),
    #     rexgradient_data_dir=cfg.get("rexgradient_data_dir", None),
    #     region=cfg["region"],
    #     weights="uniform",
    #     split="validate",
    #     text=True,
    #     img_size=cfg["img_size"],
    #     batch_size=cfg["batch_size"],
    #     num_workers=8,
    #     max_views=cfg["max_views"],
    #     max_regions=cfg["max_regions"],
    # )

    val_dataloader = build_combined_concat_dataloader(
        chexpert_data_dir=cfg.get("chexpert_data_dir", None),
        padchest_data_dir=None,
        mimic_data_dir=cfg.get("mimic_data_dir", None),
        private_data_dir=cfg.get("private_data_dir", None),
        rexgradient_data_dir=cfg.get("rexgradient_data_dir", None),
        split="validate",
        region=cfg["region"],
        text=True,
        img_size=cfg["img_size"],
        in_channels=cfg["in_chans"],
        batch_size=cfg["batch_size"],
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        max_views=cfg["max_views"],
        max_regions=cfg["max_regions"],
        expand_samples=cfg.get("expand_samples", False),
        vision_encoder_name=cfg.get("vision_encoder_name", None),
        filter_top_num_image_pct=cfg.get("filter_top_num_image_pct", None),
        max_num_image=cfg.get("max_num_image", 10),
        max_samples=smoke_val_n,
    )
    # Calculate total steps for learning rate scheduling
    print_rank_zero(f"Train dataloader length: {len(train_dataloader):,}")
    print_rank_zero(f"World size: {world_size}")
    print_rank_zero(f"Accumulate grad batches: {cfg['accumulate_grad_batches']}")
    print_rank_zero(f"Max epochs: {cfg['max_epochs']}")
    
    cfg["total_steps"] = (len(train_dataloader) // int(world_size) ) // cfg["accumulate_grad_batches"] * cfg["max_epochs"]
    cfg["warmup_steps"] = (len(train_dataloader) // int(world_size) ) // cfg["accumulate_grad_batches"] * cfg["warmup_epochs"]
    print_rank_zero(f"Warmup steps: {cfg['warmup_steps']}")
    print_rank_zero(f"Total steps: {cfg['total_steps']}")

    # Step-match (ablation only): when total_steps_target is set, scale max_epochs /
    # warmup_epochs / projection_warmup_epochs / encoder_warmup_epochs by the SAME
    # multiplier so absolute step counts per stage match the Full baseline.
    # Works in both directions (scale up when data is scarce, scale down when max_epochs
    # overshoots). Default behavior (no knob in YAML) is preserved exactly.
    target = cfg.get("total_steps_target", None)
    if target is not None and int(target) > 0:
        target = int(target)
        current_total = max(1, int(cfg["total_steps"]))
        if target != current_total:
            multiplier = float(target) / float(current_total)
            old = {
                "max_epochs": cfg["max_epochs"],
                "warmup_epochs": cfg["warmup_epochs"],
                "projection_warmup_epochs": cfg.get("projection_warmup_epochs", 0),
                "encoder_warmup_epochs": cfg.get("encoder_warmup_epochs", cfg["warmup_epochs"]),
            }
            scaled_max_epochs = old["max_epochs"] * multiplier
            cfg["max_epochs"] = max(
                1,
                int(math.ceil(scaled_max_epochs))
                if multiplier > 1.0
                else int(round(scaled_max_epochs)),
            )
            cfg["warmup_epochs"] = max(0, int(round(old["warmup_epochs"] * multiplier)))
            cfg["projection_warmup_epochs"] = max(
                0, int(round(old["projection_warmup_epochs"] * multiplier))
            )
            cfg["encoder_warmup_epochs"] = max(
                0, int(round(old["encoder_warmup_epochs"] * multiplier))
            )
            spe = max(1, (len(train_dataloader) // int(world_size)) // cfg["accumulate_grad_batches"])
            cfg["total_steps"] = spe * cfg["max_epochs"]
            cfg["warmup_steps"] = spe * cfg["warmup_epochs"]
            direction = "up" if multiplier > 1.0 else "down"
            print_rank_zero(
                f"[STEP-MATCH] scale {direction}: total_steps_target={target}, "
                f"current={current_total} → multiplier={multiplier:.4f}"
            )
            print_rank_zero(
                f"[STEP-MATCH] max_epochs: {old['max_epochs']} → {cfg['max_epochs']}, "
                f"warmup_epochs: {old['warmup_epochs']} → {cfg['warmup_epochs']}, "
                f"projection_warmup_epochs: {old['projection_warmup_epochs']} → {cfg['projection_warmup_epochs']}, "
                f"encoder_warmup_epochs: {old['encoder_warmup_epochs']} → {cfg['encoder_warmup_epochs']}"
            )
            print_rank_zero(
                f"[STEP-MATCH] new total_steps: {cfg['total_steps']}, new warmup_steps: {cfg['warmup_steps']}"
            )
        else:
            print_rank_zero(
                f"[STEP-MATCH] total_steps_target={target} already matches total_steps={current_total}; no scaling."
            )
    # ---- model ----
    if cfg["model"] != "topk":
        raise ValueError("SmartX release pretraining supports only model=topk")
    model = OursTopKWrapper(cfg)

    # ---- trainer ----
    # DDP notes:
    # - Some timm backbones (e.g. Swin with use_checkpoint=True used by ark_*) rely on
    #   re-entrant activation checkpointing, which can trigger
    #   "Expected to mark a variable ready only once" under DDP unless static_graph=True.
    # - The TopK "projection warmup" toggles requires_grad of encoder params across epochs,
    #   but the forward graph itself stays the same; static_graph is still safe here and
    #   avoids the DDP+checkpointing runtime error.
    if torch.cuda.device_count() > 1:
        find_unused = bool(cfg.get("ddp_find_unused_parameters", False))
        # Match OursTopKWrapper default: topk uses projection_warmup_epochs default 1 if omitted.
        projection_warmup = cfg.get(
            "projection_warmup_epochs", 1 if cfg["model"] == "topk" else 0
        )
        # Prefer static_graph for ark_* (Swin) to prevent DDP+checkpointing error.
        vision_name = str(cfg.get("vision_encoder_name", "")).lower()
        force_static_graph = vision_name in ["ark_base", "ark_large"]
        use_static_graph = True
        if force_static_graph:
            print_rank_zero("DDP: static_graph=True (ark_* uses activation checkpointing).")
        elif cfg["model"] == "topk" and projection_warmup > 0:
            print_rank_zero("DDP: static_graph=True (topk projection warmup enabled).")
        ddp_strategy = DDPStrategy(
            static_graph=use_static_graph,
            find_unused_parameters=find_unused,
        )
    else:
        ddp_strategy = "auto"

    num_nodes = int(os.environ.get("SLURM_NNODES", 1))

    trainer = pl.Trainer(
        accumulate_grad_batches=cfg["accumulate_grad_batches"],
        accelerator="auto",
        devices="auto",
        num_nodes=num_nodes,
        strategy=ddp_strategy,
        precision="16-mixed",
        max_epochs=cfg["max_epochs"],
        logger=[wandb_logger, tb_logger],
        callbacks=[checkpoint_cb, lr_monitor],  # , early_stop_cb],
        log_every_n_steps=cfg["log_every_n_steps"],
        gradient_clip_val=cfg.get("gradient_clip_val", None),
        deterministic=False,
    )

    # Save W&B run ID for future resuming (only for new experiments)
    if not exp_setup.get("is_resume", False) and wandb_logger.experiment:
        actual_run_id = wandb_logger.experiment.id
        save_wandb_run_id(exp_setup["experiment_dir"], actual_run_id)

    # Sync all ranks before fit() so rank 0 has finished any slow I/O (NFS, wandb, etc.)
    if "RANK" in os.environ:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()

    # ---- training ----
    try:
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=args.resume_from,
        )
    finally:
        # Clean up distributed process group
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            import torch.distributed as dist
            if dist.is_initialized():
                rank = dist.get_rank() if dist.is_initialized() else 0
                dist.destroy_process_group()
                print(f"[Rank {rank}] Process group destroyed successfully")


if __name__ == "__main__":
    main()
