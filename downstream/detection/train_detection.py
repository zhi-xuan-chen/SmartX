#!/usr/bin/env python3
"""Train or evaluate SmartX-backed VitDet detectors on COCO-format datasets."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_CONFIG = "downstream/detection/configs/detection.example.yaml"
DEFAULT_LAZY_CONFIG = "downstream/detection/configs/vitdet_xray_base.py"


def load_yaml_config(path: str | Path) -> dict:
    import yaml

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a mapping: {config_path}")
    return _expand_env(payload)


def _expand_env(value):
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_env(item) for item in value)
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def resolve_path(value: str | Path | None, base_dir: str | Path | None = None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path


def optional_path_value(value: str | Path | None) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text or "$" in text:
        return None
    return text


def load_detection_config(path: str | Path) -> dict:
    config = load_yaml_config(path)
    dataset_cfg = config.get("dataset", {})
    if not dataset_cfg.get("name"):
        raise ValueError("Detection config requires dataset.name")
    if not dataset_cfg.get("splits"):
        raise ValueError("Detection config requires dataset.splits")
    return config


def _optional_detection_error(exc: Exception) -> RuntimeError:
    return RuntimeError(
        "Detection requires optional dependencies: detectron2, fvcore, "
        "omegaconf, and pycocotools. See downstream/detection/README.md."
    )


def _split_configs(config: dict) -> dict:
    dataset_cfg = config["dataset"]
    splits = {}
    for split, split_cfg in dataset_cfg["splits"].items():
        annotation = resolve_path(split_cfg["annotation"], Path.cwd())
        image_root = resolve_path(split_cfg["image_root"], Path.cwd())
        splits[split] = {
            "annotation": str(annotation),
            "image_root": str(image_root),
        }
    return splits


def register_datasets(config: dict, force: bool = True) -> list[str]:
    try:
        from downstream.detection.xray_vitdet.data.register_datasets import register_coco_splits
    except ImportError as exc:
        raise _optional_detection_error(exc)

    dataset_cfg = config["dataset"]
    return register_coco_splits(
        dataset_name=dataset_cfg["name"],
        splits=_split_configs(config),
        class_names=dataset_cfg.get("class_names", []),
        force=force,
    )


def _set_image_augmentations(lazy_cfg, image_size: int, resize_mode: str) -> None:
    try:
        import detectron2.data.transforms as T
        from detectron2.config import LazyCall as L
        from downstream.detection.xray_vitdet.data.transforms import CenterCropFixedSize
    except ImportError as exc:
        raise _optional_detection_error(exc)

    if resize_mode in {"stretch", "warp"}:
        lazy_cfg.dataloader.train.mapper.augmentations = [
            L(T.RandomFlip)(horizontal=True),
            L(T.ResizeScale)(
                min_scale=0.8,
                max_scale=1.2,
                target_height=image_size,
                target_width=image_size,
            ),
            L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=True),
        ]
        lazy_cfg.dataloader.test.mapper.augmentations = [L(T.Resize)(shape=(image_size, image_size))]
    elif resize_mode in {"shortedge_crop", "shortedge-crop"}:
        lazy_cfg.dataloader.train.mapper.augmentations = [
            L(T.RandomFlip)(horizontal=True),
            L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=sys.maxsize),
            L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=True),
        ]
        lazy_cfg.dataloader.test.mapper.augmentations = [
            L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=sys.maxsize),
            L(CenterCropFixedSize)(crop_size=(image_size, image_size), pad=True),
        ]
    else:
        raise ValueError("model.resize_mode must be stretch, warp, or shortedge_crop")


def build_lazy_config(config: dict, command: str, opts: list[str] | None = None):
    try:
        from detectron2.config import LazyCall as L, LazyConfig
        from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
        from downstream.detection.xray_vitdet.modeling.smartx_swin import SmartXSwinBackbone
    except ImportError as exc:
        raise _optional_detection_error(exc)

    model_cfg = config.get("model", {})
    dataset_cfg = config["dataset"]
    training_cfg = config.get("training", {})
    lazy_path = model_cfg.get("lazy_config", DEFAULT_LAZY_CONFIG)
    lazy_cfg = LazyConfig.load(str(resolve_path(lazy_path, Path.cwd())))
    if opts:
        lazy_cfg = LazyConfig.apply_overrides(lazy_cfg, opts)

    checkpoint = optional_path_value(model_cfg.get("checkpoint"))
    lazy_cfg.model.backbone = L(FPN)(
        bottom_up=L(SmartXSwinBackbone)(
            checkpoint_path=checkpoint or None,
            repo_id=model_cfg.get("repo_id", "Trusure/SmartX"),
            filename=model_cfg.get("filename", "smartx_large.ckpt"),
            freeze=bool(model_cfg.get("freeze_backbone", False)),
        ),
        in_features=["res2", "res3", "res4", "res5"],
        out_channels=256,
        norm="LN",
        top_block=L(LastLevelMaxPool)(),
        fuse_type="sum",
    )

    image_size = int(model_cfg.get("image_size", 768))
    _set_image_augmentations(lazy_cfg, image_size, model_cfg.get("resize_mode", "stretch"))
    lazy_cfg.model.pixel_mean = model_cfg.get("pixel_mean", [123.675, 116.280, 103.530])
    lazy_cfg.model.pixel_std = model_cfg.get("pixel_std", [58.395, 57.120, 57.375])
    lazy_cfg.model.roi_heads.num_classes = int(dataset_cfg["num_classes"])

    dataset_name = dataset_cfg["name"]
    train_split = dataset_cfg.get("train_split", "train")
    val_split = dataset_cfg.get("val_split", "val")
    test_split = dataset_cfg.get("test_split", "test")
    lazy_cfg.dataloader.train.dataset.names = f"{dataset_name}_{train_split}"
    lazy_cfg.dataloader.val = copy.deepcopy(lazy_cfg.dataloader.test)
    lazy_cfg.dataloader.val.dataset.names = f"{dataset_name}_{val_split}"
    lazy_cfg.dataloader.test.dataset.names = f"{dataset_name}_{test_split}"

    lazy_cfg.train.output_dir = str(resolve_path(training_cfg.get("output_dir", "outputs/detection"), Path.cwd()))
    lazy_cfg.train.init_checkpoint = ""
    lazy_cfg.train.max_iter = int(training_cfg.get("max_iter", 10000))
    lazy_cfg.train.eval_period = int(training_cfg.get("eval_period", 1000))
    lazy_cfg.train.log_period = int(training_cfg.get("log_period", 50))
    lazy_cfg.train.checkpointer.period = int(training_cfg.get("checkpoint_period", 1000))
    lazy_cfg.train.checkpointer.max_to_keep = int(training_cfg.get("max_to_keep", 2))
    lazy_cfg.train.amp.enabled = bool(training_cfg.get("amp", False))
    lazy_cfg.train.device = training_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    lazy_cfg.dataloader.train.total_batch_size = int(training_cfg.get("batch_size", 2))
    lazy_cfg.dataloader.train.num_workers = int(training_cfg.get("num_workers", 4))
    lazy_cfg.dataloader.test.num_workers = int(training_cfg.get("num_workers", 4))
    lazy_cfg.dataloader.val.num_workers = int(training_cfg.get("num_workers", 4))
    lazy_cfg.optimizer.lr = float(training_cfg.get("lr", 0.0001))
    if hasattr(lazy_cfg, "lr_multiplier") and hasattr(lazy_cfg.lr_multiplier, "warmup_length"):
        lazy_cfg.lr_multiplier.warmup_length = 250 / max(1, int(lazy_cfg.train.max_iter))

    if command == "test":
        lazy_cfg.train.eval_period = 0
    os.makedirs(lazy_cfg.train.output_dir, exist_ok=True)
    return lazy_cfg


def _dataset_name(value):
    return value[0] if isinstance(value, (list, tuple)) else value


def _run_eval(lazy_cfg, model, split: str, loader_cfg=None):
    try:
        from detectron2.config import instantiate
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
    except ImportError as exc:
        raise _optional_detection_error(exc)

    loader_cfg = loader_cfg or lazy_cfg.dataloader.test
    dataset_name = _dataset_name(loader_cfg.dataset.names)
    out_dir = Path(lazy_cfg.train.output_dir) / f"inference_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_loader = instantiate(loader_cfg)
    evaluator = COCOEvaluator(dataset_name, output_dir=str(out_dir))
    results = inference_on_dataset(model, data_loader, evaluator)
    print_csv_format(results)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def do_train(args, lazy_cfg) -> None:
    try:
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import instantiate
        from detectron2.engine import AMPTrainer, SimpleTrainer, default_writers, hooks
        from detectron2.engine.defaults import create_ddp_model
        from detectron2.utils import comm
    except ImportError as exc:
        raise _optional_detection_error(exc)

    model = instantiate(lazy_cfg.model)
    model.to(lazy_cfg.train.device)
    lazy_cfg.optimizer.params.model = model
    optimizer = instantiate(lazy_cfg.optimizer)
    train_loader = instantiate(lazy_cfg.dataloader.train)
    model = create_ddp_model(model, **lazy_cfg.train.ddp)
    trainer_cls = AMPTrainer if lazy_cfg.train.amp.enabled else SimpleTrainer
    trainer = trainer_cls(model, train_loader, optimizer)
    checkpointer = DetectionCheckpointer(model, lazy_cfg.train.output_dir, trainer=trainer)

    trainer_hooks = [
        hooks.IterationTimer(),
        hooks.LRScheduler(scheduler=instantiate(lazy_cfg.lr_multiplier)),
    ]
    if comm.is_main_process():
        trainer_hooks.append(hooks.PeriodicCheckpointer(checkpointer, **lazy_cfg.train.checkpointer))
    if int(lazy_cfg.train.eval_period) > 0:
        trainer_hooks.append(
            hooks.EvalHook(
                int(lazy_cfg.train.eval_period),
                lambda: _run_eval(lazy_cfg, model, "val", lazy_cfg.dataloader.val),
            )
        )
        if comm.is_main_process():
            trainer_hooks.append(
                hooks.BestCheckpointer(
                    int(lazy_cfg.train.eval_period),
                    checkpointer,
                    val_metric="bbox/AP",
                    mode="max",
                    file_prefix="model_best",
                )
            )
    if comm.is_main_process():
        trainer_hooks.append(
            hooks.PeriodicWriter(
                default_writers(lazy_cfg.train.output_dir, int(lazy_cfg.train.max_iter)),
                period=int(lazy_cfg.train.log_period),
            )
        )
    trainer.register_hooks(trainer_hooks)
    checkpointer.resume_or_load(lazy_cfg.train.init_checkpoint, resume=args.resume)
    start_iter = trainer.iter + 1 if args.resume and checkpointer.has_checkpoint() else 0
    trainer.train(start_iter, int(lazy_cfg.train.max_iter))

    best_path = Path(lazy_cfg.train.output_dir) / "model_best.pth"
    if best_path.exists():
        checkpointer.load(str(best_path))
    results = _run_eval(lazy_cfg, model, "test", lazy_cfg.dataloader.test)
    if comm.is_main_process():
        (Path(lazy_cfg.train.output_dir) / "final_test_results.json").write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )


def do_test(args, lazy_cfg) -> None:
    try:
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import instantiate
        from detectron2.engine.defaults import create_ddp_model
        from detectron2.utils import comm
    except ImportError as exc:
        raise _optional_detection_error(exc)

    model = instantiate(lazy_cfg.model)
    model.to(lazy_cfg.train.device)
    model = create_ddp_model(model, **lazy_cfg.train.ddp)
    DetectionCheckpointer(model).load(args.checkpoint)
    results = _run_eval(lazy_cfg, model, "test", lazy_cfg.dataloader.test)
    if comm.is_main_process():
        (Path(lazy_cfg.train.output_dir) / "test_results.json").write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )


def _worker(args) -> None:
    try:
        from detectron2.engine import launch
        from detectron2.utils.logger import setup_logger
    except ImportError as exc:
        raise _optional_detection_error(exc)

    config = load_detection_config(args.config)
    register_datasets(config, force=True)
    lazy_cfg = build_lazy_config(config, args.command, args.opts)
    setup_logger(output=lazy_cfg.train.output_dir, name="detectron2")
    setup_logger(output=lazy_cfg.train.output_dir, name="smartx_detection")
    if args.command == "train":
        do_train(args, lazy_cfg)
    elif args.command == "test":
        do_test(args, lazy_cfg)
    else:
        raise ValueError(f"Unknown command: {args.command}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmartX VitDet train/test")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser):
        subparser.add_argument("--config", default=DEFAULT_CONFIG)
        subparser.add_argument("--num-gpus", type=int, default=1)
        subparser.add_argument("--num-machines", type=int, default=1)
        subparser.add_argument("--machine-rank", type=int, default=0)
        subparser.add_argument("--dist-url", default="auto")
        subparser.add_argument("--opts", nargs="*", default=[])

    train_parser = subparsers.add_parser("train")
    add_common(train_parser)
    train_parser.add_argument("--resume", action="store_true")

    test_parser = subparsers.add_parser("test")
    add_common(test_parser)
    test_parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from detectron2.engine import launch
    except ImportError as exc:
        raise _optional_detection_error(exc)
    launch(
        _worker,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    main()
