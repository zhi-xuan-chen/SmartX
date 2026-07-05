"""
基于已提取特征的 MIL 线性探针训练脚本（ABMIL 聚合）。

用法示例：
    python train_mil_lp.py \
        --config configs/base_mil_config.yaml \
        --dataset_config configs/dataset_dir.yaml \
        --features_dir /path/to/features/private_a \
        --exp_root experiments_mil
"""

import os
import argparse
import warnings
import sys
from pathlib import Path
import json

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import torch

# 控制 CPU 线程，避免无谓占用
torch.set_num_threads(4)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning import seed_everything

sys.path.append(str(Path(__file__).parent.parent))
from utils import setup_experiment, load_config

from trainer_mil_lp import MILLinearProbe
from val_thresholds_utils import SaveValThresholdsCallback
from data_utils import (
    DISEASE_LIST,
    REGION_DISEASES,
    load_metadata,
    infer_model_id,
    build_private_base_dataset,
    MILFeatureDataset,
    mil_collate_fn,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MIL Linear Probe (ABMIL)")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="configs/base_mil_config.yaml",
        help="训练配置（YAML/JSON）",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=False,
        default="configs/dataset_dir.yaml",
        help="数据集目录/类别配置",
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        required=True,
        help="extract_features 输出根目录（含 metadata.json）",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="特征子目录名，若不指定将自动在 region 子目录内推断",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="指定单个部位，默认使用所有部位",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="checkpoint 路径，继续训练",
    )
    parser.add_argument(
        "--exp_root",
        type=str,
        default="${SMARTX_OUTPUT_ROOT}/experiments/MIL_LP",
        help="实验输出根目录",
    )
    # 简单覆盖
    parser.add_argument("--lr", type=float, help="覆盖学习率")
    parser.add_argument("--batch_size", type=int, help="覆盖 batch size")
    parser.add_argument("--max_epochs", type=int, help="覆盖最大 epoch")
    parser.add_argument("--weight_decay", type=float, help="覆盖权重衰减")
    parser.add_argument("--max_views", type=int, help="覆盖 bag 最大视角数")

    return parser.parse_args()


def normalize_region(cfg_region, arg_region):
    """当前设置下我们希望每次只训练单个 region。"""
    if arg_region:
        return [arg_region]
    if cfg_region is None:
        raise ValueError("请通过 --region 或配置文件中的 region 指定单个部位。")
    if isinstance(cfg_region, str):
        return [cfg_region]
    if isinstance(cfg_region, (list, tuple)) and len(cfg_region) == 1:
        return list(cfg_region)
    raise ValueError(f"当前仅支持单个 region 训练，但收到: {cfg_region}")


def create_datasets_and_loaders(cfg, dataset_cfg, features_dir, model_id, regions):
    dataset_name = cfg["dataset_name"]
    data_dir = dataset_cfg[dataset_name]["dataset_dir"]
    max_views = cfg.get("max_views", 6)
    batch_size = cfg["batch_size"]
    num_workers = cfg.get("num_workers", 8)
    img_size = cfg.get("img_size", 224)
    in_channels = cfg.get("in_channels", 1)

    if len(regions) != 1:
        raise ValueError(f"当前仅支持单个 region 训练，但收到 {regions}")
    region = regions[0]

    # 构建基础数据集（只解析 CSV 与路径）
    base_train = build_private_base_dataset(
        data_dir, "train", img_size, in_channels, regions, max_views
    )
    base_val = build_private_base_dataset(
        data_dir, "validate", img_size, in_channels, regions, max_views
    )
    base_test = build_private_base_dataset(
        data_dir, "test", img_size, in_channels, regions, max_views
    )

    # 标签列：按该 region 合理的候选标签子集筛选，并与实际 CSV 列求交集
    candidate_cols = REGION_DISEASES.get(region, DISEASE_LIST)
    available_cols = [c for c in candidate_cols if c in base_train.data_list.columns]
    if len(available_cols) == 0:
        raise ValueError(
            f"在 CSV 中未找到 region={region} 的任何候选疾病标签列，请检查标签文件与 REGION_DISEASES 设置。"
        )

    def drop_samples_with_any_na_labels(base_dataset, split_name: str):
        """
        严格标签过滤：只保留在当前 region 标签列中全部非 NA 的病例。
        同步更新 data_list / _rows_image_paths / _flat_image_paths，避免错位。
        """
        if len(base_dataset.data_list) != len(base_dataset._rows_image_paths):
            raise ValueError(
                f"{split_name} 数据不一致: data_list={len(base_dataset.data_list)}, "
                f"rows_image_paths={len(base_dataset._rows_image_paths)}"
            )

        keep_mask = ~base_dataset.data_list[available_cols].isna().any(axis=1)
        keep_indices = [i for i, keep in enumerate(keep_mask.tolist()) if keep]
        dropped = len(base_dataset.data_list) - len(keep_indices)

        if dropped > 0:
            print(
                f"{split_name}: 丢弃 {dropped} 个样本（当前 region 标签列存在 NA），"
                f"保留 {len(keep_indices)} 个样本。"
            )

        if len(keep_indices) == 0:
            raise ValueError(f"{split_name} 在严格标签过滤后无可用样本。")

        base_dataset.data_list = base_dataset.data_list.iloc[keep_indices].reset_index(drop=True)
        base_dataset._rows_image_paths = [base_dataset._rows_image_paths[i] for i in keep_indices]

        # 与 BaseDataset 的 image-only 模式保持一致，避免后续长度/索引不一致
        if hasattr(base_dataset, "_flat_image_paths") and base_dataset._flat_image_paths is not None:
            base_dataset._flat_image_paths = []
            for paths in base_dataset._rows_image_paths:
                base_dataset._flat_image_paths.extend(paths)

    # 只要当前 region 标签列中有任意 NA，就丢弃该样本
    drop_samples_with_any_na_labels(base_train, "train")
    drop_samples_with_any_na_labels(base_val, "validate")
    drop_samples_with_any_na_labels(base_test, "test")

    # 先在训练集推断特征维度，后续复用（不需要预加载，只是为了获取维度）
    feature_dim = MILFeatureDataset(
        base_train,
        features_root=features_dir,
        model_id=model_id,
        max_views=max_views,
        disease_cols=available_cols,
        preload_features=False,
    ).feature_dim

    def build_loader(base_dataset, split):
        # 默认不预加载特征，按需从磁盘读取；如需预加载可在配置中显式设置 preload_features: true
        preload_features = cfg.get("preload_features", False)
        loader_num_workers = 0 if preload_features else num_workers
        
        print(f"\n构建 {split} 数据集...")
        dataset = MILFeatureDataset(
            base_dataset,
            features_root=features_dir,
            model_id=model_id,
            max_views=max_views,
            disease_cols=available_cols,
            feature_dim=feature_dim,
            preload_features=preload_features,
            split=split,
        )
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=loader_num_workers,
            pin_memory=True,
            collate_fn=mil_collate_fn,
        )
        print(f"✓ {split} 数据集构建完成: {len(dataset):,} 个病例, batch_size={batch_size}, num_workers={loader_num_workers}")
        return loader

    loaders = {
        "train": build_loader(base_train, "train"),
        "validate": build_loader(base_val, "validate"),
        "test": build_loader(base_test, "test"),
    }
    return loaders, feature_dim, available_cols


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dataset_cfg = load_config(args.dataset_config)

    # 打印加载的配置信息
    print(f"\n加载配置文件: {args.config}")
    print(f"配置文件中的 batch_size: {cfg.get('batch_size', 'NOT SET')}")

    # 覆盖基础配置
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.batch_size is not None:
        print(f"命令行参数覆盖 batch_size: {args.batch_size}")
        cfg["batch_size"] = args.batch_size
    
    print(f"最终使用的 batch_size: {cfg['batch_size']}")
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    if args.weight_decay is not None:
        cfg["weight_decay"] = args.weight_decay
    if args.max_views is not None:
        cfg["max_views"] = args.max_views

    # 数据集信息
    cfg["dataset_name"] = cfg.get("dataset_name", "private_a").lower()

    regions = normalize_region(cfg.get("region"), args.region)
    model_id = args.model_id or infer_model_id(Path(args.features_dir), regions)
    if model_id is None:
        raise ValueError("无法自动推断 model_id，请通过 --model_id 指定。")
    print(f"使用 model_id: {model_id}")
    print(f"部位列表: {regions}")

    # 构建数据集 + dataloader（会预加载特征到内存）
    print("\n" + "="*60)
    print("构建数据集并预加载特征...")
    print("="*60)
    loaders, feature_dim, disease_cols = create_datasets_and_loaders(
        cfg, dataset_cfg, Path(args.features_dir), model_id, regions
    )
    cfg["feature_dim"] = feature_dim
    cfg["disease_cols"] = disease_cols
    cfg["num_classes"] = len(disease_cols)
    print(f"使用的疾病标签列（{len(disease_cols)} 个）: {disease_cols}")
    print("="*60 + "\n")

    # 打印数据集和batch信息
    train_dataset_size = len(loaders["train"].dataset)
    train_batches_per_epoch = len(loaders["train"])
    print(f"\n数据集信息:")
    print(f"  训练集大小 (cases): {train_dataset_size:,}")
    print(f"  Batch size: {cfg['batch_size']}")
    print(f"  每个epoch的batch数: {train_batches_per_epoch:,}")
    print(f"  验证集大小 (cases): {len(loaders['validate'].dataset):,}")
    print(f"  测试集大小 (cases): {len(loaders['test'].dataset):,}")

    # 计算 scheduler 步数
    cfg["total_steps"] = len(loaders["train"]) * cfg["max_epochs"]
    cfg["warmup_steps"] = len(loaders["train"]) * cfg.get("warmup_epochs", 5)
    print(f"  总训练步数 (total_steps): {cfg['total_steps']:,} ({train_batches_per_epoch:,} batches/epoch × {cfg['max_epochs']} epochs)")

    seed_everything(cfg.get("seed", 42), workers=True)

    # 运行名：在 dataset_name 后面拼接当前 region，便于区分不同部位的模型
    lr = cfg.get("lr", 1e-3)
    mil_tag = cfg.get("mil_type", "clam")
    region_tag = regions[0] if isinstance(regions, (list, tuple)) and len(regions) > 0 else "all"
    dataset_with_region = f"{cfg['dataset_name']}_{region_tag}"
    run_name = f"{dataset_with_region}/{model_id}/bs{cfg['batch_size']}_lr{lr}_{mil_tag}"

    # 实验目录
    exp_setup = setup_experiment(run_name, cfg, base_dir=args.exp_root)

    # 日志
    wandb_logger = WandbLogger(
        entity="smartxray",
        project="xray-mil-probe",
        name=f"{run_name}_v{exp_setup['version']}",
        log_model="best",
        save_dir=str(exp_setup["experiment_dir"]),
    )
    tb_logger = TensorBoardLogger(str(exp_setup["tb_dir"]), name="", version="")

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(exp_setup["checkpoint_dir"]),
        filename="{epoch:02d}-{val_custom_auc:.4f}",
        save_top_k=1,
        monitor="val/custom_auc",
        mode="max",
        save_last=True,
        every_n_epochs=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    save_val_thresholds_cb = SaveValThresholdsCallback()

    model = MILLinearProbe(cfg)

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        max_epochs=cfg["max_epochs"],
        logger=[wandb_logger, tb_logger],
        callbacks=[checkpoint_cb, lr_monitor, save_val_thresholds_cb],
        log_every_n_steps=10,
        gradient_clip_val=cfg.get("gradient_clip_val", 1.0),
        deterministic=True,
        check_val_every_n_epoch=1,
    )

    print("\n开始训练 MIL 线性探针...")
    print(f"  特征维度: {cfg['feature_dim']}")
    print(f"  类别数: {cfg['num_classes']}")
    print(f"  学习率: {cfg['lr']}")
    print(f"  batch size: {cfg['batch_size']}")
    print(f"  max_views: {cfg.get('max_views', 12)}")

    trainer.fit(
        model=model,
        train_dataloaders=loaders["train"],
        val_dataloaders=loaders["validate"],
        ckpt_path=args.resume_from,
    )

    print("\n训练完成")
    print(f"最佳 checkpoint: {checkpoint_cb.best_model_path}")

    if checkpoint_cb.best_model_path:
        print("\n使用最佳 checkpoint 在测试集评估...")
        # 设置测试结果保存路径，供 Lightning 模型在 on_test_epoch_end 写出逐样本结果
        model.results_save_path = exp_setup["experiment_dir"] / "test_results.csv"
        test_results = trainer.test(
            model=model,
            dataloaders=loaders["test"],
            ckpt_path=checkpoint_cb.best_model_path,
        )
        print(f"测试结果: {test_results}")
    else:
        print("未找到最佳 checkpoint，跳过测试评估。")


if __name__ == "__main__":
    main()

