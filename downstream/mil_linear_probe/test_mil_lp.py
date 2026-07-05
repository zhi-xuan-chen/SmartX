"""
MIL 线性探针测试脚本。

用法示例：
    python test_mil_lp.py \
        --config configs/base_mil_config.yaml \
        --dataset_config configs/dataset_dir.yaml \
        --features_dir /path/to/features/private_a \
        --ckpt path/to/best.ckpt \
        --split validate
"""

import argparse
import warnings
import sys
from pathlib import Path
import json
from datetime import datetime

warnings.filterwarnings("ignore")

import torch
import pytorch_lightning as pl

sys.path.append(str(Path(__file__).parent.parent))
from utils import load_config, get_eval_dir
from trainer_mil_lp import MILLinearProbe
from data_utils import (
    DISEASE_LIST,
    load_metadata,
    infer_model_id,
    build_private_base_dataset,
    MILFeatureDataset,
    mil_collate_fn,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test MIL Linear Probe")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="configs/base_mil_config.yaml",
        help="训练/模型配置",
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
        "--ckpt",
        type=str,
        required=True,
        help="训练好的 checkpoint 路径",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validate",
        choices=["validate", "test", "val"],
        help="评估划分",
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
        "--output_dir",
        type=str,
        default=None,
        help="结果保存目录，默认跟随 checkpoint",
    )
    return parser.parse_args()


def normalize_split(split: str) -> str:
    return "validate" if split in ["val", "validate"] else "test"


def normalize_region(cfg_region, arg_region):
    """测试阶段也要求显式指定单个 region。"""
    if arg_region:
        return [arg_region]
    if cfg_region is None:
        raise ValueError("请通过 --region 或配置文件中的 region 指定单个部位。")
    if isinstance(cfg_region, str):
        return [cfg_region]
    if isinstance(cfg_region, (list, tuple)) and len(cfg_region) == 1:
        return list(cfg_region)
    raise ValueError(f"当前仅支持单个 region 测试，但收到: {cfg_region}")


def build_loader(
    cfg,
    dataset_cfg,
    features_dir,
    model_id,
    regions,
    split,
    disease_cols,
    feature_dim,
    preload_features=True,
):
    dataset_name = cfg["dataset_name"]
    data_dir = dataset_cfg[dataset_name]["dataset_dir"]
    max_views = cfg.get("max_views", 12)
    batch_size = cfg.get("batch_size", 8)
    num_workers = cfg.get("num_workers", 8)
    img_size = cfg.get("img_size", 512)
    in_channels = cfg.get("in_channels", 1)

    base_ds = build_private_base_dataset(
        data_dir, split, img_size, in_channels, regions, max_views
    )

    # 测试时也启用特征预加载以加速
    loader_num_workers = 0 if preload_features else num_workers
    
    dataset = MILFeatureDataset(
        base_ds,
        features_root=features_dir,
        model_id=model_id,
        max_views=max_views,
        disease_cols=disease_cols,
        feature_dim=feature_dim,
        preload_features=preload_features,
        split=split,
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=loader_num_workers,
        pin_memory=True,
        collate_fn=mil_collate_fn,
    )
    return loader


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dataset_cfg = load_config(args.dataset_config)
    cfg["dataset_name"] = cfg.get("dataset_name", "private_a").lower()

    split = normalize_split(args.split)
    regions = normalize_region(cfg.get("region"), args.region)
    metadata = load_metadata(args.features_dir)
    model_id = args.model_id or infer_model_id(Path(args.features_dir), regions)
    if model_id is None:
        raise ValueError("无法自动推断 model_id，请通过 --model_id 指定。")

    # 先从 checkpoint 中恢复模型与训练时使用的疾病标签子集 / 特征维度
    print(f"\n从 {args.ckpt} 加载模型...")
    model = MILLinearProbe.load_from_checkpoint(args.ckpt)

    # disease_cols 兼容字典或属性两种保存方式
    disease_cols = None
    try:
        if hasattr(model, "hparams") and isinstance(model.hparams, dict):
            disease_cols = model.hparams.get("disease_cols", None)
        elif hasattr(model, "hparams"):
            disease_cols = getattr(model.hparams, "disease_cols", None)
    except Exception:
        disease_cols = None
    if disease_cols is None:
        disease_cols = DISEASE_LIST
        print(
            "警告: 从 checkpoint 中未找到 disease_cols，回退为全局 DISEASE_LIST，"
            "请确认训练与测试时标签顺序一致。"
        )

    feature_dim = getattr(model, "feature_dim", None)
    if feature_dim is None:
        raise ValueError("从模型中未能恢复 feature_dim，请检查 checkpoint。")

    # 默认不预加载，如需加速可在配置中显式设置 preload_features: true
    preload_features = cfg.get("preload_features", False)
    loader = build_loader(
        cfg,
        dataset_cfg,
        Path(args.features_dir),
        model_id,
        regions,
        split,
        disease_cols=disease_cols,
        feature_dim=feature_dim,
        preload_features=preload_features,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        logger=False,
    )

    print(f"\n评估划分: {split}")
    results = trainer.test(model, loader)

    # 保存 JSON 结果
    eval_dir = get_eval_dir(args.ckpt, args.output_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    results_file = eval_dir / f"{split}_results.json"
    payload = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(args.ckpt),
        "features_dir": str(args.features_dir),
        "split": split,
        "results": results[0] if results else {},
    }
    with open(results_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n结果已保存: {results_file}")


if __name__ == "__main__":
    main()

