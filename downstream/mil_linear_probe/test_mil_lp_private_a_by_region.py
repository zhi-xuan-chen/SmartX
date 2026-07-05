"""
Private A 按 Region 批量重跑 test split，并覆盖各实验 version 目录下的 test_results.csv。

典型用途：更新 val_thresholds.json 后，用新阈值重新生成 in-domain 测试结果，
供 statistical_analysis/bootstrap.py 等脚本读取。

每个 Region：
  1) 加载对应 best checkpoint
  2) 在 private_a test split 上推理
  3) 从 ckpt 旁 val_thresholds.json 读取阈值做二值化
  4) 写回 {version_*}/test_results.csv（与 train_mil_lp.py 一致）
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
import sys

import pytorch_lightning as pl
import torch

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent.parent))
from utils import load_config
from trainer_mil_lp import MILLinearProbe
from data_utils import (
    DISEASE_LIST,
    build_private_base_dataset,
    MILFeatureDataset,
    mil_collate_fn,
)

DEFAULT_REGIONS = [
    "Chest",
    "Pelvis",
    "Wrist",
    "Metacarpal",
    "Lumbar_spine",
    "Cervical_spine",
    "Foot",
    "Ankle",
    "Abdominal",
    "Knee",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Region-wise Private A MIL test (in-domain)")
    p.add_argument(
        "--config",
        type=str,
        default="configs/train_private_a_our_ark_large.yaml",
        help="训练同款 yaml（img_size / batch_size / max_views 等）",
    )
    p.add_argument("--dataset_config", type=str, default="configs/dataset_dir.yaml")
    p.add_argument(
        "--features_dir",
        type=str,
        required=True,
        help="private_a 特征根目录",
    )
    p.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="特征子目录名，例如 topk_v19_last",
    )
    p.add_argument(
        "--region_ckpt_map",
        type=str,
        required=True,
        help="region -> ckpt 路径 JSON（可由 build_region_ckpt_map.py 生成）",
    )
    p.add_argument(
        "--regions",
        type=str,
        nargs="*",
        default=None,
        help="只跑指定 region；默认跑 JSON 中的全部 region",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "validate", "val"],
        help="评估划分，默认 test",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="推理 batch size，默认跟随 config（训练常为 4096）",
    )
    p.add_argument(
        "--preload_features",
        type=str,
        default=None,
        help="true/false；默认跟随 config",
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default="experiment",
        choices=["experiment", "custom"],
        help="experiment: 写回 ckpt 所在 version_* 目录；custom 需配合 --output_dir",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="results_dir=custom 时的输出根目录",
    )
    return p.parse_args()


def _parse_bool(v: str | None) -> bool | None:
    if v is None:
        return None
    v = str(v).strip().lower()
    if v in {"1", "true", "yes", "y"}:
        return True
    if v in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"无法解析布尔参数: {v}")


def _normalize_split(split: str) -> str:
    return "validate" if split in {"val", "validate"} else "test"


def _restore_disease_cols(model: MILLinearProbe) -> list[str]:
    hparams = getattr(model, "hparams", None)
    if hparams is None:
        return list(DISEASE_LIST)
    if isinstance(hparams, dict):
        cols = hparams.get("disease_cols")
    else:
        cols = getattr(hparams, "disease_cols", None)
    return list(cols) if cols else list(DISEASE_LIST)


def _drop_na_label_rows(base_dataset, available_cols: list[str], split_name: str) -> None:
    if len(base_dataset.data_list) != len(base_dataset._rows_image_paths):
        raise ValueError(
            f"{split_name} 数据不一致: data_list={len(base_dataset.data_list)}, "
            f"rows_image_paths={len(base_dataset._rows_image_paths)}"
        )
    keep_mask = ~base_dataset.data_list[available_cols].isna().any(axis=1)
    keep_indices = [i for i, keep in enumerate(keep_mask.tolist()) if keep]
    if not keep_indices:
        raise ValueError(f"{split_name} 在严格标签过滤后无可用样本。")
    base_dataset.data_list = base_dataset.data_list.iloc[keep_indices].reset_index(drop=True)
    base_dataset._rows_image_paths = [base_dataset._rows_image_paths[i] for i in keep_indices]
    if hasattr(base_dataset, "_flat_image_paths") and base_dataset._flat_image_paths is not None:
        base_dataset._flat_image_paths = []
        for paths in base_dataset._rows_image_paths:
            base_dataset._flat_image_paths.extend(paths)


def _build_loader(
    cfg: dict,
    dataset_cfg: dict,
    features_dir: Path,
    model_id: str,
    region: str,
    split: str,
    disease_cols: list[str],
    feature_dim: int,
    preload_features: bool,
    batch_size: int | None = None,
):
    dataset_name = cfg["dataset_name"]
    data_dir = dataset_cfg[dataset_name]["dataset_dir"]
    max_views = cfg.get("max_views", 6)
    batch_size = batch_size if batch_size is not None else cfg.get("batch_size", 8)
    num_workers = cfg.get("num_workers", 8)
    img_size = cfg.get("img_size", 224)
    in_channels = cfg.get("in_channels", 3)

    base_ds = build_private_base_dataset(
        data_dir, split, img_size, in_channels, [region], max_views
    )
    available_cols = [c for c in disease_cols if c in base_ds.data_list.columns]
    if not available_cols:
        raise ValueError(f"region={region} 在 CSV 中找不到 checkpoint 的标签列")
    if set(available_cols) != set(disease_cols):
        missing = [c for c in disease_cols if c not in available_cols]
        raise ValueError(
            f"region={region} 缺少 checkpoint 标签列: {missing[:10]}"
            + ("..." if len(missing) > 10 else "")
        )
    _drop_na_label_rows(base_ds, available_cols, split)

    loader_num_workers = 0 if preload_features else num_workers
    dataset = MILFeatureDataset(
        base_ds,
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
        shuffle=False,
        num_workers=loader_num_workers,
        pin_memory=True,
        collate_fn=mil_collate_fn,
    )
    return loader, len(dataset)


def _resolve_results_path(
    ckpt_path: Path, region: str, results_dir: str, output_dir: Path | None
) -> Path:
    if results_dir == "experiment":
        # .../version_k/checkpoints/foo.ckpt -> .../version_k/test_results.csv
        return ckpt_path.parent.parent / "test_results.csv"
    if output_dir is None:
        raise ValueError("results_dir=custom 时必须提供 --output_dir")
    safe_region = str(region).replace("/", "_")
    out = output_dir / safe_region
    out.mkdir(parents=True, exist_ok=True)
    return out / "test_results.csv"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dataset_cfg = load_config(args.dataset_config)
    cfg["dataset_name"] = cfg.get("dataset_name", "private_a").lower()
    if cfg["dataset_name"] != "private_a":
        raise ValueError(f"该脚本仅用于 private_a，当前 dataset_name={cfg['dataset_name']}")

    split = _normalize_split(args.split)
    batch_size = args.batch_size if args.batch_size is not None else cfg.get("batch_size", 8)
    preload_cli = _parse_bool(args.preload_features)
    preload_features = (
        preload_cli if preload_cli is not None else cfg.get("preload_features", True)
    )
    output_dir = Path(args.output_dir) if args.output_dir else None

    print(
        f"batch_size={batch_size}  preload_features={preload_features}  "
        f"(preload=false 时会从 NAS 逐文件读特征，首个 batch 可能很慢)"
    )

    with open(args.region_ckpt_map, "r") as f:
        region_ckpt_map = {str(k): str(v) for k, v in json.load(f).items()}

    target_regions = args.regions or DEFAULT_REGIONS
    summary = []
    skipped = []

    print(f"model_id={args.model_id}  split={split}  regions={len(target_regions)}")
    for region in target_regions:
        ckpt_str = region_ckpt_map.get(region)
        if not ckpt_str:
            skipped.append({"region": region, "reason": "region_ckpt_map 中无对应 ckpt"})
            print(f"[跳过] {region}: 无 checkpoint 映射")
            continue

        ckpt_path = Path(ckpt_str)
        if not ckpt_path.is_file():
            skipped.append({"region": region, "reason": f"ckpt 不存在: {ckpt_path}"})
            print(f"[跳过] {region}: ckpt 不存在")
            continue

        print(f"\n{'=' * 72}")
        print(f"Region: {region}")
        print(f"Checkpoint: {ckpt_path}")

        try:
            model = MILLinearProbe.load_from_checkpoint(str(ckpt_path))
            model._loaded_ckpt_path = str(ckpt_path)
            disease_cols = _restore_disease_cols(model)
            feature_dim = getattr(model, "feature_dim", None)
            if feature_dim is None:
                raise ValueError("checkpoint 缺少 feature_dim")

            loader, num_cases = _build_loader(
                cfg=cfg,
                dataset_cfg=dataset_cfg,
                features_dir=Path(args.features_dir),
                model_id=args.model_id,
                region=region,
                split=split,
                disease_cols=disease_cols,
                feature_dim=feature_dim,
                preload_features=preload_features,
                batch_size=batch_size,
            )

            results_path = _resolve_results_path(
                ckpt_path, region, args.results_dir, output_dir
            )
            model.results_save_path = results_path
            print(f"Cases: {num_cases}")
            print(f"Will write: {results_path}")

            trainer = pl.Trainer(
                accelerator="auto",
                devices="auto",
                precision="16-mixed",
                logger=False,
            )
            metrics = trainer.test(model, loader)
            metric_dict = metrics[0] if metrics else {}

            json_path = results_path.with_name(f"{split}_results.json")
            payload = {
                "timestamp": datetime.now().isoformat(),
                "dataset": "private_a",
                "region": region,
                "split": split,
                "checkpoint": str(ckpt_path),
                "features_dir": args.features_dir,
                "model_id": args.model_id,
                "num_cases": num_cases,
                "results_csv": str(results_path),
                "metrics": metric_dict,
            }
            with open(json_path, "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            summary.append(
                {
                    "region": region,
                    "num_cases": num_cases,
                    "csv": str(results_path),
                    "test_custom_auc": metric_dict.get("test/custom_auc"),
                    "test_custom_f1": metric_dict.get("test/custom_f1"),
                }
            )
            print(f"[完成] {region}: AUC={metric_dict.get('test/custom_auc', 'N/A')}")
        except Exception as exc:
            skipped.append({"region": region, "reason": str(exc)})
            print(f"[ERROR] {region}: {exc}")

    print(f"\n{'=' * 72}")
    print(f"完成: success={len(summary)}, skipped={len(skipped)}")
    if summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    if skipped:
        print("Skipped:")
        print(json.dumps(skipped, indent=2, ensure_ascii=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
