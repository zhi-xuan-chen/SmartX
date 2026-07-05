"""
ZSFY 按 Region 批量测试脚本。

核心能力：
1) 使用 ZSFY 全量数据（无官方 split，默认全部作为测试集）。
2) 按 CSV 中 Region 分组。
3) 每个 Region 使用对应的已训练 MIL checkpoint 测试。
4) 每个 Region 都保存：
   - test_results.csv（逐样本结果，和训练流程一致）
   - test_results.json（元信息/label 对齐信息）
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
import sys

import torch
import numpy as np

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent.parent))
from utils import load_config
from trainer_mil_lp import MILLinearProbe
from data_utils import (
    DISEASE_LIST,
    build_zsfy_base_dataset,
    MILFeatureDataset,
    mil_collate_fn,
)
from val_thresholds_utils import load_thresholds_for_ckpt


def parse_args():
    parser = argparse.ArgumentParser(description="Region-wise ZSFY MIL test")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_zsfy_mil.yaml",
        help="模型/数据参数配置（需包含 dataset_name: zsfy）",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="configs/dataset_dir.yaml",
        help="数据集目录配置",
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        required=True,
        help="ZSFY 特征根目录（例如 .../features/zsfy）",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="特征子目录名（如 ark_base / ark_large）",
    )
    parser.add_argument(
        "--region_ckpt_map",
        type=str,
        default=None,
        help="region->ckpt 映射 JSON 文件路径",
    )
    parser.add_argument(
        "--ckpt_template",
        type=str,
        default=None,
        help="ckpt 模板字符串，需包含 {region} 占位符",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录：每个 region 一个子目录",
    )
    parser.add_argument(
        "--preload_features",
        type=str,
        default=None,
        help="是否预加载特征（true/false），默认跟随 config",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default=None,
        help="可选：覆盖 ZSFY 标注 CSV 文件名（位于 dataset_dir 下）",
    )
    parser.add_argument(
        "--result_model_id",
        type=str,
        default=None,
        help="可选：结果目录/汇总中使用的模型名（默认等于 --model_id）",
    )
    return parser.parse_args()


def _parse_bool(v: str | None):
    if v is None:
        return None
    v = str(v).strip().lower()
    if v in {"1", "true", "yes", "y"}:
        return True
    if v in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"无法解析布尔参数: {v}")


def _safe_region_name(region: str) -> str:
    return str(region).replace("/", "_")


def _load_region_ckpt_map(args) -> dict:
    if args.region_ckpt_map:
        with open(args.region_ckpt_map, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("--region_ckpt_map JSON 必须是字典：{region: ckpt_path}")
        return {str(k): str(v) for k, v in data.items()}

    if args.ckpt_template:
        # 延后在每个 region 上 format
        return {}

    raise ValueError("请至少提供 --region_ckpt_map 或 --ckpt_template。")


def _resolve_ckpt(region: str, region_ckpt_map: dict, ckpt_template: str | None) -> str | None:
    if region in region_ckpt_map:
        return region_ckpt_map[region]
    if ckpt_template:
        return ckpt_template.format(region=region)
    return None


def _restore_disease_cols(model) -> list:
    disease_cols = None
    try:
        if hasattr(model, "hparams") and isinstance(model.hparams, dict):
            disease_cols = model.hparams.get("disease_cols", None)
        elif hasattr(model, "hparams"):
            disease_cols = getattr(model.hparams, "disease_cols", None)
    except Exception:
        disease_cols = None
    return disease_cols or DISEASE_LIST


def _build_loader_for_region(
    cfg: dict,
    dataset_cfg: dict,
    dataset_name: str,
    features_dir: Path,
    model_id: str,
    region: str,
    disease_cols: list,
    feature_dim: int,
    preload_features: bool,
    csv_name: str | None = None,
):
    data_dir = dataset_cfg[dataset_name]["dataset_dir"]
    max_views = cfg.get("max_views", 6)
    batch_size = cfg.get("batch_size", 8)
    num_workers = cfg.get("num_workers", 8)
    img_size = cfg.get("img_size", 224)
    in_channels = cfg.get("in_channels", 3)

    base_ds = build_zsfy_base_dataset(
        data_dir=data_dir,
        split="test",
        img_size=img_size,
        in_channels=in_channels,
        max_views=max_views,
        region_filter=region,
        csv_name=csv_name if csv_name else "zhognshanfuyi_with_labels_qwen3-30b_single_region_updated.csv",
    )
    if len(base_ds.data_list) == 0:
        return None, 0, [], [], []

    # 严格一致性策略：
    # 评估 CSV 的标签列必须与 checkpoint 里保存的 disease_cols 完全匹配；
    # 只要发现缺失（不允许“交集推理”），直接报错退出。
    available_set = set(base_ds.data_list.columns.tolist())
    missing_cols = [c for c in disease_cols if c not in available_set]
    if missing_cols:
        raise ValueError(
            f"[标签列不一致] dataset=zsfy region={region} "
            f"ckpt_disease_cols_len={len(disease_cols)} "
            f"missing_cols_len={len(missing_cols)} "
            f"missing_cols_sample={missing_cols[:10]}"
        )

    used_disease_cols = list(disease_cols)  # 保留 checkpoint disease_cols 的顺序
    used_indices_in_ckpt = list(range(len(disease_cols)))

    loader_num_workers = 0 if preload_features else num_workers
    dataset = MILFeatureDataset(
        base_ds,
        features_root=features_dir,
        model_id=model_id,
        max_views=max_views,
        disease_cols=used_disease_cols,
        feature_dim=feature_dim,
        preload_features=preload_features,
        split="test",
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=loader_num_workers,
        pin_memory=True,
        collate_fn=mil_collate_fn,
    )
    return loader, len(base_ds.data_list), used_disease_cols, missing_cols, used_indices_in_ckpt


@torch.no_grad()
def _inference_on_subset(model, dataloader, class_indices: list[int]):
    """
    推理并按交集标签切片：
    - labels 来自 dataloader（已是交集标签维度）
    - logits 从原模型 forward 得到后，按 class_indices 切片到交集维度
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    probs_all = []
    labels_all = []

    for batch in dataloader:
        if len(batch) == 4:
            bags, mask, labels, _regions = batch
        else:
            bags, mask, labels = batch
        bags = bags.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True) if mask is not None else None
        labels = labels.to(device, non_blocking=True).float()

        logits, _ = model(bags, mask)
        logits = logits[:, class_indices]
        probs = torch.sigmoid(logits)

        probs_all.append(probs.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())

    probs_np = np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0, len(class_indices)))
    labels_np = np.concatenate(labels_all, axis=0) if labels_all else np.zeros((0, len(class_indices)))
    return probs_np, labels_np


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dataset_cfg = load_config(args.dataset_config)
    cfg["dataset_name"] = cfg.get("dataset_name", "zsfy").lower()
    dataset_name = cfg["dataset_name"]
    if dataset_name != "zsfy":
        raise ValueError(f"该脚本仅用于 ZSFY，当前 dataset_name={dataset_name}")

    region_ckpt_map = _load_region_ckpt_map(args)
    result_model_id = args.result_model_id or args.model_id
    preload_features_cli = _parse_bool(args.preload_features)
    # 默认不预加载，按需从磁盘读取；如需预加载可在 config 或命令行中显式开启
    preload_features = (
        preload_features_cli
        if preload_features_cli is not None
        else cfg.get("preload_features", False)
    )

    data_dir = dataset_cfg[dataset_name]["dataset_dir"]
    scan_ds = build_zsfy_base_dataset(
        data_dir=data_dir,
        split="test",
        img_size=cfg.get("img_size", 224),
        in_channels=cfg.get("in_channels", 3),
        max_views=cfg.get("max_views", 6),
        region_filter=None,
        csv_name=args.csv_name if args.csv_name else "zhognshanfuyi_with_labels_qwen3-30b_single_region_updated.csv",
    )
    regions = sorted([str(x) for x in scan_ds.data_list["Region"].dropna().unique().tolist()])
    if len(regions) == 0:
        raise ValueError("ZSFY 数据中没有可用 Region。")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_summary = []
    skipped = []

    print(f"\n发现 Region 数量: {len(regions)}")
    for region in regions:
        ckpt_path = _resolve_ckpt(region, region_ckpt_map, args.ckpt_template)
        if ckpt_path is None or not Path(ckpt_path).exists():
            skipped.append({"region": region, "reason": f"ckpt 不存在: {ckpt_path}"})
            print(f"[跳过] {region}: ckpt 不存在")
            continue

        print(f"\n{'=' * 60}")
        print(f"Region: {region}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'=' * 60}")

        model = MILLinearProbe.load_from_checkpoint(ckpt_path)
        disease_cols = _restore_disease_cols(model)
        feature_dim = getattr(model, "feature_dim", None)
        if feature_dim is None:
            skipped.append({"region": region, "reason": "checkpoint 缺少 feature_dim"})
            print(f"[跳过] {region}: checkpoint 缺少 feature_dim")
            continue

        loader, num_cases, used_disease_cols, missing_cols, used_indices_in_ckpt = _build_loader_for_region(
            cfg=cfg,
            dataset_cfg=dataset_cfg,
            dataset_name=dataset_name,
            features_dir=Path(args.features_dir),
            model_id=args.model_id,
            region=region,
            disease_cols=disease_cols,
            feature_dim=feature_dim,
            preload_features=preload_features,
            csv_name=args.csv_name,
        )
        if loader is None:
            reason = "该 region 无样本" if num_cases == 0 else "无可评估标签（交集为空）"
            if missing_cols:
                reason += f"；缺失标签列示例: {missing_cols[:10]}" + ("..." if len(missing_cols) > 10 else "")
            skipped.append({"region": region, "reason": reason})
            print(f"[跳过] {region}: {reason}")
            continue

        region_dir = output_root / _safe_region_name(region) / result_model_id
        region_dir.mkdir(parents=True, exist_ok=True)

        # 保存逐样本 CSV（供后续 bootstrap / delong 统一分析）
        model.results_save_path = region_dir / "test_results.csv"

        label_alignment = {
            "ckpt_disease_cols": disease_cols,
            "used_disease_cols": used_disease_cols,
            "missing_disease_cols_in_zsfy": missing_cols,
        }
        if missing_cols:
            print(
                f"[提示] {region}: 标签不一致，按交集推理。"
                f" 使用 {len(used_disease_cols)}/{len(disease_cols)} 个标签；缺失示例: {missing_cols[:5]}"
            )

        probs_np, labels_np = _inference_on_subset(
            model=model, dataloader=loader, class_indices=used_indices_in_ckpt
        )

        try:
            import pandas as pd

            ckpt_thresholds = load_thresholds_for_ckpt(
                ckpt_path, expected_disease_cols=list(disease_cols)
            )
            if len(ckpt_thresholds) != len(disease_cols):
                used_thresholds = np.full(len(used_indices_in_ckpt), 0.5)
            else:
                used_thresholds = ckpt_thresholds[np.asarray(used_indices_in_ckpt, dtype=int)]
            binary_preds = (probs_np > used_thresholds[None, :]).astype(int)
            binary_labels = labels_np.astype(int)
            correct = (binary_preds == binary_labels).all(axis=1).astype(int)
            results_df = pd.DataFrame(
                {
                    "ground_truth": binary_labels.tolist(),
                    "prediction": binary_preds.tolist(),
                    "prediction_probs": probs_np.tolist(),
                    "correct": correct,
                    "region": [region] * len(probs_np),
                }
            )
            results_df.to_csv(model.results_save_path, index=False)
            print(f"Saved test results to: {model.results_save_path}")
        except Exception as e:
            print(f"Error saving test results: {e}")
            skipped.append({"region": region, "reason": f"保存 test_results.csv 失败: {e}"})
            continue

        result_payload = {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "region": region,
            "num_cases": num_cases,
            "checkpoint": str(ckpt_path),
            "features_dir": str(args.features_dir),
            "csv_name": args.csv_name,
            "model_id": result_model_id,
            "label_alignment": label_alignment,
        }
        with open(region_dir / "test_results.json", "w") as f:
            json.dump(result_payload, f, indent=2, ensure_ascii=False)

        all_summary.append(result_payload)
        print(f"[完成] {region}: cases={num_cases}, CSV={model.results_save_path}")

    summary_payload = {
        "timestamp": datetime.now().isoformat(),
        "dataset_name": dataset_name,
        "features_dir": str(args.features_dir),
        "csv_name": args.csv_name,
        "model_id": result_model_id,
        "num_regions": len(regions),
        "completed_regions": len(all_summary),
        "skipped_regions": skipped,
        "regions": all_summary,
    }
    summary_dir = output_root / "_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{result_model_id}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    print(f"\n全部完成。汇总文件: {summary_path}")
    if skipped:
        print(f"跳过 {len(skipped)} 个 Region，请检查 summary.json")


if __name__ == "__main__":
    main()

