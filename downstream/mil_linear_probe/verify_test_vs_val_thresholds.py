"""Verify saved test_results.csv matches recomputation from best ckpt + val_thresholds.json.

For each experiment:
  1) Load ckpt path recorded in val_thresholds.json (best val_custom_auc ckpt)
  2) Run test-split inference -> probs
  3) Binarize with thresholds from the same val_thresholds.json
  4) Compare probs/preds and macro AUC/b-acc against saved test_results.csv
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from save_val_thresholds import run_val_inference
from test_mil_lp_private_a_by_region import _build_loader, _restore_disease_cols
from trainer_mil_lp import MILLinearProbe
from val_thresholds_utils import read_thresholds_json

spec = importlib.util.spec_from_file_location(
    "sa_utils", ROOT / "statistical_analysis/utils.py"
)
sa_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa_utils)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def mean_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    scores = []
    for c in range(y_true.shape[1]):
        yt, yp = y_true[:, c], y_prob[:, c]
        if not np.any(yt) or len(np.unique(yt)) < 2:
            continue
        scores.append(roc_auc_score(yt, yp))
    return float(np.mean(scores)) if scores else float("nan")


def mean_bacc(y_true: np.ndarray, y_bin: np.ndarray) -> float:
    scores = []
    for c in range(y_true.shape[1]):
        yt, yp = y_true[:, c], y_bin[:, c]
        if not np.any(yt) or len(np.unique(yt)) < 2:
            continue
        scores.append(balanced_accuracy_score(yt, yp.astype(int)))
    return float(np.mean(scores)) if scores else float("nan")


def load_saved_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    y_true = np.array([ast.literal_eval(x) for x in df["ground_truth"]], dtype=float)
    y_bin = np.array([ast.literal_eval(x) for x in df["prediction"]], dtype=int)
    if "prediction_probs" not in df.columns:
        raise ValueError(f"{csv_path} 缺少 prediction_probs")
    y_prob = np.array([ast.literal_eval(x) for x in df["prediction_probs"]], dtype=float)
    return y_true, y_prob, y_bin


@torch.no_grad()
def run_test_inference(model: MILLinearProbe, loader) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    probs_chunks, labels_chunks = [], []
    for batch in loader:
        if len(batch) == 4:
            bags, mask, labels, _ = batch
        else:
            bags, mask, labels = batch
        bags = bags.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True) if mask is not None else None
        labels = labels.to(device, non_blocking=True).float()
        logits, _ = model(bags, mask)
        probs = torch.sigmoid(logits)
        probs_chunks.append(probs.detach().cpu().numpy())
        labels_chunks.append(labels.detach().cpu().numpy())
    return np.concatenate(probs_chunks, axis=0), np.concatenate(labels_chunks, axis=0)


def verify_one(
    region: str,
    model_id: str,
    mil_type: str,
    csv_path: Path,
    cfg: dict,
    dataset_cfg: dict,
    features_dir: Path,
    batch_size: int,
    preload_features: bool,
) -> dict:
    version_dir = csv_path.parent
    sidecar = version_dir / "checkpoints" / "val_thresholds.json"
    if not sidecar.is_file():
        raise FileNotFoundError(f"缺少 {sidecar}")

    payload = read_thresholds_json(sidecar)
    ckpt_str = payload.get("ckpt")
    if not ckpt_str:
        raise ValueError(f"{sidecar} 无 ckpt 字段")
    ckpt_path = Path(ckpt_str)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"ckpt 不存在: {ckpt_path}")

    thresholds = np.asarray(payload["thresholds"], dtype=np.float64)
    if thresholds.ndim != 1:
        raise ValueError(f"非法 thresholds shape: {thresholds.shape}")

    model = MILLinearProbe.load_from_checkpoint(str(ckpt_path))
    disease_cols = _restore_disease_cols(model)
    feature_dim = getattr(model, "feature_dim", None)
    if feature_dim is None:
        raise ValueError("checkpoint 缺少 feature_dim")

    loader, n_cases = _build_loader(
        cfg=cfg,
        dataset_cfg=dataset_cfg,
        features_dir=features_dir,
        model_id=model_id,
        region=region,
        split="test",
        disease_cols=disease_cols,
        feature_dim=feature_dim,
        preload_features=preload_features,
        batch_size=batch_size,
    )

    y_prob_new, y_true_new = run_test_inference(model, loader)
    if y_prob_new.shape[0] != n_cases:
        raise RuntimeError(f"样本数不一致 loader={n_cases} preds={y_prob_new.shape[0]}")

    if thresholds.shape[0] != y_prob_new.shape[1]:
        raise ValueError(
            f"阈值维度 {thresholds.shape[0]} != n_classes {y_prob_new.shape[1]}"
        )
    y_bin_new = (y_prob_new > thresholds[None, :]).astype(int)

    y_true_old, y_prob_old, y_bin_old = load_saved_csv(csv_path)
    if y_true_old.shape != y_true_new.shape:
        raise ValueError(
            f"形状不一致 saved={y_true_old.shape} recomputed={y_true_new.shape}"
        )

    prob_max_abs = float(np.max(np.abs(y_prob_old - y_prob_new)))
    pred_match = float(np.mean(y_bin_old == y_bin_new))
    label_match = float(np.mean(y_true_old == y_true_new))

    auc_old = mean_auc(y_true_old, y_prob_old)
    auc_new = mean_auc(y_true_new, y_prob_new)
    bacc_old = mean_bacc(y_true_old, y_bin_old)
    bacc_new = mean_bacc(y_true_new, y_bin_new)

    return {
        "region": region,
        "model": model_id,
        "mil_type": mil_type,
        "n_cases": n_cases,
        "ckpt": str(ckpt_path),
        "threshold_source": sidecar,
        "prob_max_abs_diff": prob_max_abs,
        "pred_match_rate": pred_match,
        "label_match_rate": label_match,
        "auc_saved": auc_old,
        "auc_recomputed": auc_new,
        "auc_abs_diff": abs(auc_old - auc_new),
        "bacc_saved": bacc_old,
        "bacc_recomputed": bacc_new,
        "bacc_abs_diff": abs(bacc_old - bacc_new),
        "pass_prob": prob_max_abs < 1e-4,
        "pass_pred": pred_match >= 1.0 - 1e-9,
        "pass_auc": abs(auc_old - auc_new) < 1e-5,
        "pass_bacc": abs(bacc_old - bacc_new) < 1e-5,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=str, required=True)
    p.add_argument("--model_id", type=str, default="topk_v19_last")
    p.add_argument(
        "--features_dir",
        type=str,
        default="${SMARTX_OUTPUT_ROOT}/features/private_a",
    )
    p.add_argument("--config", type=str, default="configs/train_private_a_our_ark_large.yaml")
    p.add_argument("--dataset_config", type=str, default="configs/dataset_dir.yaml")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--preload_features", action="store_true", default=True)
    p.add_argument("--no_preload_features", action="store_false", dest="preload_features")
    p.add_argument("--output_csv", type=str, default="analysis_results/verify_test_thresholds.csv")
    args = p.parse_args()

    cfg = load_config(args.config)
    dataset_cfg = load_config(args.dataset_config)
    cfg["dataset_name"] = "private_a"

    exps = []
    for ds, model, _cfg_name, mil, csv_path in sa_utils.find_experiment_paths(args.base_dir):
        if model != args.model_id:
            continue
        region = ds.replace("private_a_", "") if ds.startswith("private_a_") else ds
        exps.append((region, model, mil, Path(csv_path)))

    print(f"Verify {args.model_id}: {len(exps)} experiments")
    rows = []
    for region, model_id, mil_type, csv_path in tqdm(exps, desc="verify"):
        try:
            rows.append(
                verify_one(
                    region,
                    model_id,
                    mil_type,
                    csv_path,
                    cfg,
                    dataset_cfg,
                    Path(args.features_dir),
                    args.batch_size,
                    args.preload_features,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "region": region,
                    "model": model_id,
                    "mil_type": mil_type,
                    "error": str(exc),
                    "pass_prob": False,
                    "pass_pred": False,
                    "pass_auc": False,
                    "pass_bacc": False,
                }
            )

    df = pd.DataFrame(rows)
    out = ROOT / args.output_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    ok = df.get("error", pd.Series(dtype=object)).isna() if "error" in df.columns else pd.Series([True] * len(df))
    sub = df[ok.fillna(False)] if "error" in df.columns else df
    all_pass = (
        sub["pass_prob"].all()
        and sub["pass_pred"].all()
        and sub["pass_auc"].all()
        and sub["pass_bacc"].all()
    )
    print(f"\nSaved: {out}")
    print(f"Success: {len(sub)}/{len(df)}")
    if len(sub):
        print(f"prob max diff: {sub['prob_max_abs_diff'].max():.6e}")
        print(f"pred match min: {sub['pred_match_rate'].min():.6f}")
        print(f"auc abs diff max: {sub['auc_abs_diff'].max():.6e}")
        print(f"bacc abs diff max: {sub['bacc_abs_diff'].max():.6e}")
    print(f"ALL PASS: {all_pass}")
    if "error" in df.columns and df["error"].notna().any():
        print("\nFailures:")
        print(df[df["error"].notna()][["region", "mil_type", "error"]].to_string(index=False))
    if len(sub) and not all_pass:
        bad = sub[
            ~(sub["pass_prob"] & sub["pass_pred"] & sub["pass_auc"] & sub["pass_bacc"])
        ]
        print("\nMismatch rows:")
        print(
            bad[
                [
                    "region",
                    "mil_type",
                    "prob_max_abs_diff",
                    "pred_match_rate",
                    "auc_abs_diff",
                    "bacc_abs_diff",
                ]
            ].to_string(index=False)
        )
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
