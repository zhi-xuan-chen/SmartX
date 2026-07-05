import argparse
import ast
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from tqdm import tqdm

from utils import parse_model_list, model_allowed, add_model_filter_args, included_classes

METRICS = ["auc", "f1", "ap", "mcc", "bacc"]

# See bootstrap.py: per-disease CIs with fewer positives are flagged unreliable.
MIN_POS = 10


def load_test_results(csv_path: str):
    """Return (y_true, y_prob, y_pred_binary).

    F1/MCC use the *fixed* ``prediction`` column (val-threshold applied at
    test time); we no longer re-threshold inside bootstrap.
    """
    df = pd.read_csv(csv_path)
    y_true = np.array([ast.literal_eval(gt) for gt in df["ground_truth"]])
    if "prediction_probs" not in df.columns:
        raise ValueError(f"{csv_path} 缺少 prediction_probs 列，无法计算 AUC/AP。")
    y_prob = np.array([ast.literal_eval(p) for p in df["prediction_probs"]])
    if "prediction" not in df.columns:
        raise ValueError(
            f"{csv_path} 缺少 prediction 列；请重跑 test 以写入 val-threshold 二值预测。"
        )
    y_bin = np.array([ast.literal_eval(p) for p in df["prediction"]])
    return y_true, y_prob, y_bin


def compute_metrics_single_class(y_true, y_pred_probs, y_pred_binary):
    if not np.any(y_true):
        return {m: np.nan for m in METRICS} | {"has_positives": False}
    has_both = len(np.unique(y_true)) >= 2
    res = {
        "has_positives": True,
        "auc": roc_auc_score(y_true, y_pred_probs) if has_both else np.nan,
        "ap": average_precision_score(y_true, y_pred_probs),
    }
    y_bin = y_pred_binary.astype(int)
    res["f1"] = f1_score(y_true, y_bin, zero_division=0)
    res["mcc"] = matthews_corrcoef(y_true, y_bin) if has_both else np.nan
    res["bacc"] = balanced_accuracy_score(y_true, y_bin) if has_both else np.nan
    return res


def compute_all_metrics(y_true, y_pred_probs, y_pred_binary, fixed_valid=None):
    """Per-class metrics + macro mean over a class set fixed from the full
    sample (issue #2). See the in-domain ``bootstrap.py`` for the rationale."""
    n_classes = y_true.shape[1]
    results = {"per_class": [], "mean": {}, "valid": {}}
    per_class_vals = {m: np.full(n_classes, np.nan) for m in METRICS}
    for c in range(n_classes):
        m = compute_metrics_single_class(
            y_true[:, c], y_pred_probs[:, c], y_pred_binary[:, c]
        )
        m["class_idx"] = c
        results["per_class"].append(m)
        if m.get("has_positives"):
            for k in METRICS:
                if m[k] is not None and not np.isnan(m[k]):
                    per_class_vals[k][c] = m[k]
    for k in METRICS:
        vals = per_class_vals[k]
        results["valid"][k] = ~np.isnan(vals)
        mask = results["valid"][k] if fixed_valid is None else fixed_valid[k]
        sel = vals[mask]
        sel = sel[~np.isnan(sel)]
        results["mean"][k] = np.mean(sel) if sel.size else np.nan
    return results


def _stratified_per_class_boot(y_true_c, y_prob_c, y_bin_c, n_boot, rng):
    """Per-class stratified bootstrap draws; see bootstrap.py for rationale."""
    pos_idx = np.where(y_true_c == 1)[0]
    neg_idx = np.where(y_true_c == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    draws = np.full((n_boot, len(METRICS)), np.nan)
    if n_pos == 0:
        return draws
    for b in range(n_boot):
        if n_neg > 0:
            samp = np.concatenate([
                rng.choice(pos_idx, size=n_pos, replace=True),
                rng.choice(neg_idx, size=n_neg, replace=True),
            ])
        else:
            samp = rng.choice(pos_idx, size=n_pos, replace=True)
        mvals = compute_metrics_single_class(y_true_c[samp], y_prob_c[samp], y_bin_c[samp])
        for m_idx, m in enumerate(METRICS):
            draws[b, m_idx] = mvals.get(m, np.nan)
    return draws


def bootstrap_metrics(y_true, y_prob, y_bin, n_boot=1000, conf=0.95, seed=42, keep=None):
    rng = np.random.RandomState(seed)
    n_samples, n_classes = y_true.shape
    if keep is None:
        keep = np.ones(n_classes, dtype=bool)
    point = compute_all_metrics(y_true, y_prob, y_bin)
    # freeze macro-mean class set (#2) and drop excluded diseases (per-region).
    fixed_valid = {k: point["valid"][k] & keep for k in METRICS}
    # recompute so the reported point mean also respects the kept set.
    point = compute_all_metrics(y_true, y_prob, y_bin, fixed_valid=fixed_valid)

    # Pass 1 — macro mean: shared row-level case bootstrap, fixed class set.
    boot_mean = np.full((n_boot, len(METRICS)), np.nan)
    for b in range(n_boot):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        res = compute_all_metrics(y_true[idx], y_prob[idx], y_bin[idx], fixed_valid=fixed_valid)
        for m_idx, m in enumerate(METRICS):
            boot_mean[b, m_idx] = res["mean"].get(m, np.nan)

    # Pass 2 — per-disease: stratified bootstrap per class (#3 / #C).
    boot_per_class = np.full((n_boot, n_classes, len(METRICS)), np.nan)
    for c in range(n_classes):
        boot_per_class[:, c, :] = _stratified_per_class_boot(
            y_true[:, c], y_prob[:, c], y_bin[:, c], n_boot, rng
        )

    alpha = 1 - conf
    lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100
    out = {"per_class": [], "mean": {}}

    for c in range(n_classes):
        if not keep[c]:
            continue  # disease excluded for this region
        n_pos = int((y_true[:, c] == 1).sum())
        row = {
            "class_idx": c,
            "has_positives": point["per_class"][c]["has_positives"],
            "n_pos": n_pos,
            "ci_reliable": n_pos >= MIN_POS,
        }
        for m_idx, m in enumerate(METRICS):
            row[m] = point["per_class"][c].get(m, np.nan)
            valid = boot_per_class[:, c, m_idx][~np.isnan(boot_per_class[:, c, m_idx])]
            if len(valid) > 0:
                row[f"{m}_ci_low"], row[f"{m}_ci_high"] = np.percentile(valid, lo), np.percentile(valid, hi)
                row[f"{m}_std"] = np.std(valid)
            else:
                row[f"{m}_ci_low"] = np.nan
                row[f"{m}_ci_high"] = np.nan
                row[f"{m}_std"] = np.nan
        out["per_class"].append(row)

    for m_idx, m in enumerate(METRICS):
        out["mean"][m] = point["mean"].get(m, np.nan)
        valid = boot_mean[:, m_idx][~np.isnan(boot_mean[:, m_idx])]
        if len(valid) > 0:
            out["mean"][f"{m}_ci_low"], out["mean"][f"{m}_ci_high"] = np.percentile(valid, lo), np.percentile(valid, hi)
            out["mean"][f"{m}_std"] = np.std(valid)
        else:
            out["mean"][f"{m}_ci_low"] = np.nan
            out["mean"][f"{m}_ci_high"] = np.nan
            out["mean"][f"{m}_std"] = np.nan
    return out


def find_result_csvs(results_dir: str):
    """
    解析目录结构: results/<mil_type>/<dataset>/<region>/<model>/test_results.csv
    （results_dir 应指向 results/<mil_type> 目录）
    返回: (dataset_region, dataset, region, model, csv_path)
    """
    out = []
    for csv_path in Path(results_dir).glob("*/*/*/test_results.csv"):
        model = csv_path.parent.name
        region = csv_path.parent.parent.name
        dataset = csv_path.parent.parent.parent.name
        dataset_region = f"{dataset}_{region}"
        out.append((dataset_region, dataset, region, model, str(csv_path)))
    return out


def process_one(args):
    dataset_region, dataset, region, model, csv_path, n_boot = args
    result = {"success": False, "pd_rows": [], "mean_rows": []}
    try:
        y_true, y_prob, y_bin = load_test_results(csv_path)
        keep = included_classes(dataset_region, y_true.shape[1], region=region)
        boot = bootstrap_metrics(y_true, y_prob, y_bin, n_boot=n_boot, keep=keep)

        for c_res in boot["per_class"]:
            result["pd_rows"].append(
                {
                    "dataset": dataset_region,
                    "source_dataset": dataset,
                    "region": region,
                    "model": model,
                    **c_res,
                }
            )
        result["mean_rows"].append(
            {
                "dataset": dataset_region,
                "source_dataset": dataset,
                "region": region,
                "model": model,
                **{f"mean_{k}": v for k, v in boot["mean"].items()},
            }
        )
        result["success"] = True
    except Exception as exc:
        result["error"] = str(exc)
        result["dataset"] = dataset_region
        result["model"] = model
    return result


def analyze(results_dir, output_dir, n_bootstrap=1000, n_workers=None, models=None, exclude_models=None):
    os.makedirs(output_dir, exist_ok=True)
    allowed = parse_model_list(models)
    excluded = parse_model_list(exclude_models)
    jobs = [
        (dataset_region, dataset, region, model, csv_path, n_bootstrap)
        for dataset_region, dataset, region, model, csv_path in find_result_csvs(results_dir)
        if model_allowed(model, allowed, excluded)
    ]
    if not jobs:
        raise RuntimeError(f"未在 {results_dir} 下找到 test_results.csv")

    n_workers = n_workers or mp.cpu_count()
    per_disease_rows, mean_rows, errors = [], [], []
    if n_workers == 1:
        for job in tqdm(jobs):
            r = process_one(job)
            if r["success"]:
                per_disease_rows.extend(r["pd_rows"])
                mean_rows.extend(r["mean_rows"])
            else:
                errors.append(r)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_map = {executor.submit(process_one, job): job for job in jobs}
            for future in tqdm(as_completed(future_map), total=len(future_map)):
                r = future.result()
                if r["success"]:
                    per_disease_rows.extend(r["pd_rows"])
                    mean_rows.extend(r["mean_rows"])
                else:
                    errors.append(r)

    pd.DataFrame(per_disease_rows).to_csv(os.path.join(output_dir, "results_per_disease.csv"), index=False)
    pd.DataFrame(mean_rows).to_csv(os.path.join(output_dir, "results_mean.csv"), index=False)
    if errors:
        pd.DataFrame(errors).to_csv(os.path.join(output_dir, "analysis_errors.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Bootstrap 分析 results 目录中的测试结果")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="${SMARTX_REPO_ROOT}/downstream/mil_linear_probe/results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="${SMARTX_REPO_ROOT}/downstream/mil_linear_probe/analysis_results/results_region",
    )
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--n_workers", type=int, default=None)
    add_model_filter_args(parser)
    args = parser.parse_args()
    analyze(
        args.results_dir,
        args.output_dir,
        args.n_bootstrap,
        args.n_workers,
        args.models,
        args.exclude_models,
    )


if __name__ == "__main__":
    main()
