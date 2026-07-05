import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)
from tqdm import tqdm

from utils import load_results, find_experiment_paths, parse_model_list, model_allowed, add_model_filter_args, included_classes

METRICS = ["auc", "f1", "ap", "mcc", "bacc"]

# Per-disease CIs below this many positives are flagged unreliable (ci_reliable
# = False); the value is still reported. Stratified resampling keeps n_pos/n_neg
# fixed so the metric is always defined, but a handful of positives can't yield a
# trustworthy interval regardless of method.
MIN_POS = 10

# 若只想分析部分数据集，在这里列出它们的名字（base_dir 下的一级目录名）
# 例如：ALLOWED_DATASETS = ["private_a"]
ALLOWED_DATASETS = [
    "private_a_Ankle",
    "private_a_Cervical_spine",
    "private_a_Chest",
    "private_a_Foot",
    "private_a_Knee",
    "private_a_Lumbar_spine",
    "private_a_Metacarpal",
    "private_a_Pelvis",
    "private_a_Wrist",
    "private_a_Abdominal",
]


def compute_metrics_single_class(y_true, y_pred_probs, y_pred_binary):
    """F1/MCC use the *fixed* binary preds from CSV (val-threshold applied);
    AUC/AP remain threshold-free on probs."""
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
    """Per-class metrics + macro mean.

    The macro mean averages over a *fixed* set of classes so the bootstrap
    statistic is one well-defined estimand (issue #2): pass ``fixed_valid``
    (a dict ``{metric: bool array over classes}``, derived from the full-sample
    point estimate) and every resample averages over exactly those classes.
    When ``fixed_valid`` is None (the point-estimate call) the valid set is the
    classes where the metric is defined on this data, and it is returned under
    ``results["valid"]`` for the resamples to reuse.
    """
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
        sel = sel[~np.isnan(sel)]  # drop fixed-set classes that degenerate in a resample
        results["mean"][k] = np.mean(sel) if sel.size else np.nan
    return results


def _stratified_per_class_boot(y_true_c, y_p_c, y_b_c, n_boot, rng):
    """Per-class metric draws via *stratified* bootstrap.

    Positives and negatives are resampled separately with their counts held
    fixed (n_pos, n_neg), so AUC/MCC/b-acc are always defined — no resamples are
    dropped, unlike a plain row bootstrap that can leave a rare class single-
    labelled. Returns array (n_boot, n_metrics). pROC's default; Carpenter &
    Bithell (2000)."""
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
        mvals = compute_metrics_single_class(y_true_c[samp], y_p_c[samp], y_b_c[samp])
        for m_idx, m in enumerate(METRICS):
            draws[b, m_idx] = mvals.get(m, np.nan)
    return draws


def bootstrap_metrics(y_true, y_p, y_b, n_boot=1000, conf=0.95, seed=42, keep=None):
    rng = np.random.RandomState(seed)
    n_samples, n_classes = y_true.shape
    if keep is None:
        keep = np.ones(n_classes, dtype=bool)
    point = compute_all_metrics(y_true, y_p, y_b)
    # freeze the macro-mean class set from full sample (#2) and drop excluded
    # diseases (per-region exclusion) so mean/CI only cover kept classes.
    fixed_valid = {k: point["valid"][k] & keep for k in METRICS}
    # recompute so the reported point mean also respects the kept set (the CI
    # already does, via fixed_valid below).
    point = compute_all_metrics(y_true, y_p, y_b, fixed_valid=fixed_valid)

    # Pass 1 — macro mean: shared row-level case bootstrap (preserves cross-class
    # correlation; required for a coherent macro statistic), fixed class set.
    boot_mean = np.full((n_boot, len(METRICS)), np.nan)
    for b in range(n_boot):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        res = compute_all_metrics(y_true[idx], y_p[idx], y_b[idx], fixed_valid=fixed_valid)
        for m_idx, m in enumerate(METRICS):
            boot_mean[b, m_idx] = res["mean"].get(m, np.nan)

    # Pass 2 — per-disease: stratified bootstrap per class (#3 / #C).
    boot_per_class = np.full((n_boot, n_classes, len(METRICS)), np.nan)
    for c in range(n_classes):
        boot_per_class[:, c, :] = _stratified_per_class_boot(
            y_true[:, c], y_p[:, c], y_b[:, c], n_boot, rng
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
                row[f"{m}_ci_low"] = np.percentile(valid, lo)
                row[f"{m}_ci_high"] = np.percentile(valid, hi)
                row[f"{m}_std"] = np.std(valid)
            else:
                row[f"{m}_ci_low"] = row[f"{m}_ci_high"] = row[f"{m}_std"] = np.nan
        out["per_class"].append(row)
    for m_idx, m in enumerate(METRICS):
        out["mean"][m] = point["mean"].get(m, np.nan)
        valid = boot_mean[:, m_idx][~np.isnan(boot_mean[:, m_idx])]
        if len(valid) > 0:
            out["mean"][f"{m}_ci_low"] = np.percentile(valid, lo)
            out["mean"][f"{m}_ci_high"] = np.percentile(valid, hi)
            out["mean"][f"{m}_std"] = np.std(valid)
        else:
            out["mean"][f"{m}_ci_low"] = out["mean"][f"{m}_ci_high"] = out["mean"][f"{m}_std"] = np.nan
    return out


def process_single_experiment(args):
    ds, model, cfg, mil, path, n_boot, split_region = args
    res = {"dataset": ds, "model": model, "config": cfg, "mil_type": mil, "success": False, "pd": [], "mean": []}
    try:
        y_t, y_p, y_b, regions = load_results(path)
        if y_p is None:
            raise ValueError(f"{path} 缺少 prediction_probs 列，无法计算 AUC/AP。")
        if y_b is None:
            raise ValueError(f"{path} 缺少 prediction 列；请先重跑 test 以写入 val-threshold 二值预测。")

        keep = included_classes(ds, y_t.shape[1])

        def run_ds(target_y_t, target_y_p, target_y_b, name):
            boot = bootstrap_metrics(target_y_t, target_y_p, target_y_b, n_boot, keep=keep)
            for c_res in boot["per_class"]:
                res["pd"].append({"dataset": name, "model": model, "config": cfg, "mil_type": mil, **c_res})
            res["mean"].append({"dataset": name, "model": model, "config": cfg, "mil_type": mil, **{f"mean_{k}": v for k, v in boot["mean"].items()}})

        run_ds(y_t, y_p, y_b, ds)
        if split_region and regions is not None:
            for r in sorted(set(regions)):
                idx = [i for i, reg in enumerate(regions) if reg == r]
                if idx:
                    run_ds(y_t[idx], y_p[idx], y_b[idx], f"{ds}_{r}")
        res["success"] = True
    except Exception as e:
        res["error"] = str(e)
    return res


def analyze_all_experiments(
    base_dir,
    output_dir,
    n_boot=1000,
    mil_type=None,
    split_region=False,
    n_workers=None,
    models=None,
    exclude_models=None,
):
    os.makedirs(output_dir, exist_ok=True)
    n_workers = n_workers or mp.cpu_count()
    allowed = parse_model_list(models)
    excluded = parse_model_list(exclude_models)
    experiments = find_experiment_paths(base_dir)
    if ALLOWED_DATASETS is not None:
        ds_set = set(ALLOWED_DATASETS)
        experiments = [e for e in experiments if e[0] in ds_set]
    if mil_type:
        experiments = [e for e in experiments if e[3] == mil_type]
    experiments = [e for e in experiments if model_allowed(e[1], allowed, excluded)]
    jobs = [(e[0], e[1], e[2], e[3], e[4], n_boot, split_region) for e in experiments]
    all_pd, all_mean = [], []
    with ProcessPoolExecutor(max_workers=n_workers) as exc:
        for f in tqdm(
            as_completed({exc.submit(process_single_experiment, j): j for j in jobs}),
            total=len(jobs),
        ):
            r = f.result()
            if r["success"]:
                all_pd.extend(r["pd"])
                all_mean.extend(r["mean"])
    pd.DataFrame(all_pd).to_csv(os.path.join(output_dir, "results_per_disease.csv"), index=False)
    pd.DataFrame(all_mean).to_csv(os.path.join(output_dir, "results_mean.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CI Analysis for MIL Linear Probe")
    parser.add_argument("--base_dir", type=str, default="experiments/MIL_LP")
    parser.add_argument("--output_dir", type=str, default="analysis_results_bootstrap")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--mil_type", type=str, default=None)
    parser.add_argument("--split_region", action="store_true")
    parser.add_argument("--n_workers", type=int, default=None)
    add_model_filter_args(parser)
    args = parser.parse_args()
    analyze_all_experiments(
        args.base_dir,
        args.output_dir,
        args.n_bootstrap,
        args.mil_type,
        args.split_region,
        args.n_workers,
        args.models,
        args.exclude_models,
    )


if __name__ == "__main__":
    main()
