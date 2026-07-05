"""In-domain significance testing for threshold-dependent metrics (F1/MCC/b-acc).

DeLong only applies to AUC, so the two ``analyze_*_delong.py`` scripts cover the
AUC comparisons. This script is the parallel pipeline for the *threshold*
metrics, which have no closed-form variance: it compares ``reference_model``
against every other model with a **paired permutation test** (same engine as the
mean-AUC comparison), per disease and at the macro mean, then applies BH-FDR.

Metrics use the fixed val-threshold binary ``prediction`` column — identical to
how F1/MCC are computed in ``bootstrap.py`` — so the point estimates here match
the bootstrap CI tables.

Output (in ``output_dir``):
    threshold_results.csv   long format, one row per (dataset, metric, class/mean)

Raw ``p_value`` (+ ``significant``) is always present; ``p_value_fdr`` /
``significant_fdr`` are added as separate columns (BH within dataset×metric×level).
"""

import os
import argparse
import importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import load_results, find_experiment_paths, parse_model_list, model_allowed, add_model_filter_args, included_classes


def _load_repo_stat_test():
    """Load repo-root utils/stat_test.py by absolute path to avoid namespace
    clash with the local statistical_analysis/utils.py."""
    path = Path(__file__).resolve().parents[3] / "utils" / "stat_test.py"
    spec = importlib.util.spec_from_file_location("repo_stat_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_stat = _load_repo_stat_test()
paired_permutation_test = _stat.paired_permutation_test
paired_bootstrap_wilcoxon_test = _stat.paired_bootstrap_wilcoxon_test
paired_bootstrap_pvalue_test = _stat.paired_bootstrap_pvalue_test
paired_bootstrap_threshold_diff_ci = _stat.paired_bootstrap_threshold_diff_ci
add_fdr_by_group = _stat.add_fdr_by_group
THRESHOLD_METRIC_FACTORIES = _stat.THRESHOLD_METRIC_FACTORIES
DEFAULT_N_PERMUTATION = _stat.DEFAULT_N_PERMUTATION
DEFAULT_N_BOOTSTRAP = _stat.DEFAULT_N_BOOTSTRAP
DEFAULT_SEED = _stat.DEFAULT_SEED

DEFAULT_METRICS = ["f1", "mcc", "bacc"]
SIGNIFICANCE_METHODS = ("paired_permutation", "bootstrap_wilcoxon", "bootstrap_pvalue")

# 若只想分析部分数据集，在这里列出它们的名字（base_dir 下的一级目录名）
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


def _paired_threshold_metric(metric):
    """Metric over paired ``[..., (y_true, y_pred)]`` rows for bootstrap tests."""

    def metric_fn(pair_block):
        pair_block = np.asarray(pair_block)
        y_true = pair_block[..., 0]
        y_pred = pair_block[..., 1]
        return THRESHOLD_METRIC_FACTORIES[metric](y_true)(y_pred)

    return metric_fn


def _significance_test(metric, y_true_block, y_bin_a, y_bin_b, n_perm, n_boot, seed, method):
    if method == "paired_permutation":
        metric_fn = THRESHOLD_METRIC_FACTORIES[metric](y_true_block)
        pt = paired_permutation_test(
            metric_fn, y_bin_a, y_bin_b, n_resamples=n_perm, random_state=seed
        )
        return pt, {"paired_permutation_p_value": pt["p_value"]}

    if method == "bootstrap_wilcoxon":
        metric_fn = _paired_threshold_metric(metric)
        data_a = np.stack([y_true_block, y_bin_a], axis=-1)
        data_b = np.stack([y_true_block, y_bin_b], axis=-1)
        pt = paired_bootstrap_wilcoxon_test(
            metric_fn, data_a, data_b, n_resamples=n_boot, random_state=seed
        )
        return pt, {
            "bootstrap_wilcoxon_statistic": pt["statistic"],
            "bootstrap_wilcoxon_p_value": pt["p_value"],
            "bootstrap_wilcoxon_n": pt["n_bootstrap_effective"],
        }

    if method == "bootstrap_pvalue":
        metric_fn = _paired_threshold_metric(metric)
        data_a = np.stack([y_true_block, y_bin_a], axis=-1)
        data_b = np.stack([y_true_block, y_bin_b], axis=-1)
        pt = paired_bootstrap_pvalue_test(
            metric_fn, data_a, data_b, n_resamples=n_boot, random_state=seed
        )
        return pt, {
            "bootstrap_pvalue_p_value": pt["p_value"],
            "bootstrap_pvalue_n": pt["n_bootstrap_effective"],
        }

    raise ValueError(f"unknown significance_method: {method!r}")


def _permutation_row(metric, y_true_block, y_bin_a, y_bin_b, n_perm, n_boot, seed, alpha, significance_method):
    """Run one paired comparison for ``metric`` on a class block.

    ``y_true_block`` / ``y_bin_*`` are (n_samples, k) with k=1 for a single
    disease or k=n_classes for the macro mean. The selected significance method
    gives the p-value; an alignment-safe paired bootstrap gives the effect-size
    CI (ci_low/ci_high of value_a - value_b). Returns a dict of result fields,
    or None when the metric is undefined on this block (invalid class).
    """
    metric_fn = THRESHOLD_METRIC_FACTORIES[metric](y_true_block)
    value_a = metric_fn(y_bin_a)
    value_b = metric_fn(y_bin_b)
    if not (np.isfinite(value_a) and np.isfinite(value_b)):
        return None
    pt, method_cols = _significance_test(
        metric, y_true_block, y_bin_a, y_bin_b, n_perm, n_boot, seed, significance_method
    )
    dci = paired_bootstrap_threshold_diff_ci(
        metric, y_true_block, y_bin_a, y_bin_b, n_resamples=n_boot, random_state=seed
    )
    row = {
        "metric": metric,
        "value_a": float(value_a),
        "value_b": float(value_b),
        "value_diff": pt["observed_diff"],
        "ci_low": dci["diff_ci_low"],
        "ci_high": dci["diff_ci_high"],
        "p_value": pt["p_value"],
        "significant": pt["p_value"] < alpha,
        "test_method": significance_method,
        "n_permutation": n_perm,
        "n_bootstrap": n_boot,
    }
    row.update(method_cols)
    return row


def compare_pair(job):
    """Compare reference vs one model on all metrics, per-disease and mean."""
    (ds, reference_model, model_b, y_true, y_bin_ref, y_bin_b,
     metrics, n_perm, n_boot, seed, alpha, significance_method) = job
    rows = []
    n_classes = y_true.shape[1]
    keep = included_classes(ds, n_classes)  # per-region disease exclusion
    if not keep.any():
        return rows
    for metric in metrics:
        # per-disease (kept classes only)
        for c in range(n_classes):
            if not keep[c]:
                continue
            block = slice(c, c + 1)
            r = _permutation_row(
                metric, y_true[:, block], y_bin_ref[:, block], y_bin_b[:, block],
                n_perm, n_boot, seed, alpha, significance_method,
            )
            if r is None:
                continue
            rows.append({
                "dataset": ds, "class_idx": c,
                "model_a": reference_model, "model_b": model_b,
                "is_mean": False, **r,
            })
        # macro mean over kept classes
        r = _permutation_row(
            metric, y_true[:, keep], y_bin_ref[:, keep], y_bin_b[:, keep],
            n_perm, n_boot, seed, alpha, significance_method,
        )
        if r is not None:
            rows.append({
                "dataset": ds, "class_idx": None,
                "model_a": reference_model, "model_b": model_b,
                "is_mean": True, **r,
            })
    return rows


def _load_one(args):
    ds, model, path = args
    try:
        y_t, _, y_b, _ = load_results(path)
        if y_b is None:
            raise ValueError(f"{path} 缺少 prediction 列；请先重跑 test 写入二值预测。")
        return {"dataset": ds, "model": model, "y_true": y_t, "y_bin": y_b, "success": True}
    except Exception as e:
        return {"dataset": ds, "model": model, "success": False, "error": str(e)}


def analyze_all_experiments(
    base_dir,
    output_dir,
    reference_model,
    mil_type=None,
    n_workers=None,
    alpha=0.05,
    metrics=None,
    n_permutation=DEFAULT_N_PERMUTATION,
    n_bootstrap=DEFAULT_N_BOOTSTRAP,
    significance_method="bootstrap_wilcoxon",
    random_state=DEFAULT_SEED,
    models=None,
    exclude_models=None,
):
    os.makedirs(output_dir, exist_ok=True)
    n_workers = n_workers or mp.cpu_count()
    metrics = metrics or DEFAULT_METRICS
    allowed = parse_model_list(models)
    excluded = parse_model_list(exclude_models)
    if allowed is not None and reference_model not in allowed:
        raise ValueError(
            f"reference_model '{reference_model}' 不在 --models 白名单中，请将其加入或留空 --models"
        )
    if excluded is not None and reference_model in excluded:
        raise ValueError(f"reference_model '{reference_model}' 被 --exclude_models 排除")

    experiments = find_experiment_paths(base_dir)
    if ALLOWED_DATASETS is not None:
        ds_set = set(ALLOWED_DATASETS)
        experiments = [e for e in experiments if e[0] in ds_set]
    if mil_type:
        experiments = [e for e in experiments if e[3] == mil_type]
    experiments = [e for e in experiments if model_allowed(e[1], allowed, excluded)]

    # 1) load all experiments in parallel
    load_jobs = [(e[0], e[1], e[4]) for e in experiments]
    data_cache = {}
    with ProcessPoolExecutor(max_workers=n_workers) as exc:
        for f in tqdm(as_completed({exc.submit(_load_one, j): j for j in load_jobs}),
                      total=len(load_jobs), desc="load"):
            r = f.result()
            if r["success"]:
                data_cache.setdefault(r["dataset"], {})[r["model"]] = r

    # 2) build comparison jobs (reference vs every other model, matching shapes)
    jobs = []
    for ds in sorted(data_cache.keys()):
        if reference_model not in data_cache[ds]:
            continue
        ref = data_cache[ds][reference_model]
        for m, mdata in data_cache[ds].items():
            if m == reference_model:
                continue
            if ref["y_true"].shape != mdata["y_true"].shape:
                continue
            # Paired tests require identical labels/order for both models;
            # compare_pair scores model_b's preds against ref's y_true, so a
            # mismatch would silently mispair.
            if not np.array_equal(ref["y_true"], mdata["y_true"]):
                print(f"[warn] {ds}: ground_truth 不一致，跳过配对比较 {reference_model} vs {m}")
                continue
            jobs.append((
                ds, reference_model, m, ref["y_true"], ref["y_bin"], mdata["y_bin"],
                metrics, n_permutation, n_bootstrap, random_state, alpha,
                significance_method,
            ))

    # 3) run comparisons in parallel
    all_rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as exc:
        for f in tqdm(as_completed({exc.submit(compare_pair, j): j for j in jobs}),
                      total=len(jobs), desc="permute"):
            all_rows.extend(f.result())

    if not all_rows:
        print("[warn] 没有产生任何比较结果（检查 reference_model / 数据集是否存在）。")
        return
    df = pd.DataFrame(all_rows)
    # BH FDR within each (dataset, metric, is_mean) family — each region is a
    # distinct clinical scenario, each metric a distinct hypothesis, and
    # per-disease vs macro-mean are different hypothesis levels.
    df = add_fdr_by_group(df, group_cols=["dataset", "metric", "is_mean"])
    df.to_csv(os.path.join(output_dir, "threshold_results.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Paired permutation tests for threshold metrics (F1/MCC/b-acc) — MIL LP"
    )
    parser.add_argument("--base_dir", type=str, default="experiments/MIL_LP")
    parser.add_argument("--output_dir", type=str, default="analysis_results_threshold")
    parser.add_argument("--reference_model", type=str, required=True)
    parser.add_argument("--mil_type", type=str, default=None)
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--metrics", type=str, default=",".join(DEFAULT_METRICS),
                        help="逗号分隔，子集 of f1,mcc,bacc")
    parser.add_argument("--n_permutation", type=int, default=DEFAULT_N_PERMUTATION)
    parser.add_argument("--n_bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--significance_method", choices=SIGNIFICANCE_METHODS, default="bootstrap_wilcoxon")
    parser.add_argument("--random_state", type=int, default=DEFAULT_SEED)
    add_model_filter_args(parser)
    args = parser.parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    bad = [m for m in metrics if m not in THRESHOLD_METRIC_FACTORIES]
    if bad:
        raise ValueError(f"未知 metric: {bad}；可选 {list(THRESHOLD_METRIC_FACTORIES)}")
    analyze_all_experiments(
        args.base_dir,
        args.output_dir,
        args.reference_model,
        args.mil_type,
        args.n_workers,
        args.alpha,
        metrics,
        args.n_permutation,
        args.n_bootstrap,
        args.significance_method,
        args.random_state,
        args.models,
        args.exclude_models,
    )


if __name__ == "__main__":
    main()
