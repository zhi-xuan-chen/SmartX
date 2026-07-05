"""External significance testing for threshold metrics (F1/MCC/b-acc).

Parallel to ``analyze_delong_external.py`` but for the threshold-dependent
metrics that DeLong cannot handle. For each external cohort×region it compares
``our_model`` against the **best baseline** (the non-our model with the highest
mean AUC — identical selection to the DeLong-external script, so the competitor
lines up across all metrics) using a paired permutation test, per disease and at
the macro mean, then applies BH-FDR.

Metrics use the val-threshold binary ``prediction`` column (same as the
bootstrap CIs). Output: ``threshold_results.csv`` + ``threshold_skipped.csv``.
"""

import argparse
import ast
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import parse_model_list, model_allowed, add_model_filter_args
from analyze_delong_external import (
    find_result_csvs,
    is_our_model_name,
    pick_best_baseline,
    _stat,
)
from analyze_threshold_metrics import compare_pair, DEFAULT_METRICS, THRESHOLD_METRIC_FACTORIES
from analyze_threshold_metrics import SIGNIFICANCE_METHODS

make_mean_auc_metric = _stat.make_mean_auc_metric
add_fdr_by_group = _stat.add_fdr_by_group
DEFAULT_N_PERMUTATION = _stat.DEFAULT_N_PERMUTATION
DEFAULT_N_BOOTSTRAP = _stat.DEFAULT_N_BOOTSTRAP
DEFAULT_SEED = _stat.DEFAULT_SEED


def load_test_results(csv_path: str):
    df = pd.read_csv(csv_path)
    y_true = np.array([ast.literal_eval(gt) for gt in df["ground_truth"]])
    if "prediction" not in df.columns:
        raise ValueError(f"{csv_path} 缺少 prediction 列；请重跑 test 写入二值预测。")
    y_bin = np.array([ast.literal_eval(p) for p in df["prediction"]])
    y_prob = None
    if "prediction_probs" in df.columns:
        y_prob = np.array([ast.literal_eval(p) for p in df["prediction_probs"]])
    return y_true, y_bin, y_prob


def _load_one(job):
    dataset_region, dataset, region, model, csv_path = job
    res = {"dataset": dataset_region, "source_dataset": dataset, "region": region,
           "model": model, "success": False}
    try:
        y_t, y_b, y_p = load_test_results(csv_path)
        res.update({"y_true": y_t, "y_bin": y_b, "y_prob": y_p, "success": True})
        # mean AUC only for best-baseline selection (matches DeLong-external)
        if y_p is not None:
            res["mean_auc"] = make_mean_auc_metric(y_t)(y_p)
        else:
            res["mean_auc"] = np.nan
    except Exception as exc:
        res["error"] = str(exc)
    return res


def analyze(
    results_dir,
    output_dir,
    our_model="topk_v19_last",
    alpha=0.05,
    n_workers=None,
    metrics=None,
    n_permutation=DEFAULT_N_PERMUTATION,
    n_bootstrap=DEFAULT_N_BOOTSTRAP,
    significance_method="bootstrap_wilcoxon",
    random_state=DEFAULT_SEED,
    models=None,
    exclude_models=None,
):
    os.makedirs(output_dir, exist_ok=True)
    metrics = metrics or DEFAULT_METRICS
    allowed = parse_model_list(models)
    excluded = parse_model_list(exclude_models)
    if allowed is not None and our_model not in allowed:
        raise ValueError(f"our_model '{our_model}' 不在 --models 白名单中")
    if excluded is not None and our_model in excluded:
        raise ValueError(f"our_model '{our_model}' 被 --exclude_models 排除")

    load_jobs = [
        (dr, ds, rg, model, path)
        for dr, ds, rg, model, path in find_result_csvs(results_dir)
        if model_allowed(model, allowed, excluded)
    ]
    if not load_jobs:
        raise RuntimeError(f"未在 {results_dir} 下找到 test_results.csv")

    n_workers = n_workers or mp.cpu_count()
    data_cache, mean_rows, errors = {}, [], []
    with ProcessPoolExecutor(max_workers=n_workers) as exc:
        for f in tqdm(as_completed({exc.submit(_load_one, j): j for j in load_jobs}),
                      total=len(load_jobs), desc="load"):
            r = f.result()
            if r["success"]:
                data_cache.setdefault(r["dataset"], {})[r["model"]] = r
                mean_rows.append({"dataset": r["dataset"], "model": r["model"],
                                  "mean_auc": r["mean_auc"]})
            else:
                errors.append(r)

    # best baseline kept only as a display flag — inference is our-vs-ALL + FDR,
    # so there is no winner's-curse selection bias and no per-metric mismatch.
    best_baseline = pick_best_baseline(mean_rows, our_model=our_model)

    jobs, skipped = [], []
    for dataset in sorted(data_cache.keys()):
        if our_model not in data_cache[dataset]:
            skipped.append({"dataset": dataset, "reason": f"缺少 our 模型: {our_model}"})
            continue
        our_d = data_cache[dataset][our_model]
        for baseline_model, base_d in data_cache[dataset].items():
            if baseline_model == our_model:
                continue
            if our_d["y_true"].shape != base_d["y_true"].shape:
                skipped.append({"dataset": dataset, "reason": "shape 不一致",
                                "baseline_model": baseline_model})
                continue
            if not np.array_equal(our_d["y_true"], base_d["y_true"]):
                skipped.append({"dataset": dataset, "reason": "ground_truth 不一致",
                                "baseline_model": baseline_model})
                continue
            jobs.append((
                dataset, our_model, baseline_model, our_d["y_true"],
                our_d["y_bin"], base_d["y_bin"], metrics, n_permutation, n_bootstrap,
                random_state, alpha, significance_method,
            ))

    all_rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as exc:
        fut = {exc.submit(compare_pair, j): j for j in jobs}
        for f in tqdm(as_completed(fut), total=len(jobs), desc="permute"):
            job = fut[f]
            dataset, baseline_model = job[0], job[2]
            src = data_cache[dataset][our_model]
            is_best = baseline_model == best_baseline.get(dataset)
            for row in f.result():
                row["source_dataset"] = src["source_dataset"]
                row["region"] = src["region"]
                row["is_best_baseline"] = is_best
                all_rows.append(row)

    if all_rows:
        df = pd.DataFrame(all_rows)
        df = add_fdr_by_group(df, group_cols=["dataset", "metric", "is_mean"])
        df.to_csv(os.path.join(output_dir, "threshold_results.csv"), index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(os.path.join(output_dir, "threshold_skipped.csv"), index=False)
    if errors:
        pd.DataFrame(errors).to_csv(os.path.join(output_dir, "threshold_errors.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(
        description="对 results 目录执行 threshold 指标 (F1/MCC/b-acc) 的置换检验"
    )
    parser.add_argument("--results_dir", type=str,
                        default="${SMARTX_REPO_ROOT}/downstream/mil_linear_probe/results")
    parser.add_argument("--output_dir", type=str,
                        default="${SMARTX_REPO_ROOT}/downstream/mil_linear_probe/analysis_results/results_region")
    parser.add_argument("--our_model", type=str, default="topk_v19_last")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n_workers", type=int, default=None)
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
    analyze(
        args.results_dir,
        args.output_dir,
        args.our_model,
        args.alpha,
        args.n_workers,
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
