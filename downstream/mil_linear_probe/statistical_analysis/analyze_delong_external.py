import argparse
import ast
import importlib.util
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from delong_utils import compute_auc_with_ci, delong_test, paired_bootstrap_mean_test
from utils import parse_model_list, model_allowed, add_model_filter_args, included_classes


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
add_fdr_by_group = _stat.add_fdr_by_group
make_mean_auc_metric = _stat.make_mean_auc_metric
DEFAULT_N_PERMUTATION = _stat.DEFAULT_N_PERMUTATION
DEFAULT_N_BOOTSTRAP = _stat.DEFAULT_N_BOOTSTRAP
DEFAULT_SEED = _stat.DEFAULT_SEED
SIGNIFICANCE_METHODS = ("paired_permutation", "bootstrap_wilcoxon", "bootstrap_pvalue")

OUR_MODEL_FAMILY = {
    "our_ark_base",
    "our_rad_dino",
    "our_siglip2",
    "topk_v97_last",
    "topk_v83_last",
    "topk_v100_best",
}


def is_our_model_name(model_name, requested_our_model=None):
    if model_name in OUR_MODEL_FAMILY:
        return True
    if re.fullmatch(r"topk_v\d+_(last|best)", model_name):
        return True
    if requested_our_model is not None and model_name == requested_our_model:
        return True
    return False


def _paired_mean_auc_metric(paired: np.ndarray) -> float:
    """Mean AUC over paired ``[..., (y_true, y_score)]`` rows."""
    paired = np.asarray(paired)
    y_true = paired[..., 0]
    y_score = paired[..., 1]
    return make_mean_auc_metric(y_true)(y_score)


def _mean_auc_significance(
    y_true,
    y_pred_a,
    y_pred_b,
    *,
    significance_method,
    n_permutation,
    n_bootstrap,
    random_state,
):
    if significance_method == "paired_permutation":
        metric_fn = make_mean_auc_metric(y_true)
        pt = paired_permutation_test(
            metric_fn, y_pred_a, y_pred_b,
            n_resamples=n_permutation, random_state=random_state,
        )
        return pt, {"paired_permutation_p_value": pt["p_value"]}
    if significance_method == "bootstrap_wilcoxon":
        data_a = np.stack([y_true, y_pred_a], axis=-1)
        data_b = np.stack([y_true, y_pred_b], axis=-1)
        pt = paired_bootstrap_wilcoxon_test(
            _paired_mean_auc_metric, data_a, data_b,
            n_resamples=n_bootstrap, random_state=random_state,
        )
        return pt, {
            "bootstrap_wilcoxon_statistic": pt["statistic"],
            "bootstrap_wilcoxon_p_value": pt["p_value"],
            "bootstrap_wilcoxon_n": pt["n_bootstrap_effective"],
        }
    if significance_method == "bootstrap_pvalue":
        data_a = np.stack([y_true, y_pred_a], axis=-1)
        data_b = np.stack([y_true, y_pred_b], axis=-1)
        pt = paired_bootstrap_pvalue_test(
            _paired_mean_auc_metric, data_a, data_b,
            n_resamples=n_bootstrap, random_state=random_state,
        )
        return pt, {
            "bootstrap_pvalue_p_value": pt["p_value"],
            "bootstrap_pvalue_n": pt["n_bootstrap_effective"],
        }
    raise ValueError(f"unknown significance_method: {significance_method!r}")


def load_test_results(csv_path: str):
    df = pd.read_csv(csv_path)
    y_true = np.array([ast.literal_eval(gt) for gt in df["ground_truth"]])
    if "prediction_probs" not in df.columns:
        raise ValueError(f"{csv_path} 缺少 prediction_probs 列，无法计算 AUC。")
    y_prob = np.array([ast.literal_eval(pred) for pred in df["prediction_probs"]])
    return y_true, y_prob

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


def process_one(job):
    dataset_region, dataset, region, model, csv_path, alpha = job
    res = {
        "dataset": dataset_region,
        "source_dataset": dataset,
        "region": region,
        "model": model,
        "success": False,
        "per_disease": [],
        "mean": None,
    }
    try:
        y_t, y_p = load_test_results(csv_path)
        aucs = []
        for c in range(y_t.shape[1]):
            r = compute_auc_with_ci(y_t[:, c], y_p[:, c], alpha=alpha)
            res["per_disease"].append(
                {
                    "dataset": dataset_region,
                    "source_dataset": dataset,
                    "region": region,
                    "model": model,
                    "class_idx": c,
                    **r,
                }
            )
            if r["has_positives"] and not np.isnan(r["auc"]):
                aucs.append(r["auc"])
        res["mean"] = {
            "dataset": dataset_region,
            "source_dataset": dataset,
            "region": region,
            "model": model,
            "mean_auc": np.mean(aucs) if aucs else np.nan,
            "n_classes": len(aucs),
        }
        res["y_true"] = y_t
        res["y_prob"] = y_p
        res["success"] = True
    except Exception as exc:
        res["error"] = str(exc)
    return res


def pick_best_baseline(mean_rows, our_model):
    by_dataset = {}
    for row in mean_rows:
        by_dataset.setdefault(row["dataset"], []).append(row)

    picked = {}
    for dataset, rows in by_dataset.items():
        baseline_candidates = [
            r
            for r in rows
            if (not is_our_model_name(r["model"], requested_our_model=our_model))
            and (not np.isnan(r["mean_auc"]))
        ]
        if baseline_candidates:
            picked[dataset] = max(baseline_candidates, key=lambda x: x["mean_auc"])["model"]
    return picked


def analyze(
    results_dir,
    output_dir,
    our_model="our_ark_base",
    alpha=0.05,
    n_workers=None,
    n_permutation=DEFAULT_N_PERMUTATION,
    n_bootstrap=DEFAULT_N_BOOTSTRAP,
    significance_method="bootstrap_wilcoxon",
    random_state=DEFAULT_SEED,
    models=None,
    exclude_models=None,
):
    os.makedirs(output_dir, exist_ok=True)
    allowed = parse_model_list(models)
    excluded = parse_model_list(exclude_models)
    if allowed is not None and our_model not in allowed:
        raise ValueError(
            f"our_model '{our_model}' 不在 --models 白名单中，请将其加入或留空 --models"
        )
    if excluded is not None and our_model in excluded:
        raise ValueError(f"our_model '{our_model}' 被 --exclude_models 排除")
    jobs = [
        (dataset_region, dataset, region, model, csv_path, alpha)
        for dataset_region, dataset, region, model, csv_path in find_result_csvs(results_dir)
        if model_allowed(model, allowed, excluded)
    ]
    if not jobs:
        raise RuntimeError(f"未在 {results_dir} 下找到 test_results.csv")

    n_workers = n_workers or mp.cpu_count()
    auc_per_disease, auc_mean = [], []
    data_cache = {}
    errors = []

    if n_workers == 1:
        for job in tqdm(jobs):
            r = process_one(job)
            if r["success"]:
                auc_per_disease.extend(r["per_disease"])
                auc_mean.append(r["mean"])
                data_cache.setdefault(r["dataset"], {})[r["model"]] = r
            else:
                errors.append(r)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_map = {executor.submit(process_one, job): job for job in jobs}
            for future in tqdm(as_completed(future_map), total=len(future_map)):
                r = future.result()
                if r["success"]:
                    auc_per_disease.extend(r["per_disease"])
                    auc_mean.append(r["mean"])
                    data_cache.setdefault(r["dataset"], {})[r["model"]] = r
                else:
                    errors.append(r)

    # best baseline kept only as a display flag — inference is our-vs-ALL + FDR,
    # so there is no winner's-curse selection bias and no per-metric mismatch.
    best_baseline = pick_best_baseline(auc_mean, our_model=our_model)
    delong_rows = []
    skipped_rows = []

    for dataset in sorted(data_cache.keys()):
        if our_model not in data_cache[dataset]:
            skipped_rows.append(
                {"dataset": dataset, "reason": f"缺少 our 模型: {our_model}", "our_model": our_model}
            )
            continue
        our_data = data_cache[dataset][our_model]

        for baseline_model, base_data in data_cache[dataset].items():
            if baseline_model == our_model:
                continue
            is_best = baseline_model == best_baseline.get(dataset)
            if our_data["y_true"].shape != base_data["y_true"].shape:
                skipped_rows.append({
                    "dataset": dataset,
                    "reason": f"shape 不一致: {our_data['y_true'].shape} vs {base_data['y_true'].shape}",
                    "our_model": our_model, "baseline_model": baseline_model,
                })
                continue
            if not np.array_equal(our_data["y_true"], base_data["y_true"]):
                skipped_rows.append({
                    "dataset": dataset, "reason": "ground_truth 不一致，跳过 DeLong",
                    "our_model": our_model, "baseline_model": baseline_model,
                })
                continue

            n_classes = our_data["y_true"].shape[1]
            keep = included_classes(dataset, n_classes, region=our_data.get("region"))
            n_keep = int(keep.sum())
            if n_keep == 0:
                continue
            for c in range(n_classes):
                if not keep[c]:
                    continue
                tr = delong_test(
                    our_data["y_true"][:, c],
                    our_data["y_prob"][:, c],
                    base_data["y_prob"][:, c],
                    alpha=alpha,
                )
                delong_rows.append({
                    "dataset": dataset, "class_idx": c,
                    "model_a": our_model, "model_b": baseline_model,
                    "is_best_baseline": is_best, **tr, "is_mean": False,
                })

            if n_keep == 1:
                mean_row = delong_rows[-1].copy()
                mean_row["class_idx"] = None
                mean_row["is_mean"] = True
                mean_row["test_method"] = "delong"
                mean_row["n_permutation"] = np.nan
            else:
                ytk = our_data["y_true"][:, keep]
                ypk_a = our_data["y_prob"][:, keep]
                ypk_b = base_data["y_prob"][:, keep]
                metric_fn = make_mean_auc_metric(ytk)
                auc_a = metric_fn(ypk_a)
                auc_b = metric_fn(ypk_b)
                pt, method_cols = _mean_auc_significance(
                    ytk, ypk_a, ypk_b,
                    significance_method=significance_method,
                    n_permutation=n_permutation,
                    n_bootstrap=n_bootstrap,
                    random_state=random_state,
                )
                # Effect-size CI for the mean-AUC difference (permutation gives
                # only a p-value). Paired bootstrap that resamples y_true and
                # y_pred together (mean-AUC needs labels resampled with preds).
                bt = paired_bootstrap_mean_test(
                    ytk, ypk_a, ypk_b,
                    n_bootstrap=n_bootstrap, random_state=random_state,
                )
                mean_row = {
                    "dataset": dataset, "class_idx": None,
                    "model_a": our_model, "model_b": baseline_model,
                    "is_best_baseline": is_best,
                    "auc_a": auc_a, "auc_b": auc_b,
                    "auc_diff": pt["observed_diff"],
                    "p_value": pt["p_value"],
                    "significant": pt["p_value"] < alpha,
                    "ci_low": bt["ci_low"], "ci_high": bt["ci_high"],
                    "se_diff": bt["se_diff"],
                    "test_method": significance_method,
                    "n_permutation": n_permutation,
                    "n_bootstrap": n_bootstrap,
                    "is_mean": True,
                }
                mean_row.update(method_cols)
            for k in ("test_method", "n_permutation", "n_bootstrap"):
                mean_row.setdefault(k, np.nan if k in ("n_permutation", "n_bootstrap") else "delong")
            delong_rows.append(mean_row)

    pd.DataFrame(auc_per_disease).to_csv(os.path.join(output_dir, "auc_per_disease.csv"), index=False)
    pd.DataFrame(auc_mean).to_csv(os.path.join(output_dir, "auc_mean.csv"), index=False)
    if delong_rows:
        delong_df = pd.DataFrame(delong_rows)
        # BH FDR within each (dataset, is_mean) family.
        delong_df = add_fdr_by_group(delong_df, group_cols=["dataset", "is_mean"])
        delong_df.to_csv(os.path.join(output_dir, "delong_results.csv"), index=False)
    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(os.path.join(output_dir, "delong_skipped.csv"), index=False)
    if errors:
        pd.DataFrame(errors).to_csv(os.path.join(output_dir, "analysis_errors.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="对 results 目录执行 AUC + DeLong 显著性分析")
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
    parser.add_argument("--our_model", type=str, default="topk_v97_last")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--n_permutation", type=int, default=DEFAULT_N_PERMUTATION)
    parser.add_argument("--n_bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--significance_method", choices=SIGNIFICANCE_METHODS, default="bootstrap_wilcoxon")
    parser.add_argument("--random_state", type=int, default=DEFAULT_SEED)
    add_model_filter_args(parser)
    args = parser.parse_args()
    analyze(
        args.results_dir,
        args.output_dir,
        args.our_model,
        args.alpha,
        args.n_workers,
        args.n_permutation,
        args.n_bootstrap,
        args.significance_method,
        args.random_state,
        args.models,
        args.exclude_models,
    )


if __name__ == "__main__":
    main()
