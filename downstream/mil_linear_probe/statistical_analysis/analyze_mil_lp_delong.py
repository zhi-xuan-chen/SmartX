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
from delong_utils import compute_auc_with_ci, delong_test, paired_bootstrap_mean_test


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

def process_single_experiment(args):
    ds, model, cfg, mil, path, split_region = args
    res = {"dataset": ds, "model": model, "config": cfg, "mil_type": mil, "success": False, "per_disease": [], "mean": None, "y_true": None, "y_pred": None}
    try:
        y_t, y_p, _, _ = load_results(path)
        res["y_true"], res["y_pred"] = y_t, y_p
        aucs = []
        for c in range(y_t.shape[1]):
            r = compute_auc_with_ci(y_t[:, c], y_p[:, c])
            res["per_disease"].append({"dataset": ds, "model": model, "config": cfg, "mil_type": mil, "class_idx": c, **r})
            if r["has_positives"] and not np.isnan(r["auc"]): aucs.append(r["auc"])
        res["mean"] = {"dataset": ds, "model": model, "config": cfg, "mil_type": mil, "mean_auc": np.mean(aucs) if aucs else np.nan, "n_classes": len(aucs)}
        res["success"] = True
    except Exception as e: res["error"] = str(e)
    return res


def analyze_all_experiments(
    base_dir,
    output_dir,
    reference_model,
    mil_type=None,
    split_region=False,
    n_workers=None,
    alpha=0.05,
    n_permutation=DEFAULT_N_PERMUTATION,
    n_bootstrap=DEFAULT_N_BOOTSTRAP,
    significance_method="bootstrap_wilcoxon",
    random_state=DEFAULT_SEED,
    models=None,
    exclude_models=None,
):
    os.makedirs(output_dir, exist_ok=True)
    n_workers = n_workers or mp.cpu_count()
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
    if mil_type: experiments = [e for e in experiments if e[3] == mil_type]
    experiments = [e for e in experiments if model_allowed(e[1], allowed, excluded)]
    jobs = [(e[0], e[1], e[2], e[3], e[4], split_region) for e in experiments]
    all_pd, all_mean, data_cache = [], [], {}
    with ProcessPoolExecutor(max_workers=n_workers) as exc:
        for f in tqdm(as_completed({exc.submit(process_single_experiment, j): j for j in jobs}), total=len(jobs)):
            r = f.result()
            if r["success"]:
                all_pd.extend(r["per_disease"]); all_mean.append(r["mean"])
                if r["dataset"] not in data_cache: data_cache[r["dataset"]] = {}
                data_cache[r["dataset"]][r["model"]] = r
    
    all_delong = []
    for ds in tqdm(sorted(data_cache.keys())):
        if reference_model not in data_cache[ds]: continue
        ref = data_cache[ds][reference_model]
        for m, mdata in data_cache[ds].items():
            if m == reference_model:
                continue
            if ref["y_true"].shape != mdata["y_true"].shape or ref["y_pred"].shape != mdata["y_pred"].shape:
                continue
            # Paired tests require the SAME labels/samples in the same order for
            # both models; mismatched ground_truth would silently mispair.
            if not np.array_equal(ref["y_true"], mdata["y_true"]):
                print(f"[warn] {ds}: ground_truth 不一致，跳过配对比较 {reference_model} vs {m}")
                continue
            n_classes = ref["y_true"].shape[1]
            keep = included_classes(ds, n_classes)  # per-region disease exclusion
            n_keep = int(keep.sum())
            if n_keep == 0:
                continue
            for c in range(n_classes):
                if not keep[c]:
                    continue
                tr = delong_test(ref["y_true"][:, c], ref["y_pred"][:, c], mdata["y_pred"][:, c], alpha)
                all_delong.append({"dataset": ds, "class_idx": c, "model_a": reference_model, "model_b": m, **tr, "is_mean": False})

            if n_keep == 1:
                mean_result = all_delong[-1].copy()
                mean_result["class_idx"] = None
                mean_result["is_mean"] = True
                mean_result["test_method"] = "delong"
                mean_result["n_permutation"] = np.nan
            else:
                ytk, ypk_a, ypk_b = ref["y_true"][:, keep], ref["y_pred"][:, keep], mdata["y_pred"][:, keep]
                metric_fn = make_mean_auc_metric(ytk)
                auc_a = metric_fn(ypk_a)
                auc_b = metric_fn(ypk_b)
                pt, method_cols = _mean_auc_significance(
                    ytk,
                    ypk_a,
                    ypk_b,
                    significance_method=significance_method,
                    n_permutation=n_permutation,
                    n_bootstrap=n_bootstrap,
                    random_state=random_state,
                )
                # Effect-size CI for the mean-AUC difference (the permutation test
                # only gives a p-value). Paired bootstrap that resamples y_true
                # and y_pred *together* (mean-AUC needs the labels resampled with
                # the predictions; a generic predictions-only resampler would
                # break the label alignment). ci_low/ci_high keep their
                # "CI of (auc_a - auc_b)" meaning, computed by bootstrap here vs
                # DeLong per-class.
                bt = paired_bootstrap_mean_test(
                    ytk,
                    ypk_a,
                    ypk_b,
                    n_bootstrap=n_bootstrap,
                    random_state=random_state,
                )
                mean_result = {
                    "dataset": ds,
                    "class_idx": None,
                    "model_a": reference_model,
                    "model_b": m,
                    "auc_a": auc_a,
                    "auc_b": auc_b,
                    "auc_diff": pt["observed_diff"],
                    "p_value": pt["p_value"],
                    "significant": pt["p_value"] < alpha,
                    "ci_low": bt["ci_low"],
                    "ci_high": bt["ci_high"],
                    "se_diff": bt["se_diff"],
                    "test_method": significance_method,
                    "n_permutation": n_permutation,
                    "n_bootstrap": n_bootstrap,
                    "is_mean": True,
                }
                mean_result.update(method_cols)
            for k in ("test_method", "n_permutation", "n_bootstrap"):
                mean_result.setdefault(k, np.nan if k in ("n_permutation", "n_bootstrap") else "delong")
            all_delong.append(mean_result)

    pd.DataFrame(all_pd).to_csv(os.path.join(output_dir, "auc_per_disease.csv"), index=False)
    pd.DataFrame(all_mean).to_csv(os.path.join(output_dir, "auc_mean.csv"), index=False)
    if all_delong:
        delong_df = pd.DataFrame(all_delong)
        # BH FDR within each (dataset, is_mean) family.
        # Rationale: each anatomical region (dataset) is a distinct clinical
        # scenario, and per-disease vs mean-AUC are different hypothesis levels.
        delong_df = add_fdr_by_group(delong_df, group_cols=["dataset", "is_mean"])
        delong_df.to_csv(os.path.join(output_dir, "delong_results.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Analyze MIL Linear Probe results with DeLong Test")
    parser.add_argument("--base_dir", type=str, default="experiments/MIL_LP")
    parser.add_argument("--output_dir", type=str, default="analysis_results_delong")
    parser.add_argument("--reference_model", type=str, required=True)
    parser.add_argument("--mil_type", type=str, default=None)
    parser.add_argument("--split_region", action="store_true")
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n_permutation", type=int, default=DEFAULT_N_PERMUTATION)
    parser.add_argument("--n_bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--significance_method", choices=SIGNIFICANCE_METHODS, default="bootstrap_wilcoxon")
    parser.add_argument("--random_state", type=int, default=DEFAULT_SEED)
    add_model_filter_args(parser)
    args = parser.parse_args()
    analyze_all_experiments(
        args.base_dir,
        args.output_dir,
        args.reference_model,
        args.mil_type,
        args.split_region,
        args.n_workers,
        args.alpha,
        args.n_permutation,
        args.n_bootstrap,
        args.significance_method,
        args.random_state,
        args.models,
        args.exclude_models,
    )

if __name__ == "__main__": main()
