"""
MIL Linear Probe Results Analysis with Bootstrapping and T-Tests.

This script analyzes MIL linear probe test results across datasets, models,
and MIL aggregation types, computing per-disease metrics with bootstrap
confidence intervals and performing 2-sided t-tests comparing models to
a reference model + reference MIL type.
computing per-disease metrics with bootstrap confidence intervals and
performing 2-sided t-tests comparing models to a reference.

Uses multiprocessing for faster execution.

Usage:
    python analyze_mil_lp_bootstrap.py \
        --base_dir experiments/MIL_LP \
        --output_dir analysis_results \
        --n_bootstrap 1000 \
        --reference_model topk_v77_last \
        --mil_type clam \
        --n_workers 8
"""

import os
import ast
import argparse
from glob import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    f1_score,
    average_precision_score,
    matthews_corrcoef,
)
from tqdm import tqdm


# Dataset class counts（可选，不用强依赖）
DATASET_NUM_CLASSES = {
    "private_a": 46,
}

# Reference model for t-test comparison
REFERENCE_MODEL = "topk_v77_last"


def load_results(csv_path: str) -> tuple:
    """
    Load test results from CSV file.
    
    Args:
        csv_path: Path to test_results.csv
        
    Returns:
        Tuple of (ground_truth, prediction_probs, binary_predictions) as numpy arrays
        Shape: (n_samples, n_classes)
    """
    df = pd.read_csv(csv_path)
    
    # Parse string representations of lists
    ground_truth = np.array([ast.literal_eval(gt) for gt in df["ground_truth"]])
    binary_predictions = np.array([ast.literal_eval(pred) for pred in df["prediction"]])
    
    # Try to load probability predictions if available
    prediction_probs = None
    if "prediction_probs" in df.columns:
        prediction_probs = np.array([ast.literal_eval(pred) for pred in df["prediction_probs"]])
    
    # Optional: region 信息（若存在，用于后续分组分析）
    regions = df["region"].tolist() if "region" in df.columns else None
    
    return ground_truth, prediction_probs, binary_predictions, regions


def compute_metrics_single_class(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray = None,
    y_pred_binary: np.ndarray = None,
) -> dict:
    """
    Compute metrics for a single class/disease.
    
    Args:
        y_true: Ground truth labels (n_samples,)
        y_pred_probs: Probability predictions (n_samples,) - for AUC, AP
        y_pred_binary: Binary predictions (n_samples,) - fallback if no probs
        
    Returns:
        Dictionary with AUC, F1, AP, MCC values
    """
    metrics = {}
    
    # Skip classes with no positive samples
    if not np.any(y_true):
        return {
            "auc": None,
            "f1": None,
            "ap": None,
            "mcc": None,
            "has_positives": False,
        }
    
    metrics["has_positives"] = True
    
    # Use probability predictions if available
    y_pred = y_pred_probs if y_pred_probs is not None else y_pred_binary
    use_probs = y_pred_probs is not None
    
    # AUC - requires both classes present
    if len(np.unique(y_true)) < 2:
        metrics["auc"] = np.nan
    else:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_pred)
        except ValueError:
            metrics["auc"] = np.nan
    
    # AP - requires positive samples
    try:
        metrics["ap"] = average_precision_score(y_true, y_pred)
    except ValueError:
        metrics["ap"] = np.nan
    
    # For F1 and MCC: use optimal threshold from ROC curve if we have probabilities
    if use_probs and len(np.unique(y_true)) >= 2:
        try:
            # Find optimal threshold using Youden's J statistic
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            youden_j = tpr - fpr
            optimal_idx = np.argmax(youden_j)
            optimal_threshold = thresholds[optimal_idx]
            
            # Binarize predictions using optimal threshold
            y_pred_optimal = (y_pred > optimal_threshold).astype(int)
            
            # F1 with optimal threshold
            metrics["f1"] = f1_score(y_true, y_pred_optimal, zero_division=0)
            
            # MCC with optimal threshold
            try:
                metrics["mcc"] = matthews_corrcoef(y_true, y_pred_optimal)
            except ValueError:
                metrics["mcc"] = np.nan
                
        except Exception:
            # Fallback to using predictions directly
            y_pred_binary_final = y_pred_binary if y_pred_binary is not None else (y_pred > 0.5).astype(int)
            metrics["f1"] = f1_score(y_true, y_pred_binary_final, zero_division=0)
            try:
                metrics["mcc"] = matthews_corrcoef(y_true, y_pred_binary_final)
            except ValueError:
                metrics["mcc"] = np.nan
    else:
        # Use binary predictions directly
        y_pred_binary_final = y_pred_binary if y_pred_binary is not None else (y_pred > 0.5).astype(int)
        metrics["f1"] = f1_score(y_true, y_pred_binary_final, zero_division=0)
        try:
            metrics["mcc"] = matthews_corrcoef(y_true, y_pred_binary_final)
        except ValueError:
            metrics["mcc"] = np.nan
    
    return metrics


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray = None,
    y_pred_binary: np.ndarray = None,
    regions: list | None = None,
) -> dict:
    """
    Compute metrics for all classes and mean.
    Mean is computed as: first compute per-disease, then average across diseases.
    
    Args:
        y_true: Ground truth (n_samples, n_classes)
        y_pred_probs: Probability predictions (n_samples, n_classes) - optional
        y_pred_binary: Binary predictions (n_samples, n_classes) - fallback
        
    Returns:
        Dictionary with per-class and mean metrics
    """
    n_classes = y_true.shape[1]
    results = {"per_class": [], "mean": {}}
    
    # Collect metrics from classes with positive samples
    all_aucs, all_f1s, all_aps, all_mccs = [], [], [], []
    
    for class_idx in range(n_classes):
        # Get predictions for this class
        probs = y_pred_probs[:, class_idx] if y_pred_probs is not None else None
        binary = y_pred_binary[:, class_idx] if y_pred_binary is not None else None
        
        class_metrics = compute_metrics_single_class(
            y_true[:, class_idx],
            y_pred_probs=probs,
            y_pred_binary=binary,
        )
        class_metrics["class_idx"] = class_idx
        results["per_class"].append(class_metrics)
        
        # Only include in mean if class has positive samples
        if class_metrics.get("has_positives", False):
            if class_metrics["auc"] is not None and not np.isnan(class_metrics["auc"]):
                all_aucs.append(class_metrics["auc"])
            if class_metrics["ap"] is not None and not np.isnan(class_metrics["ap"]):
                all_aps.append(class_metrics["ap"])
            if class_metrics["f1"] is not None:
                all_f1s.append(class_metrics["f1"])
            if class_metrics["mcc"] is not None and not np.isnan(class_metrics["mcc"]):
                all_mccs.append(class_metrics["mcc"])
    
    # Compute mean metrics (mean of per-disease metrics)
    results["mean"]["auc"] = np.mean(all_aucs) if all_aucs else np.nan
    results["mean"]["f1"] = np.mean(all_f1s) if all_f1s else np.nan
    results["mean"]["ap"] = np.mean(all_aps) if all_aps else np.nan
    results["mean"]["mcc"] = np.mean(all_mccs) if all_mccs else np.nan
    
    return results


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray = None,
    y_pred_binary: np.ndarray = None,
    regions: list | None = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict:
    """
    Compute bootstrap confidence intervals for metrics.
    
    Args:
        y_true: Ground truth (n_samples, n_classes)
        y_pred_probs: Probability predictions (n_samples, n_classes) - optional
        y_pred_binary: Binary predictions (n_samples, n_classes) - fallback
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level for CI
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with metrics, CIs, and bootstrap samples
    """
    np.random.seed(random_state)
    n_samples = y_true.shape[0]
    n_classes = y_true.shape[1]
    
    # Store bootstrap samples for each metric
    bootstrap_samples = {
        "per_class": {
            class_idx: {"auc": [], "f1": [], "ap": [], "mcc": []}
            for class_idx in range(n_classes)
        },
        "mean": {"auc": [], "f1": [], "ap": [], "mcc": []},
    }
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_probs_boot = y_pred_probs[indices] if y_pred_probs is not None else None
        y_binary_boot = y_pred_binary[indices] if y_pred_binary is not None else None
        
        # Compute metrics for this bootstrap sample
        boot_results = compute_all_metrics(y_true_boot, y_probs_boot, y_binary_boot)
        
        # Store per-class metrics
        for class_idx, class_metrics in enumerate(boot_results["per_class"]):
            for metric in ["auc", "f1", "ap", "mcc"]:
                value = class_metrics.get(metric)
                bootstrap_samples["per_class"][class_idx][metric].append(
                    value if value is not None else np.nan
                )
        
        # Store mean metrics
        for metric in ["auc", "f1", "ap", "mcc"]:
            bootstrap_samples["mean"][metric].append(boot_results["mean"][metric])
    
    # Compute point estimates and CIs
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    # Get point estimates
    point_estimates = compute_all_metrics(y_true, y_pred_probs, y_pred_binary, regions=regions)
    
    results = {"per_class": [], "mean": {}, "bootstrap_samples": bootstrap_samples}
    
    # Process per-class results
    for class_idx in range(n_classes):
        class_result = {
            "class_idx": class_idx,
            "has_positives": point_estimates["per_class"][class_idx].get("has_positives", False),
        }
        for metric in ["auc", "f1", "ap", "mcc"]:
            samples = np.array(bootstrap_samples["per_class"][class_idx][metric], dtype=float)
            valid_samples = samples[~np.isnan(samples)]
            
            # Use mean of bootstrap samples (consistent with t-test calculation)
            if len(valid_samples) > 0:
                class_result[metric] = np.nanmean(valid_samples)
                class_result[f"{metric}_ci_low"] = np.percentile(valid_samples, lower_percentile)
                class_result[f"{metric}_ci_high"] = np.percentile(valid_samples, upper_percentile)
                class_result[f"{metric}_std"] = np.std(valid_samples)
            else:
                # Fallback to point estimate if no valid bootstrap samples
                point_val = point_estimates["per_class"][class_idx].get(metric)
                class_result[metric] = point_val if point_val is not None else np.nan
                class_result[f"{metric}_ci_low"] = np.nan
                class_result[f"{metric}_ci_high"] = np.nan
                class_result[f"{metric}_std"] = np.nan
        
        results["per_class"].append(class_result)
    
    # Process mean results
    for metric in ["auc", "f1", "ap", "mcc"]:
        samples = np.array(bootstrap_samples["mean"][metric], dtype=float)
        valid_samples = samples[~np.isnan(samples)]
        
        # 对于 mean_*，mean 本身使用原始样本的 point estimate，
        # 仅 CI / std 使用 bootstrap 分布
        results["mean"][metric] = point_estimates["mean"][metric]
        if len(valid_samples) > 0:
            results["mean"][f"{metric}_ci_low"] = np.percentile(valid_samples, lower_percentile)
            results["mean"][f"{metric}_ci_high"] = np.percentile(valid_samples, upper_percentile)
            results["mean"][f"{metric}_std"] = np.std(valid_samples)
        else:
            results["mean"][f"{metric}_ci_low"] = np.nan
            results["mean"][f"{metric}_ci_high"] = np.nan
            results["mean"][f"{metric}_std"] = np.nan
    
    return results


def perform_ttest(
    samples_a: list, samples_b: list, alpha: float = 0.05
) -> dict:
    """
    Perform 2-sided t-test between two sets of bootstrap samples.
    
    Args:
        samples_a: Bootstrap samples from model A (reference)
        samples_b: Bootstrap samples from model B (comparison)
        alpha: Significance level
        
    Returns:
        Dictionary with t-statistic, p-value, and significance
    """
    # Remove NaN values
    samples_a = np.array(samples_a)
    samples_b = np.array(samples_b)
    samples_a = samples_a[~np.isnan(samples_a)]
    samples_b = samples_b[~np.isnan(samples_b)]
    
    if len(samples_a) < 2 or len(samples_b) < 2:
        return {
            "t_statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
            "mean_diff": np.nan,
        }
    
    # Perform 2-sided t-test
    t_stat, p_value = stats.ttest_ind(samples_a, samples_b)
    
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
        "mean_diff": np.mean(samples_a) - np.mean(samples_b),
    }


def parse_config_name(config_name: str):
    """
    解析配置目录名，格式示例：bs8_lr0.001_clam
    返回 (batch_size, lr, mil_type)
    """
    try:
        parts = config_name.split("_")
        bs_part = parts[0].replace("bs", "")
        lr_part = parts[1].replace("lr", "")
        mil_type = "_".join(parts[2:]) if len(parts) > 2 else "unknown"
        return int(bs_part), lr_part, mil_type
    except Exception:
        return None, None, "unknown"


def find_experiment_paths(base_dir: str) -> list[tuple]:
    """
    Find all test_results.csv files and organize by dataset/model/config/mil_type.
    兼容旧目录结构：<model>/bs4096_lr0.001/version_X/。
    对于 bs4096_lr0.001 这样的目录，mil_type 记为 "unknown"。
    
    Args:
        base_dir: Base directory containing MIL LP experiments
        
    Returns:
        List of tuples: (dataset, model, config_name, mil_type, csv_path)
    """
    experiments = []
    
    # 目录结构: base_dir / <dataset> / <model_name> / bsXX_lrYY[_mil] / version_X / test_results.csv
    base_path = Path(base_dir)
    
    # 遍历所有 dataset 目录
    for dataset_dir in base_path.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        
        # 遍历所有 model 目录
        for model_dir in dataset_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model = model_dir.name
                
                # 优先读取新目录（带mil_type的，如 bs4096_lr0.001_abmil_full），如果不存在则使用旧目录（bs4096_lr0.001）
                # 获取所有配置目录
                all_config_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
                legacy_dir = model_dir / "bs4096_lr0.001"
                
                # 分离新目录（带mil_type的，如 bs4096_lr0.001_abmil_full）和旧目录
                # 新目录是指以 bs4096_lr0.001_ 开头的目录（带mil_type后缀）
                new_config_dirs = [d for d in all_config_dirs if d.name.startswith("bs4096_lr0.001_")]
                config_dirs = []
                
                if new_config_dirs:
                    # 如果存在新目录（带mil_type的），优先使用新目录
                    config_dirs = new_config_dirs
                elif legacy_dir.exists() and legacy_dir.is_dir():
                    # 如果新目录不存在，使用旧目录
                    config_dirs = [legacy_dir]

                for config_dir in config_dirs:
                    config_name = config_dir.name
                    _, _, mil_type = parse_config_name(config_name)

                    # 找到所有 version_* 目录
                    version_dirs = []
                    for item in config_dir.iterdir():
                        if item.is_dir() and item.name.startswith("version_"):
                            try:
                                version_num = int(item.name.split("_")[1])
                                version_dirs.append((version_num, item))
                            except (ValueError, IndexError):
                                continue

                    if not version_dirs:
                        continue

                    # 选择版本号最大的
                    version_dirs.sort(key=lambda x: x[0], reverse=True)
                    max_version_dir = version_dirs[0][1]

                    # 查找 test_results.csv
                    csv_path = max_version_dir / "test_results.csv"
                    if csv_path.exists() and csv_path.is_file():
                        experiments.append((dataset, model, config_name, mil_type, str(csv_path)))
    
    return experiments


def process_single_experiment(args: tuple) -> dict:
    """
    Worker function to process a single dataset-model combination.
    
    Args:
        args: Tuple of (dataset, model, csv_path, n_bootstrap)
        
    Returns:
        Dictionary with per_disease_results, mean_result, bootstrap_samples, and metadata
    """
    dataset, model, config_name, mil_type, csv_path, n_bootstrap, split_region = args
    
    result = {
        "dataset": dataset,
        "model": model,
        "config": config_name,
        "mil_type": mil_type,
        "csv_path": csv_path,
        "success": False,
        "error": None,
        "per_disease_results": [],
        "mean_results": [],
        "mean_result_main": None,
        "bootstrap_collections": [],  # List of (ds_name, bootstrap_samples)
    }
    
    try:
        # Load data
        y_true, y_pred_probs, y_pred_binary, regions = load_results(csv_path)
        
        # Compute metrics with bootstrap
        bootstrap_result = bootstrap_metrics(
            y_true,
            y_pred_probs=y_pred_probs,
            y_pred_binary=y_pred_binary,
            regions=regions,
            n_bootstrap=n_bootstrap,
        )
        
        # Store bootstrap samples for t-tests
        result["bootstrap_collections"].append((dataset, bootstrap_result["bootstrap_samples"]))
        
        def append_results(ds_name: str, for_ttest: bool = False):
            # Add per-disease results
            for class_result in bootstrap_result["per_class"]:
                row = {
                    "dataset": ds_name,
                    "model": model,
                    "config": config_name,
                    "mil_type": mil_type,
                    "class_idx": class_result["class_idx"],
                    "has_positives": class_result.get("has_positives", False),
                }
                for metric in ["auc", "f1", "ap", "mcc"]:
                    row[metric] = class_result[metric]
                    row[f"{metric}_ci_low"] = class_result.get(f"{metric}_ci_low", np.nan)
                    row[f"{metric}_ci_high"] = class_result.get(f"{metric}_ci_high", np.nan)
                    row[f"{metric}_std"] = class_result.get(f"{metric}_std", np.nan)
                result["per_disease_results"].append(row)

            # Add mean results
            mean_row = {
                "dataset": ds_name,
                "model": model,
                "config": config_name,
                "mil_type": mil_type,
                "has_probs": y_pred_probs is not None,
            }
            for metric in ["auc", "f1", "ap", "mcc"]:
                mean_row[f"mean_{metric}"] = bootstrap_result["mean"][metric]
                mean_row[f"mean_{metric}_ci_low"] = bootstrap_result["mean"].get(f"{metric}_ci_low", np.nan)
                mean_row[f"mean_{metric}_ci_high"] = bootstrap_result["mean"].get(f"{metric}_ci_high", np.nan)
                mean_row[f"mean_{metric}_std"] = bootstrap_result["mean"].get(f"{metric}_std", np.nan)
            result["mean_results"].append(mean_row)
            if for_ttest:
                result["mean_result_main"] = mean_row

        # 原始 dataset
        append_results(dataset, for_ttest=True)

        # 若开启 region 拆分且有 region 列，则为每个 region 复制一份
        if split_region and regions is not None:
            unique_regions = sorted(set(regions))
            region_to_indices = {r: [] for r in unique_regions}
            for idx, r in enumerate(regions):
                region_to_indices[r].append(idx)

            for r, idxs in region_to_indices.items():
                if len(idxs) == 0:
                    continue
                y_true_r = y_true[idxs]
                y_probs_r = y_pred_probs[idxs] if y_pred_probs is not None else None
                y_bin_r = y_pred_binary[idxs] if y_pred_binary is not None else None
                boot_r = bootstrap_metrics(
                    y_true_r,
                    y_pred_probs=y_probs_r,
                    y_pred_binary=y_bin_r,
                    regions=None,
                    n_bootstrap=n_bootstrap,
                )
                # 暂存一次，然后用 append_results 逻辑
                backup = bootstrap_result
                bootstrap_result = boot_r
                append_results(f"{dataset}_{r}", for_ttest=False)
                result["bootstrap_collections"].append((f"{dataset}_{r}", boot_r["bootstrap_samples"]))
                bootstrap_result = backup
        
        result["success"] = True
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def analyze_all_experiments(
    base_dir: str,
    output_dir: str,
    n_bootstrap: int = 1000,
    reference_model: str = REFERENCE_MODEL,
    mil_type: str | None = None,
    split_region: bool = False,
    n_workers: int = None,
):
    """
    Run full analysis on all experiments using multiprocessing.
    
    Args:
        base_dir: Directory containing LP experiments
        output_dir: Directory to save output CSVs
        n_bootstrap: Number of bootstrap iterations
        reference_model: Model to compare against in t-tests
        n_workers: Number of parallel workers (default: CPU count)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default number of workers
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    # Find all experiments
    print("Finding experiment files...")
    experiments = find_experiment_paths(base_dir)
    
    print(f"Found {len(experiments)} experiment configs")
    for ds, model, config_name, mil, _ in experiments:
        print(f"  {ds} / {model} / {config_name} (mil={mil})")
    
    # 过滤指定的 mil_type（若提供）
    # 如果指定了mil_type，优先保留匹配的新目录，同时保留所有旧目录（作为回退）
    if mil_type is not None:
        experiments = [e for e in experiments if e[3] == mil_type or e[3] == "unknown"]

    # Create list of all jobs (dataset, model, config, mil_type, csv_path, n_bootstrap, split_region)
    jobs = [(ds, model, cfg, mil, path, n_bootstrap, split_region) for ds, model, cfg, mil, path in experiments]
    
    print(f"\nProcessing {len(jobs)} experiments with {n_workers} workers...")
    
    # Storage for all results
    all_per_disease_results = []
    all_mean_results = []
    all_ttest_results = []
    
    # Store bootstrap samples and mil types for t-tests
    bootstrap_cache = {}
    mil_map = {}
    
    # Process experiments in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all jobs
        future_to_job = {executor.submit(process_single_experiment, job): job for job in jobs}
        
        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_job), total=len(jobs), desc="Processing experiments"):
            result = future.result()
            dataset = result["dataset"]
            model = result["model"]
            config_name = result["config"]
            mil_type = result["mil_type"]
            
            if result["success"]:
                # Store per-disease results
                all_per_disease_results.extend(result["per_disease_results"])
                
                # Store mean results (including region splits if any)
                all_mean_results.extend(result["mean_results"])
                
                # Store bootstrap samples for t-tests（包含 region 拆分）
                for ds_name, boot_samples in result["bootstrap_collections"]:
                    if ds_name not in bootstrap_cache:
                        bootstrap_cache[ds_name] = {}
                        mil_map[ds_name] = {}
                    bootstrap_cache[ds_name][model] = boot_samples
                    mil_map[ds_name][model] = mil_type
            else:
                print(f"Error processing {result['csv_path']}: {result['error']}")
    
    # Perform t-tests (sequential, depends on all bootstrap samples being collected)
    print("\nPerforming t-tests...")
    for dataset in tqdm(sorted(bootstrap_cache.keys()), desc="T-tests by dataset"):
        if reference_model not in bootstrap_cache[dataset]:
            continue
        
        ref_samples = bootstrap_cache[dataset][reference_model]
        ref_mil = mil_map.get(dataset, {}).get(reference_model, "unknown")
        
        for model_key in bootstrap_cache[dataset].keys():
            if model_key == reference_model:
                continue
            
            model_name = model_key
            model_mil = mil_map.get(dataset, {}).get(model_name, "unknown")
            model_samples = bootstrap_cache[dataset][model_key]
        
        # T-test for mean metrics
            for metric in ["auc", "f1", "ap", "mcc"]:
                ttest_result = perform_ttest(
                    ref_samples["mean"][metric],
                    model_samples["mean"][metric],
                )
                
                all_ttest_results.append({
                    "dataset": dataset,
                    "metric": f"mean_{metric}",
                    "class_idx": None,
                    "model_a": reference_model,
                    "model_b": model_name,
                    "mil": ref_mil,
                    "mean_a": np.nanmean(ref_samples["mean"][metric]),
                    "mean_b": np.nanmean(model_samples["mean"][metric]),
                    "mean_diff": ttest_result["mean_diff"],
                    "t_statistic": ttest_result["t_statistic"],
                    "p_value": ttest_result["p_value"],
                    "significant": ttest_result["significant"],
                })
            
            # T-test for per-class metrics
            n_classes = len(ref_samples["per_class"])
            for class_idx in range(n_classes):
                for metric in ["auc", "f1", "ap", "mcc"]:
                    ttest_result = perform_ttest(
                        ref_samples["per_class"][class_idx][metric],
                        model_samples["per_class"][class_idx][metric],
                    )
                    
                    all_ttest_results.append({
                        "dataset": dataset,
                        "metric": metric,
                        "class_idx": class_idx,
                        "model_a": reference_model,
                        "model_b": model_name,
                        "mil": ref_mil,
                        "mean_a": np.nanmean(ref_samples["per_class"][class_idx][metric]),
                        "mean_b": np.nanmean(model_samples["per_class"][class_idx][metric]),
                        "mean_diff": ttest_result["mean_diff"],
                        "t_statistic": ttest_result["t_statistic"],
                        "p_value": ttest_result["p_value"],
                        "significant": ttest_result["significant"],
                    })
    
    # Save results to CSV
    print("\nSaving results...")
    
    # Per-disease results
    df_per_disease = pd.DataFrame(all_per_disease_results)
    per_disease_path = os.path.join(output_dir, "results_per_disease.csv")
    df_per_disease.to_csv(per_disease_path, index=False)
    print(f"  Saved per-disease results to: {per_disease_path}")
    
    # Mean results
    df_mean = pd.DataFrame(all_mean_results)
    mean_path = os.path.join(output_dir, "results_mean.csv")
    df_mean.to_csv(mean_path, index=False)
    print(f"  Saved mean results to: {mean_path}")
    
    # T-test results
    df_ttest = pd.DataFrame(all_ttest_results)
    ttest_path = os.path.join(output_dir, "ttest_results.csv")
    df_ttest.to_csv(ttest_path, index=False)
    print(f"  Saved t-test results to: {ttest_path}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total datasets analyzed: {len(experiments)}")
    print(f"Total dataset-model combinations: {len(all_mean_results)}")
    print(f"Total per-disease results: {len(all_per_disease_results)}")
    print(f"Total t-test comparisons: {len(all_ttest_results)}")
    
    # Print mean results summary table
    if len(df_mean) > 0:
        print("\nMean AUC by Dataset and Model:")
        pivot = df_mean.pivot_table(
            index="dataset",
            columns="model",
            values="mean_auc",
            aggfunc="first",
        )
        print(pivot.to_string())
    
    return df_per_disease, df_mean, df_ttest


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MIL Linear Probe test results with bootstrapping and t-tests"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="experiments/MIL_LP",
        help="Base directory containing MIL LP experiments",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_results",
        help="Directory to save output CSVs",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap iterations",
    )
    parser.add_argument(
        "--reference_model",
        type=str,
        default=REFERENCE_MODEL,
        help="Reference model for t-test comparisons (model name only)",
    )
    parser.add_argument(
        "--mil_type",
        type=str,
        default=None,
        help="If set, only analyze experiments with this MIL type (e.g., clam, abmil_full)",
    )
    parser.add_argument(
        "--split_region",
        action="store_true",
        help="If set, also create region-level datasets by appending region to dataset name",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPU cores)",
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyze_all_experiments(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
        reference_model=args.reference_model,
        mil_type=args.mil_type,
        split_region=args.split_region,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()


