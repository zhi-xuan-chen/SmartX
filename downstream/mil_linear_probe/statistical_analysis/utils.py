import argparse
import ast
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-region disease exclusion
#
# Drop specific diseases from a region's analysis (macro mean + per-disease +
# significance + FDR family all stay consistent). Keyed by region NAME; the
# class_idx order in each test_results.csv follows ``REGION_DISEASES[region]``
# from ../data_utils.py, so we exclude by disease name (robust, clinically
# meaningful) rather than by fragile integer indices.
#
# Default empty == no change. Add entries with a clinical/objective reason
# (anatomically irrelevant, too few positives), decided a priori — NOT by
# looking at the scores (that would be a winner's-curse selection).
# Example:
#   REGION_EXCLUDE_DISEASES = {
#       "Chest": ["气胸"],     # reason: ...
#       "Foot":  ["骨化"],
#   }
# ---------------------------------------------------------------------------
REGION_EXCLUDE_DISEASES: dict[str, list[str]] = {}


@lru_cache(maxsize=1)
def _region_diseases() -> dict[str, list[str]]:
    """Read REGION_DISEASES from ../data_utils.py via AST (no torch import)."""
    path = Path(__file__).resolve().parents[1] / "data_utils.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except OSError:
        return {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "REGION_DISEASES" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    return {}


def region_from_dataset(dataset: str, region: str | None = None) -> str | None:
    """Resolve a region name from a dataset string.

    External callers can pass ``region`` directly. Otherwise match the trailing
    region against known REGION_DISEASES keys (handles multi-word regions like
    ``Cervical_spine`` and any cohort prefix, e.g. ``private_a_`` / ``ncfy_pro_``).
    """
    if region:
        return region
    for r in _region_diseases():
        if dataset == r or dataset.endswith("_" + r):
            return r
    return None


def included_classes(dataset: str, n_classes: int, region: str | None = None) -> np.ndarray:
    """Boolean mask (n_classes,) of class_idx to KEEP for ``dataset``.

    Excludes the diseases listed in ``REGION_EXCLUDE_DISEASES`` for the resolved
    region, mapping names -> class_idx via ``REGION_DISEASES[region]`` order.
    Unknown region or empty exclusion -> keep all.
    """
    keep = np.ones(n_classes, dtype=bool)
    r = region_from_dataset(dataset, region)
    if r is None:
        return keep
    excl = set(REGION_EXCLUDE_DISEASES.get(r, []))
    if not excl:
        return keep
    names = _region_diseases().get(r, [])
    for c in range(min(n_classes, len(names))):
        if names[c] in excl:
            keep[c] = False
    return keep


def parse_model_list(value: str | None) -> set[str] | None:
    """Parse comma-separated model names; None/empty means no filter."""
    if value is None or not str(value).strip():
        return None
    return {m.strip() for m in value.split(",") if m.strip()}


def model_allowed(
    model: str,
    allowed: set[str] | None = None,
    excluded: set[str] | None = None,
) -> bool:
    if allowed is not None and model not in allowed:
        return False
    if excluded is not None and model in excluded:
        return False
    return True


def add_model_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="逗号分隔的模型白名单，仅分析/比较列表中的模型",
    )
    parser.add_argument(
        "--exclude_models",
        type=str,
        default=None,
        help="逗号分隔的模型黑名单，跳过这些模型",
    )


# Dataset class counts (MIL context, might differ or be same as LP)
DATASET_NUM_CLASSES = {
    "private_a": 46, # Derived from maximum class_idx (45) in test_results.csv
}


def load_results(csv_path: str) -> tuple:
    """
    Load test results from CSV file.
    
    Args:
        csv_path: Path to test_results.csv
        
    Returns:
        Tuple of (ground_truth, prediction_probs, binary_predictions, regions) as numpy arrays
        Shape: (n_samples, n_classes) for first three
    """
    df = pd.read_csv(csv_path)
    
    # Parse string representations of lists
    ground_truth = np.array([ast.literal_eval(gt) for gt in df["ground_truth"]])
    binary_predictions = np.array([ast.literal_eval(pred) for pred in df["prediction"]])
    
    # Try to load probability predictions if available
    prediction_probs = None
    if "prediction_probs" in df.columns:
        prediction_probs = np.array([ast.literal_eval(pred) for pred in df["prediction_probs"]])
    
    # Optional: region 信息（若存在，用于分组分析）
    regions = df["region"].tolist() if "region" in df.columns else None
    
    return ground_truth, prediction_probs, binary_predictions, regions


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
    base_path = Path(base_dir)
    
    if not base_path.exists():
        return experiments

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
                
                # 获取所有配置目录
                all_config_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
                
                # MIL 常见的配置前缀
                config_prefix = "bs4096_lr0.001"
                
                # 分离新目录（带mil_type的，如 bs4096_lr0.001_abmil_full）和旧目录
                new_config_dirs = [d for d in all_config_dirs if d.name.startswith(config_prefix + "_")]
                legacy_dir = model_dir / config_prefix
                
                config_dirs = []
                if new_config_dirs:
                    config_dirs = new_config_dirs
                elif legacy_dir.exists() and legacy_dir.is_dir():
                    config_dirs = [legacy_dir]
                else:
                    # 如果都不符合，尝试所有看起来像配置的目录
                    config_dirs = [d for d in all_config_dirs if d.name.startswith("bs")]

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
