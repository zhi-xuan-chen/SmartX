"""Paired significance tests for report generation metrics.

For each (reference_model, candidate_model) pair on the same test set, this
script computes paired significance tests using either paired permutation or
Wilcoxon signed-rank over paired bootstrap metric differences.

Metrics covered out-of-the-box (no model inference required):
  - BLEU-4 (corpus-level, sacrebleu)
  - BLEU-1 (corpus-level, sacrebleu)
  - ROUGE-L (sentence-level averaged, google-research/rouge_score)
  - ROUGE-2 (sentence-level averaged, google-research/rouge_score)

Metrics that require external inference (CheXBERT, RadGraph): pass
precomputed per-report labels/rewards via ``--chexbert_labels``
``--radgraph_rewards`` (json mapping ``model_name -> {report_id: ...}``).

Output:
  ``pairwise_significance.csv`` with one row per (metric, model_b) and
  Benjamini-Hochberg FDR correction grouped by ``metric``.

  When ``--dataset`` is set, rows are appended to ``pairwise_significance_all.csv``
  (or ``--combined_csv``) with a leading ``dataset`` column.

Usage example:
  python analyze_significance.py \
      --jsonls topk_v109:preds_topk_v109.jsonl,chess:preds_chess.jsonl,\
biomedclip:preds_biomedclip.jsonl \
      --reference_model topk_v109 \
      --output_dir analysis_results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Repo-root utils import
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from utils.stat_test import (  # noqa: E402
    paired_bootstrap_wilcoxon_test,
    paired_bootstrap_pvalue_test,
    paired_bootstrap_diff_ci,
    paired_permutation_test,
    percentile_ci,
    add_fdr_by_group,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_N_PERMUTATION,
    DEFAULT_SEED,
)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_jsonl(path: str) -> dict[str, dict]:
    """Load jsonl into ``{id: row_dict}``."""
    out: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = d.get("id")
            if key is None:
                key = d.get("question_id", d.get("index", len(out)))
            out[str(key)] = d
    return out


def align_models(model_files: dict[str, str]) -> tuple[list[str], dict[str, dict[str, dict]]]:
    """Load all model jsonls and intersect on common ids.

    Returns ``(common_ids, {model: {id: row}})``.
    """
    per_model: dict[str, dict[str, dict]] = {m: load_jsonl(p) for m, p in model_files.items()}
    common = set.intersection(*[set(per_model[m].keys()) for m in per_model])
    common_ids = sorted(common, key=lambda x: (len(x), x))
    print(f"Loaded {len(model_files)} models; common ids = {len(common_ids)}")
    for m, rows in per_model.items():
        print(f"  {m}: {len(rows)} rows ({len(rows) - len(common)} dropped after intersection)")
    return common_ids, per_model


def get_predictions(per_model: dict, common_ids: list[str], model: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (predictions, references) as object arrays aligned to common_ids."""
    preds, refs = [], []
    for qid in common_ids:
        row = per_model[model][qid]
        preds.append(row["prediction"])
        refs.append(row["reference"])
    return np.array(preds, dtype=object), np.array(refs, dtype=object)


# -----------------------------------------------------------------------------
# Metric builders
# -----------------------------------------------------------------------------

def make_corpus_bleu_metric(references: np.ndarray, max_ngram_order: int = 4):
    """metric_fn(predictions_array) -> corpus BLEU score on full set (sacrebleu)."""
    from sacrebleu.metrics import BLEU
    bleu = BLEU(max_ngram_order=max_ngram_order)
    refs_list = [list(references)]  # sacrebleu expects [[ref1, ref2, ...]]

    def _fn(predictions: np.ndarray) -> float:
        return float(bleu.corpus_score(list(predictions), refs_list).score)

    return _fn


def make_paired_corpus_bleu_metric(max_ngram_order: int = 4):
    """metric_fn(paired_pred_ref_array) -> corpus BLEU on resampled pairs."""
    from sacrebleu.metrics import BLEU
    bleu = BLEU(max_ngram_order=max_ngram_order)

    def _fn(paired: np.ndarray) -> float:
        arr = np.asarray(paired, dtype=object)
        return float(bleu.corpus_score(list(arr[:, 0]), [list(arr[:, 1])]).score)

    return _fn


def make_rouge_mean_metric(per_report_scores: np.ndarray):
    """metric_fn(per_report_scores) -> mean ROUGE score.

    Because ROUGE is sentence-level averaged, we precompute per-report scores
    once and the metric is simply their mean. The ``data_a``/``data_b`` passed to
    the significance tests are these precomputed per-report score arrays
    themselves (one value per report), so permuting / resampling along axis 0
    swaps or resamples reports and recomputes the mean — exactly the paired unit
    we want. (The metric operates on the score *values*, not on indices.)
    """
    arr = np.asarray(per_report_scores, dtype=float)

    def _fn(x: np.ndarray) -> float:
        return float(np.mean(x))

    return _fn, arr


def precompute_rouge_per_report(predictions: np.ndarray, references: np.ndarray, rouge_type: str = "rougeL") -> np.ndarray:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=False)
    out = np.empty(len(predictions), dtype=float)
    for i, (p, r) in enumerate(zip(predictions, references)):
        out[i] = scorer.score(r, p)[rouge_type].fmeasure
    return out


# CheXBERT label constants (mirror rrg_eval.factuality_utils to avoid the
# heavy transitive import: avoids needing torch/transformers to read JSONs).
CHEXBERT_CONDITIONS_14 = (
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices", "No Finding",
)
CHEXBERT_CONDITIONS_5 = ("Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion")
CHEXBERT_C5_INDICES = tuple(CHEXBERT_CONDITIONS_14.index(c) for c in CHEXBERT_CONDITIONS_5)


def _chexbert_to_binary(labels_raw: np.ndarray, uncertain_as_positive: bool) -> np.ndarray:
    """Convert raw 0-3 CheXBERT labels to binary.

    Mapping (matches ``rrg_eval.factuality_utils.map_to_binary``):
        Default ("rrg"):  POSITIVE=1, others=0  (uncertain treated as negative)
        Plus variant ("rrg+"): POSITIVE=1 OR UNCERTAIN=1, others=0
    """
    if uncertain_as_positive:
        # rrg+ mode: pos OR uncertain -> 1
        return ((labels_raw == 1) | (labels_raw == 3)).astype(np.int8)
    # rrg mode: pos only -> 1
    return (labels_raw == 1).astype(np.int8)


# Parse "Micro-F1-14+" / "Macro-F1-5" / etc.
def _parse_chexbert_metric(name: str) -> tuple[str, int, bool]:
    """Return (average, n_conditions, uncertain_as_positive)."""
    uncertain_as_positive = name.endswith("+")
    base = name.rstrip("+")
    if base == "Micro-F1-14":
        return "micro", 14, uncertain_as_positive
    if base == "Macro-F1-14":
        return "macro", 14, uncertain_as_positive
    if base == "Micro-F1-5":
        return "micro", 5, uncertain_as_positive
    if base == "Macro-F1-5":
        return "macro", 5, uncertain_as_positive
    raise ValueError(f"unknown CheXBERT-F1 variant: {name!r}")


CHEXBERT_VARIANTS = (
    "Micro-F1-14", "Macro-F1-14", "Micro-F1-5", "Macro-F1-5",
    "Micro-F1-14+", "Macro-F1-14+", "Micro-F1-5+", "Macro-F1-5+",
)


def _load_chexbert_per_model(
    chexbert_labels: dict, common_ids: list[str], model: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_labels, ref_labels) aligned to common_ids, shape (N, 14), raw 0-3."""
    rows_pred, rows_ref = [], []
    for qid in common_ids:
        entry = chexbert_labels[model][qid]
        rows_pred.append(entry["pred"])
        rows_ref.append(entry["ref"])
    return np.asarray(rows_pred, dtype=np.int8), np.asarray(rows_ref, dtype=np.int8)


def make_chexbert_f1_metric(
    ref_labels_raw: np.ndarray, variant: str,
):
    """Build a metric_fn that takes per-report pred raw 0-3 labels and
    returns F1 (micro or macro) against fixed ``ref_labels_raw``.

    ``ref_labels_raw`` is the **ground-truth** CheXBERT extraction of the
    *reference* reports — it does **not** swap under permutation. Only the
    *predicted* CheXBERT labels swap between the two models.
    """
    from sklearn.metrics import f1_score

    average, n_cond, uncertain_as_positive = _parse_chexbert_metric(variant)
    if n_cond == 5:
        col_idx = list(CHEXBERT_C5_INDICES)
    else:
        col_idx = list(range(14))

    ref_bin = _chexbert_to_binary(ref_labels_raw[:, col_idx], uncertain_as_positive)

    def _fn(pred_raw: np.ndarray) -> float:
        pred_bin = _chexbert_to_binary(pred_raw[:, col_idx], uncertain_as_positive)
        return float(f1_score(ref_bin, pred_bin, average=average, zero_division=0))

    return _fn


def make_paired_chexbert_f1_metric(variant: str):
    """Build F1 metric over paired ``[pred_raw, ref_raw]`` report rows."""
    from sklearn.metrics import f1_score

    average, n_cond, uncertain_as_positive = _parse_chexbert_metric(variant)
    col_idx = list(CHEXBERT_C5_INDICES) if n_cond == 5 else list(range(14))

    def _fn(pair_raw: np.ndarray) -> float:
        arr = np.asarray(pair_raw)
        pred_bin = _chexbert_to_binary(arr[:, 0, :][:, col_idx], uncertain_as_positive)
        ref_bin = _chexbert_to_binary(arr[:, 1, :][:, col_idx], uncertain_as_positive)
        return float(f1_score(ref_bin, pred_bin, average=average, zero_division=0))

    return _fn


# -----------------------------------------------------------------------------
# Per-metric comparison loop
# -----------------------------------------------------------------------------

SIGNIFICANCE_METHODS = ("paired_permutation", "bootstrap_wilcoxon", "bootstrap_pvalue")


def _run_significance_test(
    metric_fn,
    data_a: np.ndarray,
    data_b: np.ndarray,
    *,
    significance_method: str,
    n_permutation: int,
    n_bootstrap: int,
    random_state: int,
) -> tuple[dict, dict]:
    """Run the selected paired significance test and return method columns.

    Regardless of the p-value method, the effect-size 95% bootstrap CI for the
    paired score difference is always reported (``score_diff_ci_low/high``) so
    the output table carries an effect size next to the p-value.
    """
    if significance_method == "paired_permutation":
        result = paired_permutation_test(
            metric_fn, data_a, data_b,
            n_resamples=n_permutation, random_state=random_state,
        )
        # permutation gives only a p-value; attach a bootstrap diff CI for effect size.
        diff_ci = paired_bootstrap_diff_ci(
            metric_fn, data_a, data_b,
            n_resamples=n_bootstrap, random_state=random_state,
        )
        return result, {
            "score_diff_ci_low": diff_ci["diff_ci_low"],
            "score_diff_ci_high": diff_ci["diff_ci_high"],
            "paired_permutation_p_value": result["p_value"],
        }

    if significance_method == "bootstrap_wilcoxon":
        result = paired_bootstrap_wilcoxon_test(
            metric_fn, data_a, data_b,
            n_resamples=n_bootstrap, random_state=random_state,
        )
        return result, {
            "score_diff_ci_low": result["diff_ci_low"],
            "score_diff_ci_high": result["diff_ci_high"],
            "bootstrap_wilcoxon_statistic": result["statistic"],
            "bootstrap_wilcoxon_p_value": result["p_value"],
            "bootstrap_wilcoxon_n": result["n_bootstrap_effective"],
            "bootstrap_wilcoxon_zero_frac": result["zero_frac"],
            "bootstrap_wilcoxon_tie_frac": result["tie_frac"],
        }

    if significance_method == "bootstrap_pvalue":
        result = paired_bootstrap_pvalue_test(
            metric_fn, data_a, data_b,
            n_resamples=n_bootstrap, random_state=random_state,
        )
        return result, {
            "score_diff_ci_low": result["diff_ci_low"],
            "score_diff_ci_high": result["diff_ci_high"],
            "bootstrap_pvalue_p_value": result["p_value"],
            "bootstrap_pvalue_n": result["n_bootstrap_effective"],
        }

    raise ValueError(f"unknown significance_method: {significance_method!r}")


def compare_models(
    common_ids: list[str],
    per_model: dict[str, dict[str, dict]],
    reference_model: str,
    metrics: list[str],
    n_permutation: int = DEFAULT_N_PERMUTATION,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    significance_method: str = "bootstrap_wilcoxon",
    random_state: int = DEFAULT_SEED,
    chexbert_labels: dict[str, dict[str, list[int]]] | None = None,
    radgraph_rewards: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Run paired significance tests for each metric x candidate model."""
    ref_preds, refs = get_predictions(per_model, common_ids, reference_model)

    rows: list[dict] = []
    for candidate in [m for m in per_model.keys() if m != reference_model]:
        cand_preds, _ = get_predictions(per_model, common_ids, candidate)

        for metric_name in metrics:
            print(f"  [{metric_name}] {reference_model} vs {candidate}")
            try:
                if metric_name in ("BLEU-4", "BLEU-1"):
                    order = 4 if metric_name == "BLEU-4" else 1
                    metric_fn = make_paired_corpus_bleu_metric(max_ngram_order=order)
                    data_a = np.column_stack([ref_preds, refs])
                    data_b = np.column_stack([cand_preds, refs])
                    pt, method_cols = _run_significance_test(
                        metric_fn, data_a, data_b,
                        significance_method=significance_method,
                        n_permutation=n_permutation,
                        n_bootstrap=n_bootstrap,
                        random_state=random_state,
                    )
                    score_a = metric_fn(data_a)
                    score_b = metric_fn(data_b)
                elif metric_name in ("ROUGE-L", "ROUGE-2"):
                    rtype = "rougeL" if metric_name == "ROUGE-L" else "rouge2"
                    arr_a = precompute_rouge_per_report(ref_preds, refs, rouge_type=rtype)
                    arr_b = precompute_rouge_per_report(cand_preds, refs, rouge_type=rtype)
                    metric_fn, _ = make_rouge_mean_metric(arr_a)
                    pt, method_cols = _run_significance_test(
                        metric_fn, arr_a, arr_b,
                        significance_method=significance_method,
                        n_permutation=n_permutation,
                        n_bootstrap=n_bootstrap,
                        random_state=random_state,
                    )
                    score_a = float(np.mean(arr_a))
                    score_b = float(np.mean(arr_b))
                elif metric_name in CHEXBERT_VARIANTS:
                    if chexbert_labels is None:
                        print(f"    [skip] CheXbert labels not provided")
                        continue
                    pred_a_raw, ref_a_raw = _load_chexbert_per_model(
                        chexbert_labels, common_ids, reference_model
                    )
                    pred_b_raw, ref_b_raw = _load_chexbert_per_model(
                        chexbert_labels, common_ids, candidate
                    )
                    # CheXBERT extraction of the same GT reference text should be
                    # identical across models; warn loudly if not.
                    if not np.array_equal(ref_a_raw, ref_b_raw):
                        print(f"    [warn] GT CheXBERT labels differ between "
                              f"{reference_model} and {candidate}; using {reference_model}'s.")
                    metric_fn = make_paired_chexbert_f1_metric(variant=metric_name)
                    data_a = np.stack([pred_a_raw, ref_a_raw], axis=1)
                    data_b = np.stack([pred_b_raw, ref_a_raw], axis=1)
                    pt, method_cols = _run_significance_test(
                        metric_fn, data_a, data_b,
                        significance_method=significance_method,
                        n_permutation=n_permutation,
                        n_bootstrap=n_bootstrap,
                        random_state=random_state,
                    )
                    score_a = metric_fn(data_a)
                    score_b = metric_fn(data_b)
                elif metric_name == "F1-RadGraph":
                    if radgraph_rewards is None:
                        print(f"    [skip] RadGraph rewards not provided")
                        continue
                    arr_a = np.array(
                        [radgraph_rewards[reference_model][qid] for qid in common_ids], dtype=float
                    )
                    arr_b = np.array(
                        [radgraph_rewards[candidate][qid] for qid in common_ids], dtype=float
                    )
                    metric_fn, _ = make_rouge_mean_metric(arr_a)  # mean over rewards
                    pt, method_cols = _run_significance_test(
                        metric_fn, arr_a, arr_b,
                        significance_method=significance_method,
                        n_permutation=n_permutation,
                        n_bootstrap=n_bootstrap,
                        random_state=random_state,
                    )
                    score_a = float(np.mean(arr_a))
                    score_b = float(np.mean(arr_b))
                else:
                    print(f"    [skip] unknown metric {metric_name!r}")
                    continue
            except Exception as e:
                print(f"    [error] {metric_name}: {e}")
                continue

            row = {
                "metric": metric_name,
                "model_a": reference_model,
                "model_b": candidate,
                "n_paired": len(common_ids),
                "score_a": score_a,
                "score_b": score_b,
                "score_diff": pt["observed_diff"],
                "p_value": pt["p_value"],
                "test_method": significance_method,
                "n_permutation": n_permutation,
                "n_bootstrap": n_bootstrap,
            }
            row.update(method_cols)
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        # BH within each metric — different metrics have different scales, do not mix.
        df = add_fdr_by_group(df, group_cols=["metric"])
    return df


# -----------------------------------------------------------------------------
# Per-model point estimate + 95% bootstrap CI
# -----------------------------------------------------------------------------

class _SkipMetric(Exception):
    """Raised when a metric cannot be built for a model (e.g. labels missing)."""


def _build_metric_statistic(
    metric_name: str,
    preds: np.ndarray,
    refs: np.ndarray,
    common_ids: list[str],
    model: str,
    chexbert_labels: dict | None,
    radgraph_rewards: dict | None,
):
    """Return ``(statistic, n)`` where ``statistic(indices) -> float`` evaluates
    ``metric_name`` on the (resampled) subset of reports.

    Uses exactly the same metric code as ``compare_models`` so the point
    estimate equals ``score_a`` / ``score_b`` in the pairwise table.
    """
    if metric_name in ("BLEU-4", "BLEU-1"):
        from sacrebleu.metrics import BLEU
        order = 4 if metric_name == "BLEU-4" else 1
        bleu = BLEU(max_ngram_order=order)
        preds_list, refs_list = list(preds), list(refs)

        def stat(idx):
            h = [preds_list[int(i)] for i in idx]
            r = [refs_list[int(i)] for i in idx]
            return float(bleu.corpus_score(h, [r]).score)

        return stat, len(preds_list)

    if metric_name in ("ROUGE-L", "ROUGE-2"):
        rtype = "rougeL" if metric_name == "ROUGE-L" else "rouge2"
        arr = precompute_rouge_per_report(preds, refs, rouge_type=rtype)
        return (lambda idx: float(np.mean(arr[np.asarray(idx, dtype=int)]))), len(arr)

    if metric_name in CHEXBERT_VARIANTS:
        if chexbert_labels is None:
            raise _SkipMetric(metric_name)
        from sklearn.metrics import f1_score
        pred_raw, ref_raw = _load_chexbert_per_model(chexbert_labels, common_ids, model)
        average, n_cond, uncertain_as_positive = _parse_chexbert_metric(metric_name)
        col_idx = list(CHEXBERT_C5_INDICES) if n_cond == 5 else list(range(14))
        ref_bin = _chexbert_to_binary(ref_raw[:, col_idx], uncertain_as_positive)
        pred_bin = _chexbert_to_binary(pred_raw[:, col_idx], uncertain_as_positive)

        def stat(idx):
            ii = np.asarray(idx, dtype=int)
            return float(f1_score(ref_bin[ii], pred_bin[ii], average=average, zero_division=0))

        return stat, len(ref_bin)

    if metric_name == "F1-RadGraph":
        if radgraph_rewards is None:
            raise _SkipMetric(metric_name)
        arr = np.array([radgraph_rewards[model][qid] for qid in common_ids], dtype=float)
        return (lambda idx: float(np.mean(arr[np.asarray(idx, dtype=int)]))), len(arr)

    raise _SkipMetric(metric_name)


def compute_metrics_ci(
    common_ids: list[str],
    per_model: dict[str, dict[str, dict]],
    metrics: list[str],
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    random_state: int = DEFAULT_SEED,
    chexbert_labels: dict | None = None,
    radgraph_rewards: dict | None = None,
) -> pd.DataFrame:
    """Per-model point estimate + percentile bootstrap CI for every metric.

    Computed on the same ``common_ids`` and with the same metric code as the
    permutation tests, so the ``point`` column equals ``score_a`` / ``score_b``
    in the pairwise table by construction. Never reads any ``main.csv``.
    """
    rows: list[dict] = []
    for model in per_model.keys():
        preds, refs = get_predictions(per_model, common_ids, model)
        for metric_name in metrics:
            try:
                stat, n = _build_metric_statistic(
                    metric_name, preds, refs, common_ids, model,
                    chexbert_labels, radgraph_rewards,
                )
            except _SkipMetric:
                continue
            try:
                ci = percentile_ci(
                    stat, n, n_resamples=n_bootstrap, random_state=random_state,
                )
            except Exception as e:  # noqa: BLE001
                print(f"    [error] CI {metric_name} / {model}: {e}")
                continue
            rows.append({
                "model": model,
                "metric": metric_name,
                "n": n,
                "point": ci["point"],
                "ci_l": ci["ci_l"],
                "ci_h": ci["ci_h"],
                "n_bootstrap": n_bootstrap,
            })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_jsonls_arg(arg: str) -> dict[str, str]:
    """Parse '--jsonls model1:path1,model2:path2' into a dict."""
    out: dict[str, str] = {}
    for piece in arg.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(f"--jsonls entry must be 'name:path', got {piece!r}")
        name, path = piece.split(":", 1)
        out[name.strip()] = path.strip()
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Paired significance tests for report generation"
    )
    parser.add_argument("--jsonls", type=str, required=True,
                        help="Comma-separated list of 'name:path' for each model jsonl.")
    parser.add_argument("--reference_model", type=str, required=True,
                        help="Name (key in --jsonls) of the reference model.")
    parser.add_argument("--metrics", type=str,
                        default="BLEU-4,BLEU-1,ROUGE-L,ROUGE-2",
                        help="Comma-separated list of metrics to test.")
    parser.add_argument("--output_dir", type=str, default="analysis_results")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Optional dataset tag; adds a `dataset` column and "
                             "writes/appends to the combined CSV.")
    parser.add_argument("--combined_csv", type=str, default=None,
                        help="Combined output path (default: "
                             "<output_dir>/pairwise_significance_all.csv when --dataset is set).")
    parser.add_argument("--n_permutation", type=int, default=DEFAULT_N_PERMUTATION)
    parser.add_argument("--n_bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP,
                        help="Bootstrap resamples for the per-model point+CI table.")
    parser.add_argument(
        "--significance_method",
        choices=SIGNIFICANCE_METHODS,
        default="bootstrap_wilcoxon",
        help="Primary p_value/test_method source: paired_permutation, bootstrap_wilcoxon, or bootstrap_pvalue.",
    )
    parser.add_argument("--metrics_ci_csv", type=str, default=None,
                        help="If set, also write per-model point estimate + 95%% "
                             "percentile CI here (recomputed from jsonl + cached "
                             "labels; never reads main.csv). Appended per dataset.")
    parser.add_argument("--random_state", type=int, default=DEFAULT_SEED)
    parser.add_argument("--chexbert_labels", type=str, default=None,
                        help="Optional json: {model: {report_id: [14-dim binary labels]}}.")
    parser.add_argument("--radgraph_rewards", type=str, default=None,
                        help="Optional json: {model: {report_id: float reward}}.")
    args = parser.parse_args()

    model_files = parse_jsonls_arg(args.jsonls)
    if args.reference_model not in model_files:
        raise ValueError(
            f"--reference_model {args.reference_model!r} not in --jsonls keys {list(model_files)}"
        )
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    chexbert = json.load(open(args.chexbert_labels)) if args.chexbert_labels else None
    radgraph = json.load(open(args.radgraph_rewards)) if args.radgraph_rewards else None

    common_ids, per_model = align_models(model_files)

    df = compare_models(
        common_ids=common_ids,
        per_model=per_model,
        reference_model=args.reference_model,
        metrics=metrics,
        n_permutation=args.n_permutation,
        n_bootstrap=args.n_bootstrap,
        significance_method=args.significance_method,
        random_state=args.random_state,
        chexbert_labels=chexbert,
        radgraph_rewards=radgraph,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Per-model point estimate + 95% bootstrap CI (consolidated, main.csv-free).
    if args.metrics_ci_csv:
        ci_df = compute_metrics_ci(
            common_ids=common_ids,
            per_model=per_model,
            metrics=metrics,
            n_bootstrap=args.n_bootstrap,
            random_state=args.random_state,
            chexbert_labels=chexbert,
            radgraph_rewards=radgraph,
        )
        if ci_df.empty:
            print(f"\nNo metric-CI rows for dataset={args.dataset!r}; CSV unchanged.")
        else:
            if args.dataset:
                ci_df.insert(0, "dataset", args.dataset)
            if os.path.exists(args.metrics_ci_csv):
                ci_df = pd.concat([pd.read_csv(args.metrics_ci_csv), ci_df], ignore_index=True)
            ci_df.to_csv(args.metrics_ci_csv, index=False)
            print(f"\nSaved {len(ci_df)} metric-CI rows -> {args.metrics_ci_csv}")

    if args.dataset:
        combined_path = args.combined_csv or os.path.join(
            args.output_dir, "pairwise_significance_all.csv"
        )
        if df.empty:
            print(f"\nNo rows for dataset={args.dataset!r}; combined CSV unchanged.")
        else:
            df.insert(0, "dataset", args.dataset)
            if os.path.exists(combined_path):
                existing = pd.read_csv(combined_path)
                df = pd.concat([existing, df], ignore_index=True)
            df.to_csv(combined_path, index=False)
            print(f"\nSaved {len(df)} total rows -> {combined_path}")
            print(df.to_string())
    else:
        out_path = os.path.join(args.output_dir, "pairwise_significance.csv")
        df.to_csv(out_path, index=False)
        print(f"\nSaved {len(df)} rows -> {out_path}")
        if not df.empty:
            print(df.to_string())


if __name__ == "__main__":
    main()
