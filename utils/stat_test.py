"""Shared statistical-testing utilities used across downstream tasks.

Three downstream tasks (MIL linear probe, report generation, VQA) all rely on
the same two primitives:

1. ``paired_permutation_test`` — paired permutation test for any user-defined
   metric. Wraps ``scipy.stats.permutation_test`` with sensible defaults.
2. ``add_fdr_by_group`` — Benjamini–Hochberg FDR correction on a DataFrame
   column, optionally grouped by other columns (e.g. per-region).

Unified conventions (must match the paper Methods section):

- Bootstrap CI: percentile, n=1000, confidence=0.95, point estimate = observed
  value on the full sample (not the bootstrap-distribution median). For report
  generation these are implemented in
  ``downstream/report_generation/llavarad/llava/eval/rrg_eval/rrg_eval/bootstrap_ci.py``
  (every metric routes through ``percentile_ci``).
- Permutation test: n=10000, permutation_type='samples', two-sided
- FDR correction: Benjamini–Hochberg at 5%
- Global random seed: 42 (shared by the bootstrap CIs and the permutation tests)
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


DEFAULT_SEED = 42
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_N_PERMUTATION = 10000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_ALPHA = 0.05


def paired_permutation_test(
    metric_fn: Callable[[np.ndarray], float],
    data_a: np.ndarray,
    data_b: np.ndarray,
    *,
    n_resamples: int = DEFAULT_N_PERMUTATION,
    random_state: int = DEFAULT_SEED,
    alternative: str = "two-sided",
    axis: int = 0,
) -> dict:
    """Paired permutation test comparing two models' outputs on the same samples.

    Under H0 (both models equivalent), swapping the per-sample outputs between
    the two arms is exchangeable. We swap each pair independently with prob.
    0.5 and recompute ``metric_fn(a) - metric_fn(b)`` to build the null
    distribution.

    Note on scipy
    -------------
    We deliberately do **not** use ``scipy.stats.permutation_test``. With
    ``vectorized=False`` scipy slices the input to 1-D along its ``axis`` arg
    via ``np.apply_along_axis``, which breaks any ``metric_fn`` that needs the
    full multi-dim array (e.g. mean AUC over a ``(n_samples, n_classes)``
    array, or corpus-level BLEU over a list of strings). Writing the loop
    ourselves keeps ``metric_fn`` free to operate on whatever shape the caller
    provides, and also lets us use the Phipson & Smyth (2010) ``+1`` p-value
    correction by default, which guarantees ``p > 0`` under H0.

    Args:
        metric_fn: Function mapping a single array (predictions, labels, or
            any per-sample outputs) to a scalar metric. Must accept an array
            with shape compatible to ``data_a``/``data_b``.
        data_a, data_b: Per-sample outputs from the two models, paired along
            ``axis``. For multi-class outputs (e.g. shape (n_samples, n_classes))
            ``axis=0`` is correct.
        n_resamples: Number of permutations. 10000 is enough for p≈1e-3 stability.
        random_state: Seed.
        alternative: 'two-sided', 'greater', or 'less'.
        axis: Sample axis along which to permute.

    Returns:
        Dict with keys: ``observed_diff``, ``p_value``, ``null_distribution``.
    """
    data_a = np.asarray(data_a)
    data_b = np.asarray(data_b)
    if data_a.shape != data_b.shape:
        raise ValueError(
            f"data_a and data_b must have the same shape, got {data_a.shape} vs {data_b.shape}"
        )
    if axis != 0:
        data_a = np.moveaxis(data_a, axis, 0)
        data_b = np.moveaxis(data_b, axis, 0)

    n_samples = data_a.shape[0]
    rng = np.random.default_rng(random_state)

    observed_diff = float(metric_fn(data_a) - metric_fn(data_b))

    null = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        swap = rng.random(n_samples) < 0.5
        a_perm = np.where(swap[(slice(None),) + (None,) * (data_a.ndim - 1)], data_b, data_a)
        b_perm = np.where(swap[(slice(None),) + (None,) * (data_b.ndim - 1)], data_a, data_b)
        null[i] = float(metric_fn(a_perm) - metric_fn(b_perm))

    if alternative == "two-sided":
        # P(|T*| >= |T_obs|); +1/+1 small-sample correction (Phipson & Smyth 2010)
        count = int(np.sum(np.abs(null) >= abs(observed_diff)))
    elif alternative == "greater":
        count = int(np.sum(null >= observed_diff))
    elif alternative == "less":
        count = int(np.sum(null <= observed_diff))
    else:
        raise ValueError(f"unknown alternative: {alternative!r}")
    p_value = (count + 1) / (n_resamples + 1)

    return {
        "observed_diff": observed_diff,
        "p_value": float(p_value),
        "null_distribution": null,
    }


def paired_bootstrap_diff_ci(
    metric_fn: Callable[[np.ndarray], float],
    data_a: np.ndarray,
    data_b: np.ndarray,
    *,
    n_resamples: int = DEFAULT_N_BOOTSTRAP,
    random_state: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
    axis: int = 0,
    return_distribution: bool = False,
) -> dict:
    """Percentile bootstrap CI for the *paired* difference ``metric(a) - metric(b)``.

    Complements ``paired_permutation_test``: the permutation test gives a
    p-value for H0 (no difference), while this gives an effect-size interval.
    Both consume the same paired ``data_a``/``data_b`` (already aligned by the
    caller) and the same ``metric_fn``, so they stay mutually consistent.

    Pairing is preserved: each bootstrap iteration draws **one** set of sample
    indices (with replacement) and applies it to *both* arms before recomputing
    the difference. This is the correct resampling unit for a paired contrast
    and matches the question-level resampling used for the per-model CIs.

    .. warning::
        ``metric_fn`` must be **self-contained**: everything it needs has to
        live inside ``data_a``/``data_b``, because only those arrays are
        resampled. Do NOT use this with a metric that closes over external
        per-sample state (e.g. ``make_mean_auc_metric``, which holds a fixed
        ``y_true``): the labels would stay in original order while predictions
        are resampled, breaking the per-sample alignment and collapsing the
        bootstrap diffs toward 0. For AUC use
        ``delong_utils.paired_bootstrap_mean_test`` (resamples ``y_true`` with
        the predictions); for threshold metrics pass labels+preds stacked into
        one array, or use the permutation test (which permutes in place and is
        alignment-safe).

    The point estimate is the observed difference on the full sample (NOT the
    bootstrap-distribution median), matching the convention in ``percentile_ci``
    and ``bootstrap_accuracy_ci``.

    Args:
        metric_fn: Function mapping a single array to a scalar metric. Must
            accept an array shaped like ``data_a``/``data_b``.
        data_a, data_b: Per-sample outputs from the two models, paired along
            ``axis``.
        n_resamples: Number of bootstrap resamples.
        random_state: Seed.
        confidence: Two-sided confidence level (0.95 -> 2.5/97.5 percentiles).
        axis: Sample axis to resample along.
        return_distribution: Include the bootstrap difference distribution in
            the return dict. This is mainly for secondary analyses that consume
            the resampled effects directly.

    Returns:
        Dict with keys ``observed_diff``, ``diff_ci_low``, ``diff_ci_high``.
        When ``return_distribution=True``, also includes
        ``bootstrap_distribution``.
    """
    data_a = np.asarray(data_a)
    data_b = np.asarray(data_b)
    if data_a.shape != data_b.shape:
        raise ValueError(
            f"data_a and data_b must have the same shape, got {data_a.shape} vs {data_b.shape}"
        )
    if axis != 0:
        data_a = np.moveaxis(data_a, axis, 0)
        data_b = np.moveaxis(data_b, axis, 0)

    n_samples = data_a.shape[0]
    observed_diff = float(metric_fn(data_a) - metric_fn(data_b))

    if n_samples == 0:
        result = {
            "observed_diff": observed_diff,
            "diff_ci_low": np.nan,
            "diff_ci_high": np.nan,
        }
        if return_distribution:
            result["bootstrap_distribution"] = np.array([], dtype=float)
        return result

    rng = np.random.default_rng(random_state)
    diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n_samples, size=n_samples)
        diffs[i] = float(metric_fn(data_a[idx]) - metric_fn(data_b[idx]))

    alpha = 1.0 - confidence
    lo, hi = np.nanpercentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    result = {
        "observed_diff": observed_diff,
        "diff_ci_low": float(lo),
        "diff_ci_high": float(hi),
    }
    if return_distribution:
        result["bootstrap_distribution"] = diffs
    return result


def paired_bootstrap_wilcoxon_test(
    metric_fn: Callable[[np.ndarray], float],
    data_a: np.ndarray,
    data_b: np.ndarray,
    *,
    n_resamples: int = DEFAULT_N_BOOTSTRAP,
    random_state: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
    alternative: str = "two-sided",
    zero_method: str = "wilcox",
    axis: int = 0,
) -> dict:
    """Wilcoxon signed-rank test over paired bootstrap metric differences.

    This treats the paired bootstrap effect estimates as the signed samples and
    tests whether their median differs from zero. It is exposed as an optional
    sensitivity analysis for callers that explicitly want significance from the
    bootstrap distribution; ``paired_permutation_test`` remains the default
    exchangeability-based paired test.
    """
    diff_ci = paired_bootstrap_diff_ci(
        metric_fn,
        data_a,
        data_b,
        n_resamples=n_resamples,
        random_state=random_state,
        confidence=confidence,
        axis=axis,
        return_distribution=True,
    )
    diffs = np.asarray(diff_ci["bootstrap_distribution"], dtype=float)
    diffs = diffs[np.isfinite(diffs)]

    # Tie / zero diagnostics. On low-diversity samples (e.g. a small subgroup
    # where accuracy can only take a few discrete values) the bootstrap diffs
    # land on a coarse grid: many exact zeros (dropped by zero_method='wilcox',
    # shrinking the effective n) and many repeated values (Wilcoxon's normal
    # approximation degrades). These two fractions let callers flag results whose
    # Wilcoxon p-value is not trustworthy.
    if diffs.size == 0:
        statistic = np.nan
        p_value = np.nan
        zero_frac = np.nan
        tie_frac = np.nan
    else:
        n_zero = int(np.sum(diffs == 0.0))
        n_unique = int(np.unique(diffs).size)
        zero_frac = float(n_zero / diffs.size)
        tie_frac = float(1.0 - n_unique / diffs.size)
        try:
            result = stats.wilcoxon(
                diffs,
                zero_method=zero_method,
                alternative=alternative,
            )
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
        except ValueError:
            statistic = np.nan
            p_value = np.nan

    return {
        "observed_diff": diff_ci["observed_diff"],
        "diff_ci_low": diff_ci["diff_ci_low"],
        "diff_ci_high": diff_ci["diff_ci_high"],
        "statistic": statistic,
        "p_value": p_value,
        "n_bootstrap_effective": int(diffs.size),
        "zero_frac": zero_frac,
        "tie_frac": tie_frac,
    }


def paired_bootstrap_pvalue_test(
    metric_fn: Callable[[np.ndarray], float],
    data_a: np.ndarray,
    data_b: np.ndarray,
    *,
    n_resamples: int = DEFAULT_N_BOOTSTRAP,
    random_state: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
    axis: int = 0,
) -> dict:
    """Two-sided bootstrap p-value from the paired difference distribution.

    Builds the paired bootstrap distribution of ``metric(a) - metric(b)`` (same
    resampling as ``paired_bootstrap_diff_ci``) and reports the achieved
    significance level (ASL):

        p = 2 * min( P(diff* <= 0), P(diff* >= 0) ), capped at 1.0

    i.e. twice the smaller tail mass on the wrong side of zero. This is the
    bootstrap analogue of "does the CI cross zero" expressed as a p-value, and
    unlike ``paired_bootstrap_wilcoxon_test`` it does not treat the resamples as
    independent observations. The ``+1`` numerator/denominator correction keeps
    ``p > 0`` even when no resample lands on the far side of zero (Davison &
    Hinkley 1997, §4.2), matching the floor convention in
    ``paired_permutation_test``.

    Returns the same keys as the other paired tests so callers can swap methods.
    """
    diff_ci = paired_bootstrap_diff_ci(
        metric_fn,
        data_a,
        data_b,
        n_resamples=n_resamples,
        random_state=random_state,
        confidence=confidence,
        axis=axis,
        return_distribution=True,
    )
    diffs = np.asarray(diff_ci["bootstrap_distribution"], dtype=float)
    diffs = diffs[np.isfinite(diffs)]

    n_eff = int(diffs.size)
    if n_eff == 0:
        p_value = np.nan
    else:
        n_le = int(np.sum(diffs <= 0.0))
        n_ge = int(np.sum(diffs >= 0.0))
        # +1/+1 correction so p > 0; mirrors paired_permutation_test's floor.
        p_le = (n_le + 1) / (n_eff + 1)
        p_ge = (n_ge + 1) / (n_eff + 1)
        p_value = float(min(1.0, 2.0 * min(p_le, p_ge)))

    return {
        "observed_diff": diff_ci["observed_diff"],
        "diff_ci_low": diff_ci["diff_ci_low"],
        "diff_ci_high": diff_ci["diff_ci_high"],
        "p_value": p_value,
        "n_bootstrap_effective": n_eff,
    }


def percentile_ci(
    statistic: Callable[[Sequence[int]], float],
    n: int,
    *,
    n_resamples: int = DEFAULT_N_BOOTSTRAP,
    random_state: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict:
    """Percentile bootstrap CI + observed point estimate for a corpus statistic.

    Mirrors the report-generation eval-side helper
    (``rrg_eval/rrg_eval/bootstrap_ci.py``): same method (percentile), same
    defaults (n=1000, seed=42, 95%). The metric is expressed as
    ``statistic(indices) -> float`` over a (resampled) index set, so this works
    uniformly for corpus-level metrics (BLEU, micro/macro-F1, recomputed on the
    resampled reports) and sample-averaged metrics (ROUGE, RadGraph, mean over
    the resampled per-report scores).

    Args:
        statistic: ``fn(indices) -> float``. Must accept a plain list as well as
            the numpy index array scipy passes.
        n: number of reports (the index set is ``range(n)``).

    Returns:
        ``{"point": observed, "ci_l": low, "ci_h": high}`` where ``point`` is
        ``statistic(range(n))`` (the observed value on the full sample, NOT the
        bootstrap-distribution median) and the CI is a percentile interval.
    """
    from scipy.stats import bootstrap

    observed = float(statistic(list(range(n))))
    bs = bootstrap(
        data=(list(range(n)),),
        statistic=statistic,
        method="percentile",
        paired=False,
        vectorized=False,
        confidence_level=confidence,
        random_state=random_state,
        n_resamples=n_resamples,
    )
    return {
        "point": observed,
        "ci_l": float(bs.confidence_interval.low),
        "ci_h": float(bs.confidence_interval.high),
    }


def add_fdr_by_group(
    df: pd.DataFrame,
    p_col: str = "p_value",
    group_cols: Sequence[str] | None = None,
    out_col: str = "p_value_fdr",
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Add Benjamini–Hochberg FDR-corrected p-values to a DataFrame.

    The correction is applied **within each group** defined by ``group_cols``.
    Grouping matters: BH should be applied per "family of hypotheses" (e.g.
    per anatomical region, per metric). Mixing unrelated tests inflates m
    and destroys power.

    Args:
        df: Input DataFrame containing column ``p_col``.
        p_col: Name of column holding raw p-values.
        group_cols: Columns to group by before applying BH. If None, BH is
            applied across the whole table.
        out_col: Name of new column for corrected p-values.
        alpha: FDR level used to flag significance.

    Returns:
        New DataFrame with two added columns: ``out_col`` and ``significant_fdr``.
    """
    df = df.copy()
    df[out_col] = np.nan

    def _apply(sub: pd.DataFrame) -> None:
        mask = sub[p_col].notna()
        if mask.sum() == 0:
            return
        _, p_adj, _, _ = multipletests(sub.loc[mask, p_col].values, method="fdr_bh")
        df.loc[sub.index[mask], out_col] = p_adj

    if group_cols is None:
        _apply(df)
    else:
        for _, idx in df.groupby(list(group_cols), dropna=False).groups.items():
            _apply(df.loc[idx])

    df["significant_fdr"] = df[out_col] < alpha
    return df


# -----------------------------------------------------------------------------
# Convenience metric_fn factories for the three downstream tasks
# -----------------------------------------------------------------------------


def make_mean_auc_metric(y_true: np.ndarray) -> Callable[[np.ndarray], float]:
    """Factory: build a metric_fn that returns mean AUC over valid classes.

    The returned function takes per-sample prediction probabilities
    (shape ``(n_samples, n_classes)``) and computes the macro-mean AUC over
    classes that contain both labels in ``y_true``.
    """
    from sklearn.metrics import roc_auc_score

    n_samples, n_classes = y_true.shape
    valid_classes: list[int] = []
    for c in range(n_classes):
        if len(np.unique(y_true[:, c])) >= 2:
            valid_classes.append(c)

    def metric_fn(y_pred: np.ndarray) -> float:
        if not valid_classes:
            return np.nan
        aucs = []
        for c in valid_classes:
            try:
                aucs.append(roc_auc_score(y_true[:, c], y_pred[:, c]))
            except ValueError:
                continue
        return float(np.mean(aucs)) if aucs else np.nan

    return metric_fn


# -----------------------------------------------------------------------------
# Threshold-dependent metric factories (macro-mean over valid classes)
#
# These consume *binary* predictions (the val-threshold ``prediction`` column,
# the same one F1/MCC use in the bootstrap CIs) shaped ``(n_samples, n_classes)``
# and return the macro-mean of the metric over valid classes. They are
# vectorised over classes (pure numpy, no sklearn per-class loop) so they stay
# cheap inside a 10k-iteration permutation loop. Class validity and the per-class
# positive/negative counts are derived once from ``y_true`` (fixed across
# permutations), matching the per-class inclusion rule used in
# ``downstream/mil_linear_probe/statistical_analysis/bootstrap.py``:
#   - F1  includes any class with >=1 positive (zero_division=0 elsewhere)
#   - MCC / balanced-accuracy need both labels present in the class
# -----------------------------------------------------------------------------


def _binary_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    """Per-class (TP, FP, FN, TN) for binary multi-label arrays, vectorised."""
    yp = np.asarray(y_pred)
    tp = (y_true * yp).sum(axis=0)
    fp = ((1 - y_true) * yp).sum(axis=0)
    fn = (y_true * (1 - yp)).sum(axis=0)
    tn = ((1 - y_true) * (1 - yp)).sum(axis=0)
    return tp, fp, fn, tn


def make_mean_f1_metric(y_true: np.ndarray) -> Callable[[np.ndarray], float]:
    """Factory: macro-mean F1 over classes with at least one positive.

    Matches ``sklearn.f1_score(..., zero_division=0)`` per class, then averages
    over classes that contain a positive in ``y_true``.
    """
    y_true = np.asarray(y_true)
    valid = y_true.sum(axis=0) > 0

    def metric_fn(y_pred: np.ndarray) -> float:
        if not valid.any():
            return np.nan
        tp, fp, fn, _ = _binary_confusion_counts(y_true, y_pred)
        denom = 2 * tp + fp + fn
        f1 = np.divide(2 * tp, denom, out=np.zeros(denom.shape, dtype=float), where=denom > 0)
        return float(f1[valid].mean())

    return metric_fn


def make_mean_mcc_metric(y_true: np.ndarray) -> Callable[[np.ndarray], float]:
    """Factory: macro-mean Matthews corrcoef over classes with both labels.

    Matches ``sklearn.matthews_corrcoef`` per class (0 when any factor is 0),
    then averages over classes where ``y_true`` has both a positive and a
    negative.
    """
    y_true = np.asarray(y_true)
    pos = y_true.sum(axis=0)
    neg = (1 - y_true).sum(axis=0)
    valid = (pos > 0) & (neg > 0)

    def metric_fn(y_pred: np.ndarray) -> float:
        if not valid.any():
            return np.nan
        tp, fp, fn, tn = _binary_confusion_counts(y_true, y_pred)
        num = tp * tn - fp * fn
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.divide(num, den, out=np.zeros(den.shape, dtype=float), where=den > 0)
        return float(mcc[valid].mean())

    return metric_fn


def make_mean_bacc_metric(y_true: np.ndarray) -> Callable[[np.ndarray], float]:
    """Factory: macro-mean balanced accuracy over classes with both labels.

    Balanced accuracy = (sensitivity + specificity) / 2, matching
    ``sklearn.balanced_accuracy_score`` for the binary case. Averaged over
    classes where ``y_true`` has both a positive and a negative.
    """
    y_true = np.asarray(y_true)
    pos = y_true.sum(axis=0)
    neg = (1 - y_true).sum(axis=0)
    valid = (pos > 0) & (neg > 0)
    pos_safe = np.where(pos > 0, pos, 1)
    neg_safe = np.where(neg > 0, neg, 1)

    def metric_fn(y_pred: np.ndarray) -> float:
        if not valid.any():
            return np.nan
        tp, _, _, tn = _binary_confusion_counts(y_true, y_pred)
        tpr = tp / pos_safe
        tnr = tn / neg_safe
        bacc = 0.5 * (tpr + tnr)
        return float(bacc[valid].mean())

    return metric_fn


# Registry so callers can look up the right factory by short metric name.
THRESHOLD_METRIC_FACTORIES: dict[str, Callable[[np.ndarray], Callable[[np.ndarray], float]]] = {
    "f1": make_mean_f1_metric,
    "mcc": make_mean_mcc_metric,
    "bacc": make_mean_bacc_metric,
}


def _threshold_per_class(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-class threshold metric vector (NaN where undefined on this data).

    F1 follows ``zero_division=0`` (0, not NaN, when 2TP+FP+FN=0). MCC/b-acc
    return NaN for a class lacking a positive or a negative, so callers can
    drop it from a macro mean for that resample.
    """
    tp, fp, fn, tn = _binary_confusion_counts(y_true, y_pred)
    if metric == "f1":
        den = 2 * tp + fp + fn
        return np.divide(2 * tp, den, out=np.zeros(den.shape, dtype=float), where=den > 0)
    if metric == "mcc":
        num = tp * tn - fp * fn
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.divide(num, den, out=np.zeros(den.shape, dtype=float), where=den > 0)
        pos, neg = tp + fn, tn + fp
        return np.where((pos > 0) & (neg > 0), mcc, np.nan)
    if metric == "bacc":
        pos, neg = tp + fn, tn + fp
        tpr = np.divide(tp, pos, out=np.full(pos.shape, np.nan, dtype=float), where=pos > 0)
        tnr = np.divide(tn, neg, out=np.full(neg.shape, np.nan, dtype=float), where=neg > 0)
        return 0.5 * (tpr + tnr)
    raise ValueError(f"unknown threshold metric: {metric!r}")


def paired_bootstrap_threshold_diff_ci(
    metric: str,
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    *,
    n_resamples: int = DEFAULT_N_BOOTSTRAP,
    random_state: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict:
    """Alignment-safe paired bootstrap CI for a threshold-metric difference.

    The macro-mean of F1/MCC/b-acc closes over ``y_true``, so a predictions-only
    resampler (``paired_bootstrap_diff_ci``) would misalign labels. Here every
    iteration draws one set of row indices and applies it to ``y_true`` **and**
    both prediction arms, so labels stay paired with predictions.

    The included-class set is **fixed once from the full sample** (the same
    estimand as the point value): F1 includes classes with >=1 positive,
    MCC/b-acc classes with both labels. Within a resample, a fixed-set class that
    degenerates (rare; only ultra-low-prevalence classes) is dropped from that
    resample's mean only. ``y_true``/preds are (n_samples, k); k=1 gives a
    single-disease CI, k=n_classes the macro-mean CI.

    Returns ``{observed_diff, diff_ci_low, diff_ci_high}`` (percentile interval).
    """
    y_true = np.asarray(y_true)
    a = np.asarray(y_pred_a)
    b = np.asarray(y_pred_b)
    n = y_true.shape[0]

    pos = y_true.sum(axis=0)
    neg = (1 - y_true).sum(axis=0)
    valid = pos > 0 if metric == "f1" else (pos > 0) & (neg > 0)

    def macro(yt, yp):
        if not valid.any():
            return np.nan
        vals = _threshold_per_class(metric, yt, yp)[valid]
        vals = vals[~np.isnan(vals)]
        return float(vals.mean()) if vals.size else np.nan

    observed = macro(y_true, a) - macro(y_true, b)
    if n == 0 or not valid.any():
        return {"observed_diff": observed, "diff_ci_low": np.nan, "diff_ci_high": np.nan}

    rng = np.random.default_rng(random_state)
    diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = macro(y_true[idx], a[idx]) - macro(y_true[idx], b[idx])

    alpha = 1.0 - confidence
    lo, hi = np.nanpercentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"observed_diff": float(observed), "diff_ci_low": float(lo), "diff_ci_high": float(hi)}


def make_accuracy_metric() -> Callable[[np.ndarray], float]:
    """Metric_fn for VQA-style 0/1 correctness arrays."""

    def metric_fn(correct: np.ndarray) -> float:
        if len(correct) == 0:
            return np.nan
        return float(np.mean(correct))

    return metric_fn
