"""Unified percentile bootstrap CI for report-generation metrics.

Every report-generation metric (BLEU, ROUGE, F1-RadGraph, CheXBERT-F1) reports
its 95% CI through this single helper so the conventions stay identical across
metrics and match ``utils/stat_test.py`` (and the paper Methods section):

  - **percentile** bootstrap (not normal-approx, not BCa)
  - ``n_resamples = 1000``
  - ``confidence = 0.95``
  - ``seed = 42`` (shared with the permutation tests in utils/stat_test.py)
  - the reported point estimate (``"median"`` field) is the statistic evaluated
    on the FULL sample (the observed value) — NOT the bootstrap-distribution
    median, which is biased for skewed metrics like F1 / RadGraph.

The metric is always expressed as ``statistic(indices) -> float`` operating on
report indices, so resampling is just resampling the index set. This works
uniformly for corpus-level metrics (BLEU, micro/macro-F1, recomputed on the
resampled reports) and for sample-averaged metrics (ROUGE, RadGraph, mean over
the resampled per-report scores).
"""

from __future__ import annotations

from typing import Callable, Sequence

from scipy.stats import bootstrap

SEED = 42
N_RESAMPLES = 1000
CONFIDENCE = 0.95


def percentile_ci(
    statistic: Callable[[Sequence[int]], float],
    n: int,
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    confidence: float = CONFIDENCE,
) -> dict[str, float]:
    """Percentile bootstrap CI + observed point estimate for a corpus statistic.

    Args:
        statistic: ``fn(indices) -> float``. Given a (resampled) sequence of
            report indices, returns the metric value on those reports. Must
            accept a plain list as well as the numpy index array scipy passes.
        n: number of reports (the index set is ``range(n)``).
        n_resamples, seed, confidence: bootstrap settings (defaults are the
            project-wide unified conventions; override only if you know why).

    Returns:
        ``{"median": observed, "ci_l": low, "ci_h": high}`` where ``observed``
        is ``statistic(range(n))`` and the CI is a percentile interval.
    """
    observed = float(statistic(list(range(n))))
    bs = bootstrap(
        data=(list(range(n)),),
        statistic=statistic,
        method="percentile",
        paired=False,
        vectorized=False,
        confidence_level=confidence,
        random_state=seed,
        n_resamples=n_resamples,
    )
    return {
        "median": observed,
        "ci_l": float(bs.confidence_interval.low),
        "ci_h": float(bs.confidence_interval.high),
    }
