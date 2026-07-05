import math

import numpy as np

from utils.stat_test import (
    make_accuracy_metric,
    paired_bootstrap_diff_ci,
    paired_bootstrap_wilcoxon_test,
)


def test_paired_bootstrap_diff_ci_can_return_distribution():
    metric = make_accuracy_metric()
    data_a = np.array([1, 1, 1, 0, 1, 1])
    data_b = np.array([0, 1, 0, 0, 1, 0])

    result = paired_bootstrap_diff_ci(
        metric,
        data_a,
        data_b,
        n_resamples=25,
        random_state=0,
        return_distribution=True,
    )

    assert result["bootstrap_distribution"].shape == (25,)
    assert math.isclose(result["observed_diff"], metric(data_a) - metric(data_b))


def test_paired_bootstrap_wilcoxon_test_uses_bootstrap_distribution():
    metric = make_accuracy_metric()
    data_a = np.array([1, 1, 1, 0, 1, 1])
    data_b = np.array([0, 1, 0, 0, 1, 0])

    result = paired_bootstrap_wilcoxon_test(
        metric,
        data_a,
        data_b,
        n_resamples=25,
        random_state=0,
    )

    assert result["observed_diff"] > 0
    assert result["statistic"] >= 0
    assert 0 <= result["p_value"] <= 1
    assert result["n_bootstrap_effective"] == 25
