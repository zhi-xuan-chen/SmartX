import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def _safe_cov(matrix):
    """
    Compute covariance while handling the degenerate case with <= 1 observation.

    `matrix` is expected to have shape (n_classifiers, n_observations).
    """
    n_classifiers, n_observations = matrix.shape
    if n_observations <= 1:
        if n_classifiers == 1:
            return 0.0
        return np.zeros((n_classifiers, n_classifiers), dtype=float)
    return np.cov(matrix)


def compute_midrank(x):
    """Computes midranks for a 1D numpy array."""
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1)
        i = k

    out = np.empty(n, dtype=float)
    out[j] = t + 1
    return out


def fast_delong(predictions_sorted_transposed, label_1_count):
    """Fast DeLong implementation from Sun & Xu (2014)."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = _safe_cov(v01)
    sy = _safe_cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def calc_pvalue(aucs, sigma):
    """Computes log10(p-value) for the difference between two correlated AUCs."""
    contrast = np.array([[1, -1]])
    variance = np.dot(np.dot(contrast, sigma), contrast.T)
    variance_scalar = float(np.asarray(variance).item())
    if variance_scalar <= 0:
        diff = float(np.abs(np.diff(aucs)).item())
        return 0.0 if diff == 0.0 else -np.inf
    z = float(np.abs(np.diff(aucs)).item()) / np.sqrt(variance_scalar)
    return np.log10(2.0) + stats.norm.logsf(z, loc=0, scale=1) / np.log(10.0)


def compute_ground_truth_statistics(ground_truth):
    unique = np.unique(ground_truth)
    if len(unique) != 2 or not np.array_equal(unique, [0, 1]):
        raise ValueError("ground_truth must contain both 0 and 1")
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth, predictions):
    """Computes ROC AUC variance for a single predictor."""
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delong_cov = fast_delong(predictions_sorted_transposed, label_1_count)
    return aucs[0], float(np.asarray(delong_cov).item())


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """Computes log10(p-value) for the difference between two correlated AUCs."""
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delong_cov = fast_delong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delong_cov)


def delong_test(y_true, y_pred_a, y_pred_b, alpha=0.05):
    """Perform DeLong test to compare two AUCs."""
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0 or len(np.unique(y_true)) < 2:
        return {
            "auc_a": np.nan,
            "auc_b": np.nan,
            "auc_diff": np.nan,
            "p_value": np.nan,
            "significant": False,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "se_diff": np.nan,
        }

    try:
        auc_a = roc_auc_score(y_true, y_pred_a)
        auc_b = roc_auc_score(y_true, y_pred_b)
        log10_p = delong_roc_test(y_true, y_pred_a, y_pred_b)
        log10_p_scalar = float(np.asarray(log10_p).item())
        p_value = 10 ** log10_p_scalar

        auc_pair = np.array([auc_a, auc_b])
        order, label_1_count = compute_ground_truth_statistics(y_true)
        predictions_sorted_transposed = np.vstack((y_pred_a, y_pred_b))[:, order]
        _, delong_cov = fast_delong(predictions_sorted_transposed, label_1_count)
        contrast = np.array([[1, -1]])
        var_diff = float(np.asarray(np.dot(np.dot(contrast, delong_cov), contrast.T)).item())
        se_diff = np.sqrt(max(0.0, var_diff))
        z_crit = stats.norm.ppf(1 - alpha / 2)
        auc_diff = float(auc_pair[0] - auc_pair[1])
        ci_low = auc_diff - z_crit * se_diff
        ci_high = auc_diff + z_crit * se_diff
    except Exception:
        return {
            "auc_a": np.nan,
            "auc_b": np.nan,
            "auc_diff": np.nan,
            "p_value": np.nan,
            "significant": False,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "se_diff": np.nan,
        }

    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "auc_diff": auc_diff,
        "p_value": p_value,
        "significant": p_value < alpha,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "se_diff": se_diff,
    }


def compute_auc_with_ci(y_true, y_pred, alpha=0.05):
    """Compute AUC with DeLong-based confidence interval."""
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0 or len(np.unique(y_true)) < 2:
        return {
            "auc": np.nan,
            "auc_se": np.nan,
            "auc_ci_low": np.nan,
            "auc_ci_high": np.nan,
            "has_positives": n_pos > 0,
        }

    try:
        auc, variance = delong_roc_variance(y_true, y_pred)
        se = np.sqrt(variance) if variance >= 0 else np.nan
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_low = max(0.0, auc - z_crit * se)
        ci_high = min(1.0, auc + z_crit * se)
    except Exception:
        return {
            "auc": np.nan,
            "auc_se": np.nan,
            "auc_ci_low": np.nan,
            "auc_ci_high": np.nan,
            "has_positives": True,
        }

    return {
        "auc": auc,
        "auc_se": se,
        "auc_ci_low": ci_low,
        "auc_ci_high": ci_high,
        "has_positives": True,
    }


def paired_bootstrap_mean_test(
    y_true,
    y_pred_a,
    y_pred_b,
    n_bootstrap=2000,
    confidence_level=0.95,
    random_state=42,
):
    """Paired bootstrap hypothesis test for mean AUC comparison."""
    n_samples, n_classes = y_true.shape
    rng = np.random.RandomState(random_state)
    alpha = 1.0 - confidence_level

    valid_class_indices = []
    valid_classes_a = []
    valid_classes_b = []

    for class_idx in range(n_classes):
        y_class = y_true[:, class_idx]
        if len(np.unique(y_class)) < 2:
            continue
        try:
            auc_a = roc_auc_score(y_class, y_pred_a[:, class_idx])
            auc_b = roc_auc_score(y_class, y_pred_b[:, class_idx])
        except Exception:
            continue
        valid_class_indices.append(class_idx)
        valid_classes_a.append(auc_a)
        valid_classes_b.append(auc_b)

    if not valid_class_indices:
        return {
            "mean_auc_a": np.nan,
            "mean_auc_b": np.nan,
            "auc_diff": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "se_diff": np.nan,
            "p_value": np.nan,
            "significant": False,
            "n_bootstrap": n_bootstrap,
            "test_method": "paired_bootstrap",
        }

    mean_auc_a = float(np.mean(valid_classes_a))
    mean_auc_b = float(np.mean(valid_classes_b))
    observed_diff = mean_auc_a - mean_auc_b

    boot_diffs = []
    for _ in range(n_bootstrap):
        boot_indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[boot_indices]
        y_pred_a_boot = y_pred_a[boot_indices]
        y_pred_b_boot = y_pred_b[boot_indices]

        boot_aucs_a = []
        boot_aucs_b = []
        skip_iteration = False

        for class_idx in valid_class_indices:
            y_class_boot = y_true_boot[:, class_idx]
            if len(np.unique(y_class_boot)) < 2:
                skip_iteration = True
                break
            try:
                auc_a_boot = roc_auc_score(y_class_boot, y_pred_a_boot[:, class_idx])
                auc_b_boot = roc_auc_score(y_class_boot, y_pred_b_boot[:, class_idx])
            except Exception:
                skip_iteration = True
                break
            boot_aucs_a.append(auc_a_boot)
            boot_aucs_b.append(auc_b_boot)

        if not skip_iteration and boot_aucs_a:
            boot_diffs.append(float(np.mean(boot_aucs_a) - np.mean(boot_aucs_b)))

    boot_diffs = np.asarray(boot_diffs, dtype=float)
    if len(boot_diffs) == 0:
        return {
            "mean_auc_a": mean_auc_a,
            "mean_auc_b": mean_auc_b,
            "auc_diff": observed_diff,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "se_diff": np.nan,
            "p_value": np.nan,
            "significant": False,
            "n_bootstrap": n_bootstrap,
            "test_method": "paired_bootstrap",
        }

    ci_low = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_high = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    se_diff = np.std(boot_diffs, ddof=1) if len(boot_diffs) > 1 else 0.0
    centered_boot = boot_diffs - np.mean(boot_diffs)
    p_value = float(np.mean(np.abs(centered_boot) >= abs(observed_diff)))
    p_value = max(p_value, 1.0 / n_bootstrap)

    return {
        "mean_auc_a": mean_auc_a,
        "mean_auc_b": mean_auc_b,
        "auc_diff": observed_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "se_diff": se_diff,
        "p_value": p_value,
        "significant": p_value < alpha,
        "n_bootstrap": n_bootstrap,
        "test_method": "paired_bootstrap",
    }
