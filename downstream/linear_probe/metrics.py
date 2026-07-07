"""Classification metrics used by SmartX downstream linear probes."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _safe_float(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def _valid_binary_columns(y_true: np.ndarray) -> list[int]:
    return [
        idx
        for idx in range(y_true.shape[1])
        if len(np.unique(y_true[:, idx])) == 2
    ]


def meanAUC(ground_truth, predictions):
    ground_truth = np.asarray(ground_truth)
    predictions = np.asarray(predictions)
    auc_scores = []
    for idx in _valid_binary_columns(ground_truth):
        auc_scores.append(roc_auc_score(ground_truth[:, idx], predictions[:, idx]))
    return float(np.mean(auc_scores)) if auc_scores else None, auc_scores


def meanAP(ground_truth, predictions):
    ground_truth = np.asarray(ground_truth)
    predictions = np.asarray(predictions)
    ap_scores = []
    for idx in _valid_binary_columns(ground_truth):
        ap_scores.append(average_precision_score(ground_truth[:, idx], predictions[:, idx]))
    return float(np.mean(ap_scores)) if ap_scores else None, ap_scores


def meanF1(ground_truth, predictions):
    ground_truth = np.asarray(ground_truth)
    predictions = np.asarray(predictions)
    f1_scores = []
    for idx in _valid_binary_columns(ground_truth):
        fpr, tpr, thresholds = roc_curve(ground_truth[:, idx], predictions[:, idx])
        threshold = thresholds[np.argmax(tpr - fpr)]
        binary_predictions = (predictions[:, idx] > threshold).astype(int)
        f1_scores.append(f1_score(ground_truth[:, idx], binary_predictions, zero_division=0))
    return float(np.mean(f1_scores)) if f1_scores else None, f1_scores


def meanMCC(ground_truth, predictions):
    ground_truth = np.asarray(ground_truth)
    predictions = np.asarray(predictions)
    mcc_scores = []
    for idx in _valid_binary_columns(ground_truth):
        fpr, tpr, thresholds = roc_curve(ground_truth[:, idx], predictions[:, idx])
        threshold = thresholds[np.argmax(tpr - fpr)]
        binary_predictions = (predictions[:, idx] > threshold).astype(int)
        mcc_scores.append(matthews_corrcoef(ground_truth[:, idx], binary_predictions))
    return float(np.mean(mcc_scores)) if mcc_scores else None, mcc_scores


def _as_multiclass_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim > 1 and labels.shape[1] > 1:
        return np.argmax(labels, axis=1)
    return labels.reshape(-1).astype(int)


def compute_classification_metrics(
    y_true,
    y_score,
    *,
    task_type: str,
    threshold: float = 0.5,
) -> dict[str, object]:
    """Compute release-safe classification metrics.

    ``task_type`` follows the upstream naming: ``binary``, ``multi-label``, or
    ``multi-class``.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    metrics: dict[str, object] = {"task_type": task_type, "num_samples": int(len(y_true))}

    if task_type == "multi-class":
        class_labels = _as_multiclass_labels(y_true)
        probabilities = y_score
        if probabilities.ndim == 1:
            probabilities = np.stack([1.0 - probabilities, probabilities], axis=1)
        predictions = np.argmax(probabilities, axis=1)
        metrics.update(
            {
                "accuracy": _safe_float(accuracy_score(class_labels, predictions)),
                "f1_macro": _safe_float(f1_score(class_labels, predictions, average="macro", zero_division=0)),
                "precision_macro": _safe_float(precision_score(class_labels, predictions, average="macro", zero_division=0)),
                "recall_macro": _safe_float(recall_score(class_labels, predictions, average="macro", zero_division=0)),
            }
        )
        try:
            metrics["auc_macro"] = _safe_float(
                roc_auc_score(class_labels, probabilities, multi_class="ovr", average="macro")
            )
        except ValueError:
            metrics["auc_macro"] = None
        return metrics

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_score.ndim == 1:
        y_score = y_score.reshape(-1, 1)

    predictions = (y_score >= threshold).astype(int)
    if task_type == "binary" and y_true.shape[1] == 1:
        yt = y_true[:, 0]
        yp = predictions[:, 0]
        ys = y_score[:, 0]
        metrics.update(
            {
                "accuracy": _safe_float(accuracy_score(yt, yp)),
                "f1_macro": _safe_float(f1_score(yt, yp, average="binary", zero_division=0)),
                "precision_macro": _safe_float(precision_score(yt, yp, average="binary", zero_division=0)),
                "recall_macro": _safe_float(recall_score(yt, yp, average="binary", zero_division=0)),
                "mcc_macro": _safe_float(matthews_corrcoef(yt, yp)),
            }
        )
        try:
            metrics["auc_macro"] = _safe_float(roc_auc_score(yt, ys))
        except ValueError:
            metrics["auc_macro"] = None
        try:
            metrics["ap_macro"] = _safe_float(average_precision_score(yt, ys))
        except ValueError:
            metrics["ap_macro"] = None
        return metrics

    auc_macro, auc_per_class = meanAUC(y_true, y_score)
    ap_macro, ap_per_class = meanAP(y_true, y_score)
    f1_macro, f1_per_class = meanF1(y_true, y_score)
    mcc_macro, mcc_per_class = meanMCC(y_true, y_score)
    metrics.update(
        {
            "accuracy": _safe_float(np.mean(predictions == y_true)),
            "subset_accuracy": _safe_float(np.mean(np.all(predictions == y_true, axis=1))),
            "auc_macro": _safe_float(auc_macro),
            "ap_macro": _safe_float(ap_macro),
            "f1_macro": _safe_float(f1_macro),
            "mcc_macro": _safe_float(mcc_macro),
            "auc_per_class": auc_per_class,
            "ap_per_class": ap_per_class,
            "f1_per_class": f1_per_class,
            "mcc_per_class": mcc_per_class,
        }
    )
    return metrics
