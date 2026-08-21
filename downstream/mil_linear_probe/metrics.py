"""Validation-thresholded metrics for public MIL evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_curve,
)

from downstream.linear_probe.metrics import compute_classification_metrics


def fit_youden_thresholds(labels, scores) -> np.ndarray:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.ndim == 1:
        labels = labels[:, None]
    if scores.ndim == 1:
        scores = scores[:, None]
    thresholds = np.full(labels.shape[1], 0.5, dtype=np.float64)
    for idx in range(labels.shape[1]):
        if len(np.unique(labels[:, idx])) != 2:
            continue
        fpr, tpr, candidates = roc_curve(labels[:, idx], scores[:, idx])
        value = float(candidates[np.argmax(tpr - fpr)])
        if np.isfinite(value):
            thresholds[idx] = value
    return thresholds


def compute_mil_metrics(labels, scores, thresholds=None) -> dict[str, object]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.ndim == 1:
        labels = labels[:, None]
    if scores.ndim == 1:
        scores = scores[:, None]
    result = compute_classification_metrics(
        labels,
        scores,
        task_type="binary" if labels.shape[1] == 1 else "multi-label",
    )
    thresholds = np.asarray(thresholds if thresholds is not None else np.full(labels.shape[1], 0.5))
    if thresholds.shape != (labels.shape[1],):
        raise ValueError(f"Expected {labels.shape[1]} thresholds, got {thresholds.shape}")
    predictions = (scores >= thresholds[None, :]).astype(int)
    valid = [idx for idx in range(labels.shape[1]) if len(np.unique(labels[:, idx])) == 2]
    f1_per_class = [f1_score(labels[:, idx], predictions[:, idx], zero_division=0) for idx in valid]
    mcc_per_class = [matthews_corrcoef(labels[:, idx], predictions[:, idx]) for idx in valid]
    balanced_accuracy_per_class = [
        balanced_accuracy_score(labels[:, idx], predictions[:, idx]) for idx in valid
    ]
    result.update(
        {
            "accuracy": float(np.mean(predictions == labels)),
            "subset_accuracy": float(np.mean(np.all(predictions == labels, axis=1))),
            "f1_macro": float(np.mean(f1_per_class)) if f1_per_class else None,
            "mcc_macro": float(np.mean(mcc_per_class)) if mcc_per_class else None,
            "balanced_accuracy_macro": (
                float(np.mean(balanced_accuracy_per_class))
                if balanced_accuracy_per_class
                else None
            ),
            "f1_per_class": f1_per_class,
            "mcc_per_class": mcc_per_class,
            "balanced_accuracy_per_class": balanced_accuracy_per_class,
            "thresholds": thresholds.tolist(),
        }
    )
    return result
