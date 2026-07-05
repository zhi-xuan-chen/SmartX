"""Per-class val-set threshold utilities for MIL Linear Probe.

Thresholds are selected on the validation split (Youden's J: argmax(TPR-FPR))
and serialised to a sidecar JSON next to the checkpoint, so both internal
(Private A) and external (NCFY/NYFS/ZSFS/ZSFY) test scripts can reuse them
without re-thresholding on the test set itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import roc_curve


THRESHOLDS_FILENAME = "val_thresholds.json"


def compute_per_class_youden(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    fallback: float = 0.5,
) -> np.ndarray:
    """Return shape (n_classes,) per-class Youden's J thresholds on val.

    Classes without both labels in val fall back to ``fallback`` so the
    returned array is always full-sized and safe to broadcast.
    """
    if y_true.ndim != 2 or y_pred_probs.ndim != 2:
        raise ValueError("Expected 2D y_true / y_pred_probs (n_samples, n_classes)")
    n_classes = y_true.shape[1]
    thresholds = np.full(n_classes, fallback, dtype=np.float64)
    for c in range(n_classes):
        yt = y_true[:, c]
        yp = y_pred_probs[:, c]
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, thr = roc_curve(yt, yp)
        if len(thr) == 0:
            continue
        opt_idx = int(np.argmax(tpr - fpr))
        thresholds[c] = float(thr[opt_idx])
    return thresholds


def write_thresholds_json(
    path: Path | str,
    disease_cols: list[str],
    thresholds: Iterable[float],
    region: str | None = None,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "disease_cols": list(disease_cols),
        "thresholds": [float(t) for t in thresholds],
    }
    if region is not None:
        payload["region"] = region
    if extra:
        payload.update(extra)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_thresholds_json(path: Path | str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def thresholds_path_for_ckpt(ckpt_path: Path | str) -> Path:
    """Sidecar JSON lives in the same dir as the ckpt file."""
    return Path(ckpt_path).parent / THRESHOLDS_FILENAME


def load_thresholds_for_ckpt(
    ckpt_path: Path | str,
    expected_disease_cols: list[str] | None = None,
    fallback_threshold: float = 0.5,
) -> np.ndarray:
    """Load per-class thresholds and align to expected_disease_cols.

    If JSON missing or disease_cols disagree, fall back to a uniform
    ``fallback_threshold`` and warn — the caller can decide whether that
    is acceptable.
    """
    sidecar = thresholds_path_for_ckpt(ckpt_path)
    if not sidecar.exists():
        n = len(expected_disease_cols) if expected_disease_cols else 0
        print(
            f"[val_thresholds] WARN: {sidecar} not found; "
            f"falling back to uniform {fallback_threshold} for {n} classes."
        )
        return np.full(n, fallback_threshold, dtype=np.float64)
    payload = read_thresholds_json(sidecar)
    cols = payload.get("disease_cols", [])
    thr = np.asarray(payload.get("thresholds", []), dtype=np.float64)
    if expected_disease_cols is not None and list(cols) != list(expected_disease_cols):
        # Tolerate ordering mismatch by reindexing if cols are a permutation.
        if set(cols) == set(expected_disease_cols):
            idx = [cols.index(c) for c in expected_disease_cols]
            thr = thr[idx]
        else:
            print(
                f"[val_thresholds] WARN: disease_cols mismatch in {sidecar}; "
                f"falling back to uniform {fallback_threshold}."
            )
            return np.full(
                len(expected_disease_cols), fallback_threshold, dtype=np.float64
            )
    return thr


def _scalar_metric(value) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


class SaveValThresholdsCallback(pl.Callback):
    """Compute per-class Youden's J on val and write sidecar JSON.

    Writes ``val_thresholds.json`` only when the monitored validation metric
    matches ``ModelCheckpoint.best_model_score`` for the current epoch — i.e.
    the same epoch whose weights are kept as the best checkpoint. Saving
    ``last.ckpt`` at train end no longer overwrites the sidecar.

    **Callback order:** register this callback *after* ``ModelCheckpoint`` so
    ``on_validation_epoch_end`` runs once the checkpoint callback has updated
    ``best_model_score``.
    """

    def __init__(self, fallback_threshold: float = 0.5):
        super().__init__()
        self.fallback = fallback_threshold

    @staticmethod
    def _is_best_validation_epoch(trainer: pl.Trainer) -> bool:
        ckpt_cb = getattr(trainer, "checkpoint_callback", None)
        if ckpt_cb is None or not getattr(ckpt_cb, "monitor", None):
            return False
        current = _scalar_metric(trainer.callback_metrics.get(ckpt_cb.monitor))
        best = _scalar_metric(getattr(ckpt_cb, "best_model_score", None))
        if current is None or best is None:
            return False
        return abs(current - best) <= 1e-6

    @staticmethod
    def _resolve_disease_cols_and_region(
        pl_module: pl.LightningModule, n_classes: int
    ) -> tuple[list[str], str | None]:
        disease_cols = None
        hparams = getattr(pl_module, "hparams", None)
        if hparams is not None:
            disease_cols = (
                hparams.get("disease_cols")
                if isinstance(hparams, dict)
                else getattr(hparams, "disease_cols", None)
            )
        if disease_cols is None:
            disease_cols = [str(i) for i in range(n_classes)]
        region = None
        if hparams is not None:
            region_field = (
                hparams.get("region")
                if isinstance(hparams, dict)
                else getattr(hparams, "region", None)
            )
            if isinstance(region_field, (list, tuple)) and len(region_field) == 1:
                region = region_field[0]
            elif isinstance(region_field, str):
                region = region_field
        return list(disease_cols), region

    def _write_sidecar(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        thresholds: np.ndarray,
    ) -> None:
        ckpt_cb = getattr(trainer, "checkpoint_callback", None)
        ckpt_dir = getattr(ckpt_cb, "dirpath", None) if ckpt_cb is not None else None
        if not ckpt_dir:
            return
        disease_cols, region = self._resolve_disease_cols_and_region(
            pl_module, len(thresholds)
        )
        write_thresholds_json(
            Path(ckpt_dir) / THRESHOLDS_FILENAME,
            disease_cols=disease_cols,
            thresholds=thresholds,
            region=region,
            extra={
                "selection_rule": "youden_j",
                "epoch": int(trainer.current_epoch),
            },
        )

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not self._is_best_validation_epoch(trainer):
            return
        preds_list = getattr(pl_module, "val_predictions", None)
        labels_list = getattr(pl_module, "val_labels", None)
        if not preds_list or not labels_list:
            return
        try:
            y_pred = np.concatenate(preds_list, axis=0)
            y_true = np.concatenate(labels_list, axis=0)
        except ValueError:
            return
        thresholds = compute_per_class_youden(y_true, y_pred, fallback=self.fallback)
        pl_module._latest_val_thresholds = thresholds
        self._write_sidecar(trainer, pl_module, thresholds)
