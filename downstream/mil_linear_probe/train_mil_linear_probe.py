#!/usr/bin/env python3
"""Train a CLAM-style case-level classifier on cached SmartX features."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.mil_linear_probe.data import (
    load_cached_split,
    load_yaml_config,
    make_cached_loader,
    resolve_class_names,
    resolve_path,
    write_json,
)
from downstream.mil_linear_probe.metrics import compute_mil_metrics, fit_youden_thresholds
from downstream.mil_linear_probe.model import CLAMLinearProbe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SmartX MIL probe.")
    parser.add_argument(
        "--config",
        default="downstream/mil_linear_probe/configs/mil_linear_probe.example.yaml",
    )
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


def _predict(model, loader, device: torch.device):
    model.eval()
    scores = []
    labels = []
    with torch.no_grad():
        for features, masks, batch_labels in loader:
            logits, _ = model(features.to(device), masks.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(labels), np.concatenate(scores)


def _model_config(model_cfg: dict, *, feature_dim: int, num_classes: int) -> dict:
    return {
        "feature_dim": int(feature_dim),
        "num_classes": int(num_classes),
        "hidden_dim": int(model_cfg.get("hidden_dim", 512)),
        "attention_dim": int(model_cfg.get("attention_dim", 256)),
        "dropout": float(model_cfg.get("dropout", 0.25)),
        "gated": bool(model_cfg.get("gated", True)),
    }


def _monitor(metrics: dict) -> float:
    for name in ("auc_macro", "ap_macro", "accuracy"):
        value = metrics.get(name)
        if value is not None:
            return float(value)
    return -float("inf")


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    feature_cfg = cfg.get("features", {})
    training_cfg = cfg.get("training", {})
    model_cfg = cfg.get("mil_model", {})

    features_dir = resolve_path(args.features_dir or feature_cfg.get("dir"), Path.cwd())
    output_dir = resolve_path(args.output_dir or training_cfg.get("output_dir"), Path.cwd())
    if features_dir is None:
        raise ValueError("Set features.dir or pass --features-dir")
    if output_dir is None:
        raise ValueError("Set training.output_dir or pass --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_features, train_masks, train_labels, _ = load_cached_split(features_dir, "train")
    valid_features, valid_masks, valid_labels, _ = load_cached_split(features_dir, "valid")
    if train_labels.ndim == 1:
        train_labels = train_labels[:, None]
        valid_labels = valid_labels[:, None]

    batch_size = int(args.batch_size or training_cfg.get("batch_size", 4096))
    train_loader = make_cached_loader(
        train_features, train_masks, train_labels, batch_size=batch_size, shuffle=True
    )
    valid_loader = make_cached_loader(
        valid_features, valid_masks, valid_labels, batch_size=batch_size, shuffle=False
    )

    seed = int(training_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(training_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    resolved_model_cfg = _model_config(
        model_cfg,
        feature_dim=int(train_features.shape[-1]),
        num_classes=int(train_labels.shape[-1]),
    )
    model = CLAMLinearProbe(**resolved_model_cfg).to(device)
    decay_params, no_decay_params = [], []
    for name, parameter in model.named_parameters():
        (no_decay_params if name.endswith(".bias") else decay_params).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay_params,
                "weight_decay": float(training_cfg.get("weight_decay", 0.0001)),
            },
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=float(args.lr or training_cfg.get("lr", 0.001)),
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    criterion = nn.BCEWithLogitsLoss()
    max_epochs = int(args.max_epochs or training_cfg.get("max_epochs", 50))
    warmup_epochs = int(training_cfg.get("warmup_epochs", 5))
    if max_epochs < 1 or warmup_epochs < 0:
        raise ValueError("Require max_epochs >= 1 and warmup_epochs >= 0")
    total_steps = len(train_loader) * max_epochs
    warmup_steps = len(train_loader) * warmup_epochs

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    gradient_clip_val = float(training_cfg.get("gradient_clip_val", 1.0))

    best_metric = -float("inf")
    best_path = output_dir / "mil_linear_probe_best.pt"
    history = []
    metadata_path = features_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    class_names = resolve_class_names(
        metadata.get("class_names"), resolved_model_cfg["num_classes"]
    )

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for features, masks, labels in train_loader:
            features = features.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(features, masks)
            loss = criterion(logits, labels)
            loss.backward()
            if gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.detach().cpu()) * len(features)

        valid_y, valid_scores = _predict(model, valid_loader, device)
        thresholds = fit_youden_thresholds(valid_y, valid_scores)
        valid_metrics = compute_mil_metrics(valid_y, valid_scores, thresholds)
        monitor = _monitor(valid_metrics)
        row = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(1, len(train_loader.dataset)),
            "valid_metrics": valid_metrics,
        }
        history.append(row)
        if monitor > best_metric:
            best_metric = monitor
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": resolved_model_cfg,
                    "class_names": class_names,
                    "thresholds": thresholds.tolist(),
                    "training_config": training_cfg,
                    "metadata": metadata,
                },
                best_path,
            )
        print(
            f"epoch={epoch + 1} loss={row['train_loss']:.5f} "
            f"valid_auc={valid_metrics.get('auc_macro')}"
        )

    write_json(output_dir / "train_history.json", history)
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    valid_y, valid_scores = _predict(model, valid_loader, device)
    summary = {
        "best_checkpoint": str(best_path),
        "best_metric": best_metric,
        "valid_metrics": compute_mil_metrics(
            valid_y, valid_scores, checkpoint["thresholds"]
        ),
    }
    try:
        test_features, test_masks, test_labels, _ = load_cached_split(features_dir, "test")
    except FileNotFoundError:
        pass
    else:
        if test_labels.ndim == 1:
            test_labels = test_labels[:, None]
        test_loader = make_cached_loader(
            test_features, test_masks, test_labels, batch_size=batch_size, shuffle=False
        )
        test_y, test_scores = _predict(model, test_loader, device)
        summary["test_metrics"] = compute_mil_metrics(
            test_y, test_scores, checkpoint["thresholds"]
        )
    write_json(output_dir / "summary.json", summary)
    print(f"Wrote MIL training outputs to {output_dir}")


if __name__ == "__main__":
    main()
