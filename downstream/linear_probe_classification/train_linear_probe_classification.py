#!/usr/bin/env python3
"""Train a linear classifier on cached SmartX features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.linear_probe_classification.data import (
    load_feature_split,
    load_yaml_config,
    make_feature_loader,
    resolve_path,
    write_json,
)
from downstream.linear_probe_classification.metrics import compute_classification_metrics, fit_youden_thresholds
from downstream.linear_probe_classification.model import LinearProbe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SmartX linear probe.")
    parser.add_argument("--config", default="downstream/linear_probe_classification/configs/linear_probe_classification.example.yaml")
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


def _labels_for_loss(labels: torch.Tensor, task_type: str) -> torch.Tensor:
    if task_type == "multi-class":
        if labels.ndim > 1 and labels.shape[-1] > 1:
            return labels.argmax(dim=-1).long()
        return labels.reshape(-1).long()
    return labels.float()


def _loss(logits: torch.Tensor, labels: torch.Tensor, task_type: str, criterion) -> torch.Tensor:
    labels = _labels_for_loss(labels, task_type)
    if task_type == "binary":
        return criterion(logits.reshape_as(labels.float()), labels.float())
    return criterion(logits, labels)


def _scores(logits: torch.Tensor, task_type: str) -> np.ndarray:
    if task_type == "multi-class":
        return torch.softmax(logits, dim=-1).detach().cpu().numpy()
    return torch.sigmoid(logits).detach().cpu().numpy()


def _evaluate(model, loader, *, task_type: str, device: torch.device, thresholds=0.5, fit_thresholds=False) -> dict[str, object]:
    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features.to(device))
            all_scores.append(_scores(logits, task_type))
            all_labels.append(labels.cpu().numpy())
    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    if fit_thresholds and task_type != "multi-class":
        thresholds = fit_youden_thresholds(labels, scores)
    return compute_classification_metrics(labels, scores, task_type=task_type, threshold=thresholds)


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    training_cfg = cfg.get("training", {})
    feature_cfg = cfg.get("features", {})
    dataset_cfg = cfg.get("dataset", {})

    features_dir = resolve_path(args.features_dir or feature_cfg.get("dir"), Path.cwd())
    if features_dir is None:
        raise ValueError("Set features.dir or pass --features-dir")
    output_dir = resolve_path(args.output_dir or training_cfg.get("output_dir"), Path.cwd())
    if output_dir is None:
        raise ValueError("Set training.output_dir or pass --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {}
    metadata_path = features_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    task_type = dataset_cfg.get("task_type") or metadata.get("task_type", "multi-label")
    num_classes = int(
        dataset_cfg.get("num_classes")
        or metadata.get("num_classes")
        or len(dataset_cfg.get("class_names", []))
        or 1
    )
    train_features, train_labels = load_feature_split(features_dir, "train")
    valid_features, valid_labels = load_feature_split(features_dir, "valid")
    feature_dim = int(metadata.get("feature_dim") or train_features.shape[-1])

    batch_size = int(args.batch_size or training_cfg.get("batch_size", 128))
    max_epochs = int(args.max_epochs or training_cfg.get("max_epochs", 100))
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    lr = float(args.lr or training_cfg.get("lr", 0.001))
    weight_decay = float(training_cfg.get("weight_decay", 0.0001))
    dropout = float(training_cfg.get("dropout", 0.0))
    seed = int(training_cfg.get("seed", 42))
    torch.manual_seed(seed)

    train_loader = make_feature_loader(train_features, train_labels, batch_size=batch_size, shuffle=True)
    valid_loader = make_feature_loader(valid_features, valid_labels, batch_size=batch_size, shuffle=False)

    device = torch.device(training_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = LinearProbe(feature_dim=feature_dim, num_classes=num_classes, dropout=dropout).to(device)
    if task_type == "multi-class":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_metric = -float("inf")
    history = []
    best_path = output_dir / "linear_probe_classification_best.pt"
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = _loss(logits, labels, task_type, criterion)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * int(features.shape[0])

        metrics = _evaluate(model, valid_loader, task_type=task_type, device=device, fit_thresholds=True)
        monitor = metrics.get(training_cfg.get("monitor", "auc_macro"))
        if monitor is None:
            monitor = metrics.get("accuracy", 0.0)
        row = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(1, len(train_loader.dataset)),
            "valid_metrics": metrics,
        }
        history.append(row)
        if float(monitor) > best_metric:
            best_metric = float(monitor)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_dim": feature_dim,
                    "num_classes": num_classes,
                    "task_type": task_type,
                    "thresholds": metrics.get("thresholds"),
                    "class_names": dataset_cfg.get("class_names") or metadata.get("class_names", []),
                    "training_config": training_cfg,
                    "metadata": metadata,
                },
                best_path,
            )
        print(
            f"epoch={epoch + 1} loss={row['train_loss']:.5f} "
            f"monitor={float(monitor):.5f}"
        )

    write_json(output_dir / "train_history.json", history)
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    summary = {
        "best_checkpoint": str(best_path),
        "best_metric": best_metric,
        "valid_metrics": _evaluate(model, valid_loader, task_type=task_type, device=device, fit_thresholds=True),
    }
    try:
        test_features, test_labels = load_feature_split(features_dir, "test")
    except FileNotFoundError:
        pass
    else:
        test_loader = make_feature_loader(test_features, test_labels, batch_size=batch_size, shuffle=False)
        summary["test_metrics"] = _evaluate(model, test_loader, task_type=task_type, device=device, thresholds=checkpoint.get("thresholds") or 0.5)
    write_json(output_dir / "summary.json", summary)
    print(f"Wrote linear-probe outputs to {output_dir}")


if __name__ == "__main__":
    main()
