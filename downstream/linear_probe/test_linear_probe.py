#!/usr/bin/env python3
"""Evaluate a trained linear probe on cached SmartX features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.linear_probe.data import (
    concatenate_splits,
    load_yaml_config,
    make_feature_loader,
    resolve_path,
    write_json,
)
from downstream.linear_probe.metrics import compute_classification_metrics
from downstream.linear_probe.model import LinearProbe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a SmartX linear probe.")
    parser.add_argument("--config", default="downstream/linear_probe/configs/linear_probe.example.yaml")
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def _scores(logits: torch.Tensor, task_type: str) -> np.ndarray:
    if task_type == "multi-class":
        return torch.softmax(logits, dim=-1).detach().cpu().numpy()
    return torch.sigmoid(logits).detach().cpu().numpy()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    feature_cfg = cfg.get("features", {})
    eval_cfg = cfg.get("evaluation", {})
    features_dir = resolve_path(args.features_dir or feature_cfg.get("dir"), Path.cwd())
    output_dir = resolve_path(args.output_dir or eval_cfg.get("output_dir"), Path.cwd())
    if features_dir is None:
        raise ValueError("Set features.dir or pass --features-dir")
    if output_dir is None:
        raise ValueError("Set evaluation.output_dir or pass --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(resolve_path(args.checkpoint, Path.cwd()), map_location="cpu")
    task_type = checkpoint["task_type"]
    model = LinearProbe(
        feature_dim=int(checkpoint["feature_dim"]),
        num_classes=int(checkpoint["num_classes"]),
        dropout=0.0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    splits = args.splits or eval_cfg.get("splits") or ["test"]
    features, labels, used_splits = concatenate_splits(features_dir, splits)
    batch_size = int(args.batch_size or eval_cfg.get("batch_size", 256))
    loader = make_feature_loader(features, labels, batch_size=batch_size, shuffle=False)

    device = torch.device(eval_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()
    all_scores = []
    with torch.no_grad():
        for batch_features, _ in loader:
            all_scores.append(_scores(model(batch_features.to(device)), task_type))
    scores = np.concatenate(all_scores, axis=0)
    label_array = labels.cpu().numpy()
    metrics = compute_classification_metrics(label_array, scores, task_type=task_type)

    write_json(
        output_dir / "metrics.json",
        {
            "splits": used_splits,
            "checkpoint": str(resolve_path(args.checkpoint, Path.cwd())),
            "metrics": metrics,
        },
    )
    score_columns = [f"score_{idx}" for idx in range(scores.shape[1] if scores.ndim > 1 else 1)]
    score_frame = pd.DataFrame(scores.reshape(len(scores), -1), columns=score_columns)
    score_frame.insert(0, "sample_index", np.arange(len(score_frame)))
    score_frame.to_csv(output_dir / "predictions.csv", index=False)
    print(f"Wrote evaluation outputs to {output_dir}")


if __name__ == "__main__":
    main()
