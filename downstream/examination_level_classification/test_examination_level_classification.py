#!/usr/bin/env python3
"""Evaluate a trained CLAM-style MIL probe on cached SmartX features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.examination_level_classification.data import (
    load_cached_split,
    load_yaml_config,
    make_cached_loader,
    resolve_class_names,
    resolve_path,
    write_json,
)
from downstream.examination_level_classification.metrics import compute_mil_metrics
from downstream.examination_level_classification.model import CLAMLinearProbe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a SmartX MIL probe.")
    parser.add_argument(
        "--config",
        default="downstream/examination_level_classification/configs/examination_level_classification.example.yaml",
    )
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    feature_cfg = cfg.get("features", {})
    eval_cfg = cfg.get("evaluation", {})
    features_dir = resolve_path(args.features_dir or feature_cfg.get("dir"), Path.cwd())
    output_dir = resolve_path(args.output_dir or eval_cfg.get("output_dir"), Path.cwd())
    checkpoint_path = resolve_path(args.checkpoint, Path.cwd())
    if features_dir is None:
        raise ValueError("Set features.dir or pass --features-dir")
    if output_dir is None:
        raise ValueError("Set evaluation.output_dir or pass --output-dir")
    if checkpoint_path is None:
        raise ValueError("Checkpoint path is empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = resolve_class_names(
        checkpoint.get("class_names"), checkpoint["model_config"]["num_classes"]
    )
    model = CLAMLinearProbe(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    split = args.split or eval_cfg.get("split", "test")
    features, masks, labels, sample_ids = load_cached_split(features_dir, split)
    if labels.ndim == 1:
        labels = labels[:, None]
    loader = make_cached_loader(
        features,
        masks,
        labels,
        batch_size=int(args.batch_size or eval_cfg.get("batch_size", 64)),
        shuffle=False,
    )

    device = torch.device(eval_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    all_scores = []
    all_attention = []
    with torch.no_grad():
        for batch_features, batch_masks, _ in loader:
            logits, attention = model(batch_features.to(device), batch_masks.to(device))
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
            all_attention.append(attention.cpu().numpy())
    scores = np.concatenate(all_scores)
    attention = np.concatenate(all_attention)
    label_array = labels.numpy()
    thresholds = np.asarray(checkpoint.get("thresholds", [0.5] * scores.shape[1]))
    metrics = compute_mil_metrics(label_array, scores, thresholds)

    write_json(
        output_dir / "metrics.json",
        {
            "split": split,
            "checkpoint": str(checkpoint_path),
            "metrics": metrics,
        },
    )
    rows = {"case_uid": sample_ids}
    predictions = scores >= thresholds[None, :]
    for idx, class_name in enumerate(class_names):
        rows[f"{class_name}_label"] = label_array[:, idx]
        rows[f"{class_name}_score"] = scores[:, idx]
        rows[f"{class_name}_prediction"] = predictions[:, idx].astype(int)
    pd.DataFrame(rows).to_csv(output_dir / "predictions.csv", index=False)
    if bool(eval_cfg.get("write_attention", True)):
        np.save(output_dir / "attention.npy", attention.astype(np.float32, copy=False))
    print(f"Wrote MIL evaluation outputs to {output_dir}")


if __name__ == "__main__":
    main()
