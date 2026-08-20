import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from downstream.mil_linear_probe.data import (
    MILManifestDataset,
    load_cached_split,
    resolve_class_names,
)
from downstream.mil_linear_probe.metrics import (
    compute_mil_metrics,
    fit_youden_thresholds,
)
from downstream.mil_linear_probe.model import CLAMLinearProbe


def test_manifest_and_masked_attention(tmp_path):
    manifest = tmp_path / "cases.csv"
    pd.DataFrame(
        [
            {
                "case_uid": "case-1",
                "image_paths": "a.png;b.png;c.png",
                "finding_a": 1,
                "finding_b": 0,
            }
        ]
    ).to_csv(manifest, index=False)
    dataset = MILManifestDataset(
        manifest,
        label_columns=["finding_a", "finding_b"],
        max_views=2,
        transform=lambda image: image,
    )
    assert dataset.samples[0].case_uid == "case-1"
    assert dataset.samples[0].image_paths == ("a.png", "b.png")

    model = CLAMLinearProbe(
        feature_dim=4,
        num_classes=2,
        hidden_dim=8,
        attention_dim=4,
        dropout=0.0,
    ).eval()
    features = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    logits, attention = model(features, mask)
    assert logits.shape == (2, 2)
    assert attention.shape == (2, 2, 3)
    assert torch.all(attention.masked_select(~mask[:, None, :]) == 0)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(2, 2))


def test_manifest_rejects_invalid_labels(tmp_path):
    manifest = tmp_path / "invalid.csv"
    pd.DataFrame(
        [{"case_uid": "case-1", "image_paths": "a.png", "a": 1, "b": np.nan}]
    ).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="non-finite label"):
        MILManifestDataset(
            manifest,
            label_columns=["a", "b"],
            transform=lambda image: image,
        )

    pd.DataFrame(
        [{"case_uid": "case-1", "image_paths": "a.png", "a": 1, "b": 2}]
    ).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="outside"):
        MILManifestDataset(
            manifest,
            label_columns=["a", "b"],
            transform=lambda image: image,
        )

    with pytest.raises(ValueError, match="class names"):
        resolve_class_names(["finding_a"], 2)

    features_dir = tmp_path / "features"
    features_dir.mkdir()
    torch.save(torch.zeros(1, 1, 2), features_dir / "train_features.pt")
    torch.save(torch.ones(1, 1, dtype=torch.bool), features_dir / "train_masks.pt")
    torch.save(torch.tensor([[1.0, float("nan")]]), features_dir / "train_labels.pt")
    (features_dir / "train_samples.json").write_text('["case-1"]')
    with pytest.raises(ValueError, match="non-finite label"):
        load_cached_split(features_dir, "train")


def test_balanced_accuracy_uses_fixed_thresholds():
    metrics = compute_mil_metrics(
        [[0, 1], [1, 0]],
        [[0.1, 0.9], [0.9, 0.1]],
        [0.5, 0.5],
    )
    assert metrics["balanced_accuracy_macro"] == 1.0
    assert metrics["balanced_accuracy_per_class"] == [1.0, 1.0]


def test_train_and_evaluate_cached_bags(tmp_path):
    root = Path(__file__).resolve().parents[1]
    features_dir = tmp_path / "features"
    output_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    features_dir.mkdir()
    generator = torch.Generator().manual_seed(7)

    for split, count in (("train", 16), ("valid", 8), ("test", 8)):
        labels = torch.tensor([[idx % 2, (idx // 2) % 2] for idx in range(count)]).float()
        features = torch.randn(count, 3, 6, generator=generator)
        features[:, 0, :2] += labels
        masks = torch.tensor([[True, True, False]] * count)
        features[:, 2] = 0
        torch.save(features, features_dir / f"{split}_features.pt")
        torch.save(masks, features_dir / f"{split}_masks.pt")
        torch.save(labels, features_dir / f"{split}_labels.pt")
        (features_dir / f"{split}_samples.json").write_text(
            json.dumps([f"{split}-{idx}" for idx in range(count)])
        )
    (features_dir / "metadata.json").write_text(
        json.dumps({"class_names": ["finding_a", "finding_b"]})
    )

    config = tmp_path / "mil.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "features": {"dir": str(features_dir)},
                "mil_model": {
                    "hidden_dim": 8,
                    "attention_dim": 4,
                    "dropout": 0.0,
                    "gated": True,
                },
                "training": {
                    "output_dir": str(output_dir),
                    "batch_size": 4,
                    "max_epochs": 2,
                    "lr": 0.01,
                    "device": "cpu",
                    "seed": 7,
                    "warmup_epochs": 1,
                    "gradient_clip_val": 1.0,
                },
                "evaluation": {
                    "output_dir": str(eval_dir),
                    "split": "test",
                    "batch_size": 4,
                    "device": "cpu",
                    "write_attention": True,
                },
            }
        )
    )
    subprocess.run(
        [
            sys.executable,
            "downstream/mil_linear_probe/train_mil_linear_probe.py",
            "--config",
            str(config),
        ],
        cwd=root,
        check=True,
    )
    checkpoint = output_dir / "mil_linear_probe_best.pt"
    assert checkpoint.exists()
    subprocess.run(
        [
            sys.executable,
            "downstream/mil_linear_probe/test_mil_linear_probe.py",
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
        ],
        cwd=root,
        check=True,
    )
    assert (eval_dir / "metrics.json").exists()
    assert (eval_dir / "predictions.csv").exists()
    assert np.load(eval_dir / "attention.npy").shape == (8, 2, 3)
    payload = torch.load(checkpoint, map_location="cpu")
    assert len(payload["thresholds"]) == 2
    assert fit_youden_thresholds([0, 1], [0.1, 0.9]).shape == (1,)
