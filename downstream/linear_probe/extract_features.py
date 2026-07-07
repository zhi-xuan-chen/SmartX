#!/usr/bin/env python3
"""Extract SmartX image embeddings for linear probing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.linear_probe.data import (
    ClassificationManifestDataset,
    build_image_transform,
    load_yaml_config,
    optional_path_value,
    resolve_path,
)
from downstream.modules import create_clip_style_model_global


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SmartX features.")
    parser.add_argument("--config", default="downstream/linear_probe/configs/linear_probe.example.yaml")
    parser.add_argument("--checkpoint", default=None, help="Optional local SmartX checkpoint.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _model_config(cfg: dict, checkpoint_override: str | None) -> dict:
    model_cfg = cfg.get("model", {})
    checkpoint = optional_path_value(checkpoint_override or model_cfg.get("checkpoint"))
    return {
        "model_name": model_cfg.get("name", "smartx_large"),
        "checkpoint": checkpoint,
        "repo_id": model_cfg.get("repo_id"),
        "filename": model_cfg.get("filename"),
        "text_encoder_name": model_cfg.get("text_encoder_name", "pubmedbert"),
        "output_dim": model_cfg.get("output_dim", 512),
    }


def _split_manifest(dataset_cfg: dict, split: str) -> str:
    manifests = dataset_cfg.get("manifests", {})
    if split in manifests:
        return manifests[split]
    if split == "valid" and "val" in manifests:
        return manifests["val"]
    raise KeyError(f"No manifest configured for split '{split}'")


def _make_dataset(cfg: dict, split: str):
    dataset_cfg = cfg.get("dataset", {})
    feature_cfg = cfg.get("feature_extraction", {})
    image_size = int(feature_cfg.get("image_size", dataset_cfg.get("image_size", 768)))
    in_channels = int(feature_cfg.get("in_channels", dataset_cfg.get("in_channels", 3)))
    transform = build_image_transform(image_size=image_size, in_channels=in_channels)
    return ClassificationManifestDataset(
        resolve_path(_split_manifest(dataset_cfg, split), Path.cwd()),
        image_root=resolve_path(dataset_cfg.get("image_root"), Path.cwd()),
        split=dataset_cfg.get("split_filter"),
        image_column=dataset_cfg.get("image_column"),
        label_column=dataset_cfg.get("label_column"),
        label_columns=dataset_cfg.get("label_columns"),
        class_names=dataset_cfg.get("class_names", []),
        task_type=dataset_cfg.get("task_type", "multi-label"),
        transform=transform,
        in_channels=in_channels,
        verify_images=bool(dataset_cfg.get("verify_images", False)),
    )


def _batch_features(vision_model, images: torch.Tensor) -> torch.Tensor:
    output = vision_model(images)
    features = getattr(output, "pooler_output", output)
    if isinstance(features, (tuple, list)):
        features = features[0]
    return features


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    feature_cfg = cfg.get("feature_extraction", {})
    dataset_cfg = cfg.get("dataset", {})

    splits = args.splits or feature_cfg.get("splits") or ["train", "valid", "test"]
    splits = ["valid" if split in {"val", "validation", "validate"} else split for split in splits]
    output_dir = resolve_path(args.output_dir or feature_cfg.get("output_dir"), Path.cwd())
    if output_dir is None:
        raise ValueError("Set feature_extraction.output_dir or pass --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = [output_dir / f"{split}_features.pt" for split in splits]
    if any(path.exists() for path in existing) and not args.force:
        raise FileExistsError(
            f"Cached features already exist under {output_dir}; pass --force to overwrite."
        )

    device = torch.device(feature_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    vision_model, _, _, model_info = create_clip_style_model_global(
        _model_config(cfg, args.checkpoint)
    )
    vision_model.to(device)
    vision_model.eval()

    feature_dim = None
    batch_size = int(feature_cfg.get("batch_size", 16))
    num_workers = int(feature_cfg.get("num_workers", 0))
    loader_kwargs = {"batch_size": batch_size, "shuffle": False, "num_workers": num_workers}
    if num_workers > 0 and feature_cfg.get("prefetch_factor") is not None:
        loader_kwargs["prefetch_factor"] = int(feature_cfg["prefetch_factor"])

    with torch.no_grad():
        for split in splits:
            dataset = _make_dataset(cfg, split)
            if len(dataset) == 0:
                raise ValueError(f"Split '{split}' has no samples.")
            dataloader = DataLoader(dataset, **loader_kwargs)
            all_features = []
            all_labels = []
            sample_ids = []
            for images, labels, ids in tqdm(dataloader, desc=f"Extracting {split}"):
                images = images.to(device, non_blocking=device.type == "cuda")
                features = _batch_features(vision_model, images).detach().cpu()
                all_features.append(features)
                all_labels.append(labels.cpu())
                sample_ids.extend(str(item) for item in ids)

            features_tensor = torch.cat(all_features, dim=0)
            labels_tensor = torch.cat(all_labels, dim=0)
            feature_dim = int(features_tensor.shape[-1])
            torch.save(features_tensor, output_dir / f"{split}_features.pt")
            torch.save(labels_tensor, output_dir / f"{split}_labels.pt")
            (output_dir / f"{split}_samples.json").write_text(
                json.dumps(sample_ids, indent=2),
                encoding="utf-8",
            )

    metadata = {
        "dataset_name": dataset_cfg.get("name", "custom"),
        "task_type": dataset_cfg.get("task_type", "multi-label"),
        "class_names": dataset_cfg.get("class_names", []),
        "num_classes": int(dataset_cfg.get("num_classes") or len(dataset_cfg.get("class_names", [])) or 1),
        "model": model_info,
        "feature_dim": feature_dim,
        "splits": splits,
        "feature_format": "{split}_features.pt, {split}_labels.pt",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote cached features to {output_dir}")


if __name__ == "__main__":
    main()
