#!/usr/bin/env python3
"""Extract frozen SmartX features for case-level MIL."""

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

from downstream.linear_probe.data import optional_path_value
from downstream.mil_linear_probe.data import (
    build_manifest_dataset,
    load_yaml_config,
    resolve_path,
)
from downstream.modules import create_clip_style_model_global


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SmartX case-level MIL features.")
    parser.add_argument(
        "--config",
        default="downstream/mil_linear_probe/configs/mil_linear_probe.example.yaml",
    )
    parser.add_argument("--checkpoint", default=None, help="Optional local SmartX checkpoint.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _model_config(cfg: dict, checkpoint_override: str | None) -> dict:
    model_cfg = cfg.get("model", {})
    payload = {
        "model_name": model_cfg.get("name", "smartx_large"),
        "checkpoint": optional_path_value(checkpoint_override or model_cfg.get("checkpoint")),
        "repo_id": model_cfg.get("repo_id"),
        "filename": model_cfg.get("filename"),
        "text_encoder_name": model_cfg.get("text_encoder_name", "pubmedbert"),
        "output_dim": model_cfg.get("output_dim", 512),
    }
    return {key: value for key, value in payload.items() if value is not None}


def _vision_features(vision_model, images: torch.Tensor) -> torch.Tensor:
    output = vision_model(images)
    features = getattr(output, "pooler_output", output)
    if isinstance(features, (tuple, list)):
        features = features[0]
    if features.ndim != 2:
        features = features.flatten(start_dim=1)
    return features


def _extract_split(vision_model, dataset, *, batch_size: int, num_workers: int, device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    all_features = []
    all_masks = []
    all_labels = []
    sample_ids = []
    with torch.no_grad():
        for images, masks, labels, ids in tqdm(loader, desc="Encoding MIL cases"):
            images = images.to(device, non_blocking=device.type == "cuda")
            masks_device = masks.to(device)
            valid_features = _vision_features(vision_model, images[masks_device])
            batch_features = valid_features.new_zeros(
                (*masks_device.shape, valid_features.shape[-1])
            )
            batch_features[masks_device] = valid_features
            all_features.append(batch_features.cpu())
            all_masks.append(masks.bool())
            all_labels.append(labels.float())
            sample_ids.extend(str(item) for item in ids)
    return (
        torch.cat(all_features),
        torch.cat(all_masks),
        torch.cat(all_labels),
        sample_ids,
    )


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
            f"Cached MIL features already exist under {output_dir}; pass --force to overwrite."
        )

    device = torch.device(feature_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    vision_model, _, _, model_info = create_clip_style_model_global(
        _model_config(cfg, args.checkpoint)
    )
    vision_model.to(device).eval()

    feature_dim = None
    split_sizes = {}
    for split in splits:
        dataset = build_manifest_dataset(cfg, split)
        if not dataset:
            raise ValueError(f"Split '{split}' has no usable cases")
        features, masks, labels, sample_ids = _extract_split(
            vision_model,
            dataset,
            batch_size=int(feature_cfg.get("batch_size", 4)),
            num_workers=int(feature_cfg.get("num_workers", 0)),
            device=device,
        )
        feature_dim = int(features.shape[-1])
        split_sizes[split] = len(features)
        torch.save(features, output_dir / f"{split}_features.pt")
        torch.save(masks, output_dir / f"{split}_masks.pt")
        torch.save(labels, output_dir / f"{split}_labels.pt")
        (output_dir / f"{split}_samples.json").write_text(
            json.dumps(sample_ids, indent=2),
            encoding="utf-8",
        )

    metadata = {
        "dataset_name": dataset_cfg.get("name", "custom"),
        "task_type": dataset_cfg.get("task_type", "multi-label"),
        "class_names": dataset_cfg.get("class_names", []),
        "num_classes": int(
            dataset_cfg.get("num_classes")
            or len(dataset_cfg.get("class_names", []))
            or 1
        ),
        "max_views": int(dataset_cfg.get("max_views", 6)),
        "feature_dim": feature_dim,
        "splits": split_sizes,
        "model": model_info,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote cached MIL features to {output_dir}")


if __name__ == "__main__":
    main()
