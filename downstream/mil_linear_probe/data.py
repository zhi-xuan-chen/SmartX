"""Manifest and cached-feature helpers for public case-level MIL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from downstream.linear_probe.data import (
    ID_COLUMNS,
    IMAGE_COLUMNS,
    LABEL_COLUMNS,
    SPLIT_COLUMNS,
    build_image_transform,
    coerce_label,
    load_image,
    load_yaml_config,
    parse_image_paths,
    pick_column,
    read_table,
    resolve_path,
    write_json,
)


def _validate_binary_labels(labels, context: str) -> torch.Tensor:
    tensor = torch.as_tensor(labels, dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{context} has a non-finite label")
    if not torch.all((tensor == 0) | (tensor == 1)):
        raise ValueError(f"{context} has a label outside {{0, 1}}")
    return tensor


@dataclass(frozen=True)
class MILSample:
    case_uid: str
    image_paths: tuple[str, ...]
    label: object


class MILManifestDataset(Dataset):
    """One manifest row per case, with one or more image paths per row."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        image_root: str | Path | None = None,
        split: str | None = None,
        image_column: str | None = None,
        label_column: str | None = None,
        label_columns: Sequence[str] | None = None,
        class_names: Sequence[str] | None = None,
        task_type: str = "multi-label",
        max_views: int = 6,
        transform=None,
        in_channels: int = 3,
        verify_images: bool = False,
    ) -> None:
        if task_type not in {"binary", "multi-label"}:
            raise ValueError("MIL currently supports binary or multi-label classification")
        if max_views < 1:
            raise ValueError("max_views must be at least 1")

        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root).expanduser() if image_root else None
        self.class_names = list(class_names or [])
        self.task_type = task_type
        self.max_views = int(max_views)
        self.transform = transform
        self.in_channels = int(in_channels)

        frame = read_table(self.manifest_path)
        if split:
            split_col = pick_column(frame.columns, SPLIT_COLUMNS)
            if split_col:
                frame = frame[frame[split_col].astype(str).str.lower() == split.lower()]

        image_col = image_column or pick_column(frame.columns, ("image_paths", *IMAGE_COLUMNS))
        if image_col is None:
            raise ValueError(f"No image column found in {self.manifest_path}")
        if label_columns is not None and not label_columns:
            raise ValueError("dataset.label_columns must not be empty")
        if (
            label_columns is not None
            and self.class_names
            and len(label_columns) != len(self.class_names)
        ):
            raise ValueError(
                "dataset.label_columns and dataset.class_names must have the same "
                "length"
            )
        if label_columns is not None:
            missing = [column for column in label_columns if column not in frame.columns]
            if missing:
                raise ValueError(f"Missing label columns in {self.manifest_path}: {missing}")
        else:
            label_column = label_column or pick_column(frame.columns, LABEL_COLUMNS)
            if label_column is None:
                raise ValueError(f"No label column found in {self.manifest_path}")

        id_col = pick_column(frame.columns, ID_COLUMNS)
        samples: list[MILSample] = []
        expected_labels = 1 if task_type == "binary" else len(
            label_columns or self.class_names
        ) or None
        for idx, row in frame.iterrows():
            image_paths = parse_image_paths(row[image_col], self.image_root)[: self.max_views]
            if not image_paths:
                continue
            if verify_images and any(not Path(path).exists() for path in image_paths):
                continue
            if label_columns is not None:
                label = [float(row[column]) for column in label_columns]
            else:
                label = coerce_label(
                    row[label_column],
                    class_names=self.class_names,
                    num_classes=len(self.class_names) or None,
                    task_type=task_type,
                )
            label_tensor = _validate_binary_labels(
                label, f"Case {row[id_col] if id_col else idx}"
            ).reshape(-1)
            if expected_labels is None:
                expected_labels = len(label_tensor)
            if len(label_tensor) != expected_labels:
                raise ValueError(
                    f"Expected {expected_labels} labels, got {len(label_tensor)}"
                )
            samples.append(
                MILSample(
                    case_uid=str(row[id_col]) if id_col else str(idx),
                    image_paths=image_paths,
                    label=label,
                )
            )
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        images = [load_image(path, in_channels=self.in_channels) for path in sample.image_paths]
        if self.transform is not None:
            images = [self.transform(image) for image in images]
        if not images or not isinstance(images[0], torch.Tensor):
            raise ValueError("MILManifestDataset requires a tensor image transform")

        padded = images[0].new_zeros((self.max_views, *images[0].shape))
        mask = torch.zeros(self.max_views, dtype=torch.bool)
        for view_idx, image in enumerate(images):
            padded[view_idx] = image
            mask[view_idx] = True
        label = torch.as_tensor(sample.label, dtype=torch.float32).reshape(-1)
        return padded, mask, label, sample.case_uid


def manifest_for_split(dataset_cfg: dict, split: str) -> Path:
    manifests = dataset_cfg.get("manifests", {})
    key = "valid" if split in {"val", "validation", "validate"} else split
    value = manifests.get(key)
    if value is None and key == "valid":
        value = manifests.get("val")
    if value is None:
        raise KeyError(f"No manifest configured for split '{split}'")
    path = resolve_path(value, Path.cwd())
    if path is None:
        raise ValueError(f"Empty manifest path for split '{split}'")
    return path


def build_manifest_dataset(cfg: dict, split: str) -> MILManifestDataset:
    dataset_cfg = cfg.get("dataset", {})
    feature_cfg = cfg.get("feature_extraction", {})
    image_size = int(feature_cfg.get("image_size", 768))
    in_channels = int(feature_cfg.get("in_channels", 3))
    return MILManifestDataset(
        manifest_for_split(dataset_cfg, split),
        image_root=resolve_path(dataset_cfg.get("image_root"), Path.cwd()),
        split=dataset_cfg.get("split_filter"),
        image_column=dataset_cfg.get("image_column"),
        label_column=dataset_cfg.get("label_column"),
        label_columns=dataset_cfg.get("label_columns"),
        class_names=dataset_cfg.get("class_names", []),
        task_type=dataset_cfg.get("task_type", "multi-label"),
        max_views=int(dataset_cfg.get("max_views", 6)),
        transform=build_image_transform(image_size=image_size, in_channels=in_channels),
        in_channels=in_channels,
        verify_images=bool(dataset_cfg.get("verify_images", False)),
    )


def load_cached_split(features_dir: str | Path, split: str):
    features_dir = Path(features_dir)
    key = "valid" if split in {"val", "validation", "validate"} else split
    paths = {
        "features": features_dir / f"{key}_features.pt",
        "masks": features_dir / f"{key}_masks.pt",
        "labels": features_dir / f"{key}_labels.pt",
        "samples": features_dir / f"{key}_samples.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing cached MIL files: {missing}")
    features = torch.load(paths["features"], map_location="cpu")
    masks = torch.load(paths["masks"], map_location="cpu").bool()
    labels = torch.load(paths["labels"], map_location="cpu").float()
    _validate_binary_labels(labels, f"Cached MIL split '{key}'")
    sample_ids = json.loads(paths["samples"].read_text(encoding="utf-8"))
    if not (len(features) == len(masks) == len(labels) == len(sample_ids)):
        raise ValueError(f"Cached MIL split '{key}' has inconsistent lengths")
    return features, masks, labels, [str(item) for item in sample_ids]


def resolve_class_names(
    class_names: Sequence[str] | None, num_classes: int
) -> list[str]:
    names = [str(name) for name in class_names or []]
    if not names:
        return [f"class_{idx}" for idx in range(num_classes)]
    if len(names) != num_classes:
        raise ValueError(f"Expected {num_classes} class names, got {len(names)}")
    return names


def make_cached_loader(
    features: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(features.float(), masks.bool(), labels.float()),
        batch_size=int(batch_size),
        shuffle=shuffle,
    )


__all__ = [
    "MILManifestDataset",
    "build_manifest_dataset",
    "load_cached_split",
    "load_yaml_config",
    "resolve_class_names",
    "make_cached_loader",
    "resolve_path",
    "write_json",
]
