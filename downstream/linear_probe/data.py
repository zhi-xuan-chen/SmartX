"""Data and cached-feature helpers for linear probing."""

from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, TensorDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_COLUMNS = (
    "image_path",
    "image",
    "path",
    "filename",
    "file_name",
    "Path",
    "PngPath",
)
LABEL_COLUMNS = ("labels", "label", "target", "targets", "class", "class_name")
SPLIT_COLUMNS = ("split", "Split", "partition", "subset")
ID_COLUMNS = ("sample_id", "case_uid", "uid", "id", "study_id")


def load_yaml_config(path: str | Path) -> dict:
    import yaml

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a mapping: {config_path}")
    return _expand_env(payload)


def _expand_env(value):
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_env(item) for item in value)
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def resolve_path(value: str | Path | None, base_dir: str | Path | None = None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path


def optional_path_value(value: str | Path | None) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text or "$" in text:
        return None
    return text


def write_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("samples", "cases", "items", "data"):
                if isinstance(payload.get(key), list):
                    return pd.DataFrame(payload[key])
        return pd.DataFrame(payload)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def pick_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    column_list = list(columns)
    column_set = set(column_list)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    lower_to_original = {column.lower(): column for column in column_list}
    for candidate in candidates:
        match = lower_to_original.get(candidate.lower())
        if match:
            return match
    return None


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def parse_sequence(value: object) -> list[object]:
    if _is_missing(value):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)

    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except Exception:
                continue
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
    for separator in (";", "|"):
        if separator in text:
            return [item.strip() for item in text.split(separator) if item.strip()]
    if "," in text:
        parts = [item.strip() for item in text.split(",") if item.strip()]
        if len(parts) > 1:
            return parts
    return [value]


def parse_image_paths(value: object, image_root: str | Path | None = None) -> tuple[str, ...]:
    paths = []
    root = Path(image_root).expanduser() if image_root else None
    for item in parse_sequence(value):
        text = str(item).strip()
        if not text:
            continue
        path = Path(text).expanduser()
        if root is not None and not path.is_absolute():
            path = root / path
        paths.append(str(path))
    return tuple(paths)


def _label_index(value: object, class_names: Sequence[str] | None) -> int:
    if isinstance(value, str):
        text = value.strip()
        if class_names and text in class_names:
            return int(class_names.index(text))
        return int(float(text))
    return int(value)


def coerce_label(
    value: object,
    *,
    class_names: Sequence[str] | None = None,
    num_classes: int | None = None,
    task_type: str = "multi-label",
):
    values = parse_sequence(value)
    if not values:
        raise ValueError("Empty label value")

    if task_type == "multi-class" and len(values) == 1:
        return _label_index(values[0], class_names)

    if len(values) == 1 and num_classes == 1:
        return [float(values[0])]

    if class_names and any(isinstance(item, str) and not str(item).replace(".", "", 1).isdigit() for item in values):
        vector = np.zeros(len(class_names), dtype=np.float32)
        for item in values:
            vector[_label_index(item, class_names)] = 1.0
        return vector.tolist()

    return [float(item) for item in values]


def _array_to_pil(array: np.ndarray, in_channels: int = 3) -> Image.Image:
    array = np.asarray(array)
    if array.ndim == 3:
        array = array[0]
    array = array.astype(np.float32)
    amin = float(np.nanmin(array))
    amax = float(np.nanmax(array))
    if amax > amin:
        array = (array - amin) / (amax - amin)
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("L" if in_channels == 1 else "RGB")


def load_image(path: str | Path, in_channels: int = 3) -> Image.Image:
    path = Path(path)
    suffix = path.suffix.lower()
    path_text = str(path).lower()
    if suffix in {".dcm", ".dicom"}:
        import pydicom

        dataset = pydicom.dcmread(str(path), force=True)
        return _array_to_pil(dataset.pixel_array, in_channels=in_channels)
    if suffix in {".mha", ".mhd", ".nii"} or path_text.endswith(".nii.gz"):
        import SimpleITK as sitk

        image = sitk.ReadImage(str(path))
        return _array_to_pil(sitk.GetArrayFromImage(image), in_channels=in_channels)
    return Image.open(path).convert("L" if in_channels == 1 else "RGB")


def build_image_transform(
    image_size: int = 768,
    *,
    in_channels: int = 3,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
):
    from torchvision import transforms

    if mean is None:
        mean = (0.485, 0.456, 0.406) if in_channels == 3 else (0.5,)
    if std is None:
        std = (0.229, 0.224, 0.225) if in_channels == 3 else (0.5,)
    return transforms.Compose(
        [
            transforms.Resize((int(image_size), int(image_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(mean), std=tuple(std)),
        ]
    )


@dataclass(frozen=True)
class ClassificationSample:
    image_path: str
    label: object
    sample_id: str


class ClassificationManifestDataset(Dataset):
    """Manifest-backed image classification dataset for linear probes."""

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
        transform=None,
        in_channels: int = 3,
        verify_images: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root).expanduser() if image_root else None
        self.class_names = list(class_names or [])
        self.task_type = task_type
        self.transform = transform
        self.in_channels = in_channels

        frame = read_table(self.manifest_path)
        if split:
            split_col = pick_column(frame.columns, SPLIT_COLUMNS)
            if split_col:
                frame = frame[frame[split_col].astype(str).str.lower() == split.lower()]

        image_col = image_column or pick_column(frame.columns, IMAGE_COLUMNS)
        if image_col is None:
            raise ValueError(f"No image column found in {self.manifest_path}")

        if label_columns is not None:
            missing = [column for column in label_columns if column not in frame.columns]
            if missing:
                raise ValueError(f"Missing label columns in {self.manifest_path}: {missing}")
        else:
            label_column = label_column or pick_column(frame.columns, LABEL_COLUMNS)
            if label_column is None:
                raise ValueError(f"No label column found in {self.manifest_path}")

        id_col = pick_column(frame.columns, ID_COLUMNS)
        samples: list[ClassificationSample] = []
        for idx, row in frame.iterrows():
            image_paths = parse_image_paths(row[image_col], self.image_root)
            if not image_paths:
                continue
            image_path = image_paths[0]
            if verify_images and not Path(image_path).exists():
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
            sample_id = str(row[id_col]) if id_col else str(idx)
            samples.append(ClassificationSample(image_path, label, sample_id))
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = load_image(sample.image_path, in_channels=self.in_channels)
        if self.transform is not None:
            image = self.transform(image)
        label = torch.as_tensor(sample.label)
        if self.task_type != "multi-class":
            label = label.float()
        return image, label, sample.sample_id


def normalize_split(split: str) -> str:
    split = split.lower()
    if split in {"val", "validation", "validate"}:
        return "valid"
    return split


def load_feature_split(features_dir: str | Path, split: str):
    split = normalize_split(split)
    root = Path(features_dir)
    features_path = root / f"{split}_features.pt"
    labels_path = root / f"{split}_labels.pt"
    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Missing cached feature files for split '{split}' under {root}"
        )
    features = torch.load(features_path, map_location="cpu")
    labels = torch.load(labels_path, map_location="cpu")
    return features.float(), labels


def available_splits(features_dir: str | Path) -> list[str]:
    root = Path(features_dir)
    splits = []
    for path in sorted(root.glob("*_features.pt")):
        split = path.name[: -len("_features.pt")]
        if (root / f"{split}_labels.pt").exists():
            splits.append(split)
    return splits


def make_feature_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        TensorDataset(features.float(), labels),
        batch_size=int(batch_size),
        shuffle=shuffle,
    )


def concatenate_splits(features_dir: str | Path, splits: Iterable[str]):
    feature_parts = []
    label_parts = []
    used = []
    for split in splits:
        try:
            features, labels = load_feature_split(features_dir, split)
        except FileNotFoundError:
            continue
        feature_parts.append(features)
        label_parts.append(labels)
        used.append(normalize_split(split))
    if not feature_parts:
        raise FileNotFoundError(f"No requested splits were found under {features_dir}")
    return torch.cat(feature_parts), torch.cat(label_parts), used
