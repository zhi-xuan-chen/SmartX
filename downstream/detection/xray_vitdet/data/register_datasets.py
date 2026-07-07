"""Register user-provided COCO detection datasets with Detectron2."""

from __future__ import annotations

from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances


def register_coco_splits(
    *,
    dataset_name: str,
    splits: dict,
    class_names: list[str] | None = None,
    force: bool = False,
) -> list[str]:
    registered = []
    metadata = {"thing_classes": class_names or []}
    for split, split_cfg in splits.items():
        key = f"{dataset_name}_{split}"
        annotation = Path(split_cfg["annotation"]).expanduser()
        image_root = Path(split_cfg["image_root"]).expanduser()
        if force and key in DatasetCatalog.list():
            DatasetCatalog.remove(key)
            MetadataCatalog.remove(key)
        register_coco_instances(key, metadata, str(annotation), str(image_root))
        registered.append(key)
    return registered
