"""Shared manifest and image helpers for zero-shot classification."""
from downstream.linear_probe_classification.data import (
    ClassificationManifestDataset, ClassificationSample,
    IMAGE_COLUMNS, LABEL_COLUMNS, SPLIT_COLUMNS, ID_COLUMNS,
    load_yaml_config, resolve_path, optional_path_value, write_json,
    read_table, pick_column, parse_sequence, parse_image_paths, coerce_label,
    load_image, build_image_transform,
)
