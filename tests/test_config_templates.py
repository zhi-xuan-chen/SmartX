from pathlib import Path

import yaml


def test_yaml_templates_load():
    root = Path(__file__).resolve().parents[1]
    for path in [
        "pretrain/configs/topk_ark_base.example.yaml",
        "demo/configs/extract_features.yaml",
        "demo/configs/zero_shot.yaml",
        "demo/configs/retrieval.yaml",
        "downstream/linear_probe/configs/linear_probe.example.yaml",
        "downstream/zeroshot/configs/zeroshot.example.yaml",
        "downstream/retrieval/configs/retrieval.example.yaml",
        "downstream/detection/configs/detection.example.yaml",
    ]:
        with (root / path).open() as f:
            assert yaml.safe_load(f) is not None
