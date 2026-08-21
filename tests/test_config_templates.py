from pathlib import Path

import pytest
import yaml
from pretrain.utils import load_config


def test_yaml_templates_load():
    root = Path(__file__).resolve().parents[1]
    for path in [
        "pretrain/configs/topk_ark_base.example.yaml",
        "downstream/linear_probe/configs/linear_probe.example.yaml",
        "downstream/mil_linear_probe/configs/mil_linear_probe.example.yaml",
        "downstream/zeroshot/configs/zeroshot.example.yaml",
        "downstream/retrieval/configs/retrieval.example.yaml",
        "downstream/detection/configs/detection.example.yaml",
    ]:
        with (root / path).open() as f:
            assert yaml.safe_load(f) is not None

def test_pretrain_config_expands_environment(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("SMARTX_DATA_ROOT", "/datasets")
    monkeypatch.delenv("SMARTX_OUTPUT_DIR", raising=False)
    config = load_config(root / "pretrain/configs/topk_ark_base.example.yaml")

    assert config["base_dir"] == "experiments"
    assert config["chexpert_data_dir"] == "/datasets/CheXpert-Plus"
    assert config["mimic_data_dir"] == "/datasets/MIMIC-CXR"


def test_pretrain_config_reports_missing_environment(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.delenv("SMARTX_DATA_ROOT", raising=False)

    with pytest.raises(ValueError, match="SMARTX_DATA_ROOT"):
        load_config(root / "pretrain/configs/topk_ark_base.example.yaml")


def test_requirement_files_pin_validated_packages():
    root = Path(__file__).resolve().parents[1]
    for relative_path in [
        "requirements.txt",
        "downstream/detection/requirements-optional.txt",
        "downstream/report_generation/requirements-optional.txt",
    ]:
        lines = (root / relative_path).read_text().splitlines()
        requirements = [
            line for line in lines if line.strip() and not line.startswith("#")
        ]
        assert requirements
        assert all("==" in requirement for requirement in requirements)
