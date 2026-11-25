from pathlib import Path

import pytest
import yaml
from pretrain.utils import load_config


def test_yaml_templates_load():
    root = Path(__file__).resolve().parents[1]
    for path in [
        "pretrain/configs/smartx_large.example.yaml",
        "downstream/linear_probe_classification/configs/linear_probe_classification.example.yaml",
        "downstream/examination_level_classification/configs/examination_level_classification.example.yaml",
        "downstream/zero_shot_classification/configs/zero_shot_classification.example.yaml",
        "downstream/retrieval/configs/retrieval.example.yaml",
        "downstream/detection/configs/detection.example.yaml",
    ]:
        with (root / path).open() as f:
            assert yaml.safe_load(f) is not None

def test_pretrain_config_expands_environment(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("SMARTX_DATA_ROOT", "/datasets")
    monkeypatch.delenv("SMARTX_OUTPUT_DIR", raising=False)
    config = load_config(root / "pretrain/configs/smartx_large.example.yaml")

    assert config["base_dir"] == "experiments"
    assert config["chexpert_data_dir"] == "/datasets/CheXpert-Plus"
    assert config["mimic_data_dir"] == "/datasets/MIMIC-CXR"


def test_pretrain_config_reports_missing_environment(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.delenv("SMARTX_DATA_ROOT", raising=False)

    with pytest.raises(ValueError, match="SMARTX_DATA_ROOT"):
        load_config(root / "pretrain/configs/smartx_large.example.yaml")


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


def test_public_pretraining_configuration(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv('SMARTX_DATA_ROOT', '/datasets')
    monkeypatch.setenv('SMARTX_CHECKPOINT', '/weights/smartx_large.ckpt')
    path = root / 'pretrain/configs/smartx_large.example.yaml'
    config = load_config(path)
    assert config['model'] == config['vision_encoder_name'] == 'smartx_large'
    assert config['smartx_checkpoint'] == '/weights/smartx_large.ckpt'
    assert (config['img_size'], config['vision_encoder_dim'], config['batch_size']) == (768, 1536, 72)
    assert config['text_lr'] == 7e-4
    assert config['local_alignment_transition_epochs'] == 1
    for yaml_path in root.rglob('*.example.yaml'):
        content = yaml_path.read_text().lower()
        assert not any(name in content or name in yaml_path.name for name in ('ark', 'arc_large', 'topk'))

    assert [p.name for p in (root / 'pretrain/configs').glob('*.yaml')] == ['smartx_large.example.yaml']
