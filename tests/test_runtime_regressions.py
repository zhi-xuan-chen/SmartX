import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from downstream.modules.utils import _checkpoint_state_dict, _load_compatible_state_dict
from downstream.linear_probe_classification.metrics import compute_classification_metrics
from downstream.retrieval.datasets import load_case_manifest
from downstream.retrieval.metrics import metrics_from_similarity


def test_incomplete_checkpoint_is_rejected_before_mutating_model():
    model = nn.Linear(2, 2)
    original = model.weight.detach().clone()
    with pytest.raises(ValueError, match="Incompatible"):
        _load_compatible_state_dict(model, {"weight": torch.zeros(2, 2)})
    assert torch.equal(model.weight, original)
    with pytest.raises(ValueError, match="Incompatible"):
        _load_compatible_state_dict(model, {"weight": torch.zeros(3, 2), "bias": torch.zeros(2)})


def test_checkpoint_prefixes_only_stripped_at_start(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save({"state_dict": {"module.model.block.model.weight": torch.ones(1)}}, path)
    assert list(_checkpoint_state_dict(path)) == ["block.model.weight"]


def test_multilabel_f1_uses_fixed_threshold_and_includes_boundary():
    labels = [[0, 1], [1, 0]]
    scores = [[0.1, 0.4], [0.4, 0.1]]
    fixed = compute_classification_metrics(labels, scores, task_type="multi-label")
    assert fixed["f1_macro"] == 0.0
    configured = compute_classification_metrics(labels, scores, task_type="multi-label", threshold=0.4)
    assert configured["f1_macro"] == 1.0
    assert configured["mcc_macro"] == 1.0


def test_two_class_multiclass_auc():
    result = compute_classification_metrics([0, 1], [[0.9, 0.1], [0.1, 0.9]], task_type="multi-class")
    assert result["auc_macro"] == 1.0


@pytest.mark.parametrize("matrix", [np.full((2, 2), np.nan), np.full((2, 2), np.inf), np.empty((0, 0))])
def test_retrieval_rejects_invalid_similarity(matrix):
    with pytest.raises(ValueError, match="finite"):
        metrics_from_similarity(matrix)


def test_retrieval_filters_requested_split(tmp_path):
    path = tmp_path / "cases.csv"
    pd.DataFrame([{"case_uid": s, "split": s, "image": s + ".png", "report": "clear lungs"}
                  for s in ["train", "test"]]).to_csv(path, index=False)
    result = load_case_manifest(path, dataset="sample", split="test")
    assert [c.case_uid for c in result.cases] == ["test"]


def test_detection_matches_swin_stages(monkeypatch):
    # The real timm backbone runs on a small image; only Detectron2's base types are stubbed.
    from timm.models.swin_transformer import SwinTransformer
    layers = types.ModuleType("detectron2.layers")
    layers.ShapeSpec = lambda **kw: kw
    modeling = types.ModuleType("detectron2.modeling")
    modeling.Backbone = nn.Module
    monkeypatch.setitem(sys.modules, "detectron2.layers", layers)
    monkeypatch.setitem(sys.modules, "detectron2.modeling", modeling)
    path = Path(__file__).resolve().parents[1] / "downstream/detection/xray_vitdet/modeling/smartx_swin.py"
    spec = importlib.util.spec_from_file_location("audit_swin", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.SmartXSwinBackbone.__new__(module.SmartXSwinBackbone)
    nn.Module.__init__(model)
    model._frozen = False
    model.swin = SwinTransformer(img_size=32, patch_size=4, window_size=2,
        embed_dim=8, depths=(1, 1, 1, 1), num_heads=(1, 2, 4, 8), num_classes=0)
    outputs = model(torch.randn(2, 3, 32, 32))
    assert [tuple(v.shape) for v in outputs.values()] == [(2, 8, 8, 8), (2, 16, 4, 4), (2, 32, 2, 2), (2, 64, 1, 1)]
    sum(v.square().mean() for v in outputs.values()).backward()
    assert all(block.attn.qkv.weight.grad is not None for stage in model.swin.layers[:3] for block in stage.blocks)
    assert all(block.attn.qkv.weight.grad is None for block in model.swin.layers[3].blocks)


@pytest.mark.parametrize("label", ["[-1, 0]", "[1, 2]", "[0.5, 1]", "[1]"])
def test_manifest_rejects_invalid_multilabel_values(tmp_path, label):
    from downstream.linear_probe_classification.data import ClassificationManifestDataset
    manifest = tmp_path / "labels.csv"
    pd.DataFrame([{"image": "a.png", "labels": label}]).to_csv(manifest, index=False)
    with pytest.raises(ValueError):
        ClassificationManifestDataset(manifest, class_names=["a", "b"])


def test_16bit_and_first_slice_images(tmp_path):
    from PIL import Image
    from downstream.modules.images import load_image, array_to_pil
    pixels = np.array([[0, 1000], [2000, 4000]], dtype=np.uint16)
    path = tmp_path / "image.png"
    Image.fromarray(pixels).save(path)
    result = np.asarray(load_image(path, in_channels=1))
    assert np.array_equal(result, np.array([[0, 255], [255, 255]], dtype=np.uint8))
    volume = np.stack([pixels, np.zeros_like(pixels)])
    assert np.array_equal(np.asarray(array_to_pil(volume, in_channels=1)), np.array([[0, 63], [127, 255]], dtype=np.uint8))


def test_multiframe_monochrome1_dicom(tmp_path):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian
    from downstream.modules.images import load_image
    path = tmp_path / "image.dcm"
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=bytes(128))
    ds.Rows = ds.Columns = 2
    ds.NumberOfFrames = 2
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = np.array([[[0, 100], [200, 300]], [[0, 0], [0, 0]]], dtype=np.uint16).tobytes()
    ds.save_as(path)
    image = np.asarray(load_image(path, in_channels=1))
    assert image[0, 0] == 0 and image[-1, -1] == 255


@pytest.mark.parametrize("filename", ["model_mimic_cxr.py", "model_mimic_cxr_custom.py"])
def test_report_inference_partition_and_variable_length_batches(filename):
    import ast
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[1] / "downstream/report_generation/llavarad/llava/eval" / filename
    tree = ast.parse(path.read_text())
    helpers = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
        and n.name in {"split_list", "get_chunk", "create_batches"}], type_ignores=[])
    namespace = {}
    exec(compile(helpers, str(path), "exec"), namespace)
    chunks = namespace["split_list"]([1, 2], 4)
    assert len(chunks) == 4 and sum(chunks, []) == [1, 2]
    assert namespace["get_chunk"]([], 3, 2) == []
    tokenizer = lambda text: SimpleNamespace(input_ids=text.split())
    data = [{"conversations": [{"value": text}]} for text in ["a", "a b", "b c", "c"]]
    batches = namespace["create_batches"](data, 2, False, tokenizer)
    assert sum(batches, []) == data
    assert all(len({len(x["conversations"][0]["value"].split()) for x in batch}) == 1 for batch in batches)


def test_linear_probe_checkpoint_reuses_validation_thresholds(tmp_path):
    import json
    import subprocess
    import yaml
    from downstream.linear_probe_classification.metrics import fit_youden_thresholds
    from downstream.linear_probe_classification.model import LinearProbe
    root = Path(__file__).resolve().parents[1]
    features = tmp_path / "features"
    features.mkdir()
    generator = torch.Generator().manual_seed(9)
    for split in ["train", "valid", "test"]:
        torch.save(torch.randn(8, 3, generator=generator), features / f"{split}_features.pt")
        labels = torch.tensor([[i % 2, (i // 2) % 2] for i in range(8)]).float()
        torch.save(labels, features / f"{split}_labels.pt")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"features": {"dir": str(features)},
        "dataset": {"task_type": "multi-label", "num_classes": 2},
        "training": {"output_dir": str(tmp_path / "run"), "max_epochs": 2, "batch_size": 4, "device": "cpu"},
        "evaluation": {"output_dir": str(tmp_path / "eval"), "device": "cpu"}}))
    subprocess.run([sys.executable, "downstream/linear_probe_classification/train_linear_probe_classification.py", "--config", str(config)], cwd=root, check=True)
    checkpoint = tmp_path / "run/linear_probe_classification_best.pt"
    state = torch.load(checkpoint, weights_only=True)
    model = LinearProbe(3, 2)
    model.load_state_dict(state["model_state_dict"])
    with torch.no_grad():
        scores = model(torch.load(features / "valid_features.pt", weights_only=True)).sigmoid().numpy()
    expected = fit_youden_thresholds(torch.load(features / "valid_labels.pt", weights_only=True).numpy(), scores)
    assert np.allclose(state["thresholds"], expected)
    subprocess.run([sys.executable, "downstream/linear_probe_classification/test_linear_probe_classification.py", "--config", str(config), "--checkpoint", str(checkpoint)], cwd=root, check=True)
    result = json.loads((tmp_path / "eval/metrics.json").read_text())
    assert result["metrics"]["thresholds"] == state["thresholds"]


def test_experiment_rank_receives_exact_version(tmp_path, monkeypatch):
    from pretrain.utils import experiment_utils as experiment
    monkeypatch.setattr(experiment.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(experiment.dist, "get_rank", lambda: 1)
    def broadcast(selected, src):
        assert src == 0
        selected[0] = 7
    monkeypatch.setattr(experiment.dist, "broadcast_object_list", broadcast)
    result = experiment.setup_experiment("run", {}, base_dir=tmp_path / "nested/base")
    assert result["version"] == 7
    assert result["experiment_dir"] == tmp_path / "nested/base/run/version_7"


def test_snapshot_does_not_copy_itself(tmp_path):
    from pretrain.utils.experiment_utils import _save_code_snapshot
    destination = tmp_path / "code_snapshot"
    destination.mkdir()
    _save_code_snapshot(destination)
    assert (destination / "pretrain/run.py").is_file()
    assert not list(destination.rglob("code_snapshot/code_snapshot"))


def test_per_class_metrics_preserve_positions_for_constant_labels():
    result = compute_classification_metrics([[0, 0], [0, 1]], [[0.1, 0.1], [0.2, 0.9]], task_type="multi-label")
    assert result["auc_per_class"] == [None, 1.0]
    assert result["f1_per_class"] == [None, 1.0]
    from downstream.examination_level_classification.metrics import compute_mil_metrics
    result = compute_mil_metrics([[0, 0], [0, 1]], [[0.1, 0.1], [0.2, 0.9]])
    assert result["balanced_accuracy_per_class"] == [None, 1.0]


def test_mpt_generation_cache_compatibility(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "downstream/report_generation/llavarad"))
    from llava.model.language_model.mpt.configuration_mpt import MPTConfig
    from llava.model.language_model.mpt.modeling_mpt import MPTForCausalLM
    config = MPTConfig(d_model=16, n_heads=2, n_layers=1, expansion_ratio=2,
        max_seq_len=16, vocab_size=32, init_device="mixed", attn_config={"attn_impl": "torch"},
        bos_token_id=1, eos_token_id=2, pad_token_id=0)
    model = MPTForCausalLM(config).eval()
    with torch.no_grad():
        result = model.generate(torch.tensor([[1, 3, 4]]), min_new_tokens=2,
                                max_new_tokens=2, use_cache=True, do_sample=False)
    assert result.shape == (1, 5)
