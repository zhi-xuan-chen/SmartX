from pathlib import Path
import importlib
import importlib.util

import numpy as np
import pytest


def test_downstream_import_smoke():
    for module_name in [
        "downstream.linear_probe.data",
        "downstream.linear_probe.metrics",
        "downstream.linear_probe.model",
        "downstream.mil_linear_probe.data",
        "downstream.mil_linear_probe.metrics",
        "downstream.mil_linear_probe.model",
        "downstream.zeroshot.data",
        "downstream.zeroshot.metrics",
        "downstream.retrieval.datasets",
        "downstream.retrieval.metrics",
        "downstream.detection.train_detection",
    ]:
        assert importlib.import_module(module_name)


def test_retrieval_metrics_perfect_diagonal():
    from downstream.retrieval.metrics import metrics_from_similarity

    metrics = metrics_from_similarity(np.eye(3), ks=(1, 2))
    assert metrics["image_to_text"]["recall@1"] == 1.0
    assert metrics["text_to_image"]["recall@1"] == 1.0
    assert metrics["image_to_text"]["mrr"] == 1.0
    assert metrics["image_to_text"]["mean_rank"] == 1.0
    assert metrics["image_to_text"]["median_rank"] == 1.0


def test_linear_probe_binary_metrics():
    from downstream.linear_probe.metrics import compute_classification_metrics

    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    metrics = compute_classification_metrics(labels, scores, task_type="binary")
    assert metrics["accuracy"] == 1.0
    assert metrics["auc_macro"] == 1.0
    assert metrics["ap_macro"] == 1.0


def test_detection_config_loads_without_detectron2():
    from downstream.detection.train_detection import load_detection_config

    root = Path(__file__).resolve().parents[1]
    cfg = load_detection_config(root / "downstream/detection/configs/detection.example.yaml")
    assert cfg["dataset"]["name"] == "xray_detection"
    assert cfg["model"]["backbone"] == "smartx_large"


def test_detection_lazy_config_skips_when_detectron2_missing():
    if importlib.util.find_spec("detectron2") is None:
        pytest.skip("Detectron2 is optional for the release test suite.")
    from downstream.detection.train_detection import build_lazy_config, load_detection_config

    root = Path(__file__).resolve().parents[1]
    cfg = load_detection_config(root / "downstream/detection/configs/detection.example.yaml")
    lazy_cfg = build_lazy_config(cfg, "test", opts=[])
    assert lazy_cfg.model.roi_heads.num_classes == 1


def test_downstream_release_hygiene():
    root = Path(__file__).resolve().parents[1]
    release_roots = [
        root / "pretrain",
        root / "downstream",
        root / "demo",
        root / "docs",
    ]
    banned = [
        "/" + "jhcnas",
        "/" + "mnt/hanoverdev",
        "/home/" + "chenzhixuan",
        "/home/" + "kyle",
        "/data/" + "kyle",
        "/" + "ssddata" + "2",
        "jhcpu",
        "zhefan.pem",
    ]
    generated_suffixes = {".pt", ".pth", ".ckpt", ".npy", ".npz", ".pyc"}
    for release_root in release_roots:
        for path in release_root.rglob("*"):
            rel = path.relative_to(release_root).as_posix()
            if "__pycache__" in path.parts:
                continue
            assert "/splits/" not in f"/{rel}/"
            if path.is_dir():
                assert path.name not in {"logs", "features", "experiments", "results"}
                continue
            assert path.suffix not in generated_suffixes
            if path.suffix in {".py", ".md", ".yaml", ".yml", ".txt"}:
                text = path.read_text(encoding="utf-8")
                lowered = text.lower()
                for pattern in banned:
                    assert pattern not in lowered, f"{pattern} found in {path}"
