from pathlib import Path


def test_release_directories_exist():
    root = Path(__file__).resolve().parents[1]
    for path in [
        "docs/code_availability.md",
        "docs/data_availability.md",
        "docs/model_zoo.md",
        "checkpoints/README.md",
        "demo/README.md",
        "pretrain/run.py",
        "pretrain/configs/topk_ark_base.example.yaml",
        "downstream/modules",
        "downstream/linear_probe/README.md",
        "downstream/retrieval/README.md",
    ]:
        assert (root / path).exists(), path
