from pathlib import Path


def test_downstream_task_guide_exists_and_names_pending_tasks():
    root = Path(__file__).resolve().parents[1]
    guide = root / "docs" / "downstream_task_guide.md"
    text = guide.read_text()
    for task in ["zeroshot", "retrieval", "linear_probe", "detection"]:
        assert f"downstream/{task}/" in text
    assert "create_clip_style_model_global" in text


def test_pending_task_readmes_point_to_guide():
    root = Path(__file__).resolve().parents[1]
    for task in ["zeroshot", "retrieval", "linear_probe", "detection"]:
        text = (root / "downstream" / task / "README.md").read_text()
        assert "docs/downstream_task_guide.md" in text
        assert "create_clip_style_model_global" in text
