from pathlib import Path


def test_downstream_task_guide_names_all_tasks():
    root = Path(__file__).resolve().parents[1]
    guide = root / "docs" / "downstream_task_guide.md"
    text = guide.read_text()
    for task in ["zero_shot_classification", "retrieval", "linear_probe_classification", "examination_level_classification", "detection"]:
        assert f"downstream/{task}/" in text
    assert "create_clip_style_model_global" in text


def test_task_readmes_point_to_guide():
    root = Path(__file__).resolve().parents[1]
    for task in ["zero_shot_classification", "retrieval", "linear_probe_classification", "examination_level_classification", "detection"]:
        text = (root / "downstream" / task / "README.md").read_text()
        assert "docs/downstream_task_guide.md" in text
        assert "create_clip_style_model_global" in text
