from pathlib import Path


def test_release_checklist_tracks_known_open_work():
    root = Path(__file__).resolve().parents[1]
    text = (root / "docs" / "release_checklist.md").read_text()
    for item in ["demo/", "zeroshot", "retrieval", "linear_probe", "detection", "CITATION.cff"]:
        assert item in text
    assert "create_clip_style_model_global" in text
    assert "Final Audit" in text
