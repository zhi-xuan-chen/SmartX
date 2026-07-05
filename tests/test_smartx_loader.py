from pathlib import Path

import pytest

from downstream.modules import SMARTX_HF_REPO, SMARTX_LARGE_FILENAME, resolve_smartx_checkpoint


def test_smartx_hf_defaults_match_model_zoo():
    assert SMARTX_HF_REPO == "Trusure/SmartX"
    assert SMARTX_LARGE_FILENAME == "smartx_large.ckpt"


def test_resolve_smartx_checkpoint_uses_local_path(tmp_path):
    ckpt = tmp_path / "smartx_large.ckpt"
    ckpt.write_bytes(b"placeholder")
    assert resolve_smartx_checkpoint(ckpt) == str(ckpt)


def test_resolve_smartx_checkpoint_rejects_missing_local_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_smartx_checkpoint(tmp_path / "missing.ckpt")
