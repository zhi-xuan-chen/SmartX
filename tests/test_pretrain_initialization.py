import importlib.util
from pathlib import Path
import sys
import types
import pytest
import torch

source = Path(__file__).resolve().parents[1] / "pretrain/modules/initialization.py"
spec = importlib.util.spec_from_file_location("smartx_initialization", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_full_initialization_update_and_reload(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    path = tmp_path / "smartx.ckpt"
    torch.save({"state_dict": {"model." + k: v for k, v in initial.items()}}, path)
    target = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    module.initialize_from_smartx(target, path)
    assert all(torch.equal(target.state_dict()[k], v) for k, v in initial.items())
    optimizer = torch.optim.AdamW(target.parameters(), lr=0.01)
    target(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    assert any(not torch.equal(target.state_dict()[k], v) for k, v in initial.items())
    torch.save({"state_dict": {"model." + k: v for k, v in target.state_dict().items()}}, path)
    module.initialize_from_smartx(model, path)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in target.state_dict().items())


@pytest.mark.parametrize("kind", ["missing", "unexpected", "shape"])
def test_rejects_incompatible_weights(tmp_path, kind):
    model = torch.nn.Linear(3, 2)
    state = model.state_dict()
    if kind == "missing": state.pop("bias")
    if kind == "unexpected": state["extra"] = torch.zeros(1)
    if kind == "shape": state["weight"] = torch.zeros(1)
    path = tmp_path / "bad.ckpt"
    torch.save(state, path)
    with pytest.raises(RuntimeError): module.initialize_from_smartx(model, path)


def test_auto_download(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1)
    path = tmp_path / "model.ckpt"
    torch.save(model.state_dict(), path)
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=download))
    module.initialize_from_smartx(model, "auto")
    assert calls == [{"repo_id": "Trusure/SmartX", "filename": "smartx_large.ckpt"}]


def test_training_stage_roundtrip():
    import ast
    import types
    source = Path(__file__).resolve().parents[1] / "pretrain/trainers/trainer_ours_topk.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OursTopKWrapper")
    hooks = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("on_save_checkpoint", "on_load_checkpoint")]
    ns = {}
    exec(compile(ast.Module(body=hooks, type_ignores=[]), str(source), "exec"), ns)
    original = types.SimpleNamespace(_in_projection_warmup=False, _encoder_warmup_start_step=123)
    restored = types.SimpleNamespace(_in_projection_warmup=False, _encoder_warmup_start_step=None)
    checkpoint = {}
    ns["on_save_checkpoint"](original, checkpoint)
    ns["on_load_checkpoint"](restored, checkpoint)
    assert restored._encoder_warmup_start_step == 123
    assert restored._in_projection_warmup is False
