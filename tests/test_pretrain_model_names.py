"""Check public model names and vision initialization without large weights."""
import ast
import os
from pathlib import Path
import sys
import types

import pytest
import torch


def test_public_factory_loads_teacher_checkpoint(tmp_path, monkeypatch):
    # Exercise the actual factory in isolation from optional model imports.
    source = Path(__file__).resolve().parents[1] / 'pretrain/modules/ours_topk.py'
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'create_topk_vlp_model')
    captured = {}
    class Vision(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            captured.update(kwargs)
    stub = types.ModuleType('timm.models.swin_transformer')
    stub.SwinTransformer = Vision
    monkeypatch.setitem(sys.modules, 'timm.models.swin_transformer', stub)
    namespace = dict(torch=torch, os=os, print_rank_zero=lambda *args: None,
                     create_text_encoder=lambda **kwargs: (None, None),
                     XRayTopKVLP=lambda **kwargs: types.SimpleNamespace(**kwargs))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    checkpoint = tmp_path / 'vision.pth'
    torch.save({'teacher': {'module.weight': torch.tensor([3.])}}, checkpoint)
    model = namespace['create_topk_vlp_model'](
        vision_encoder_name='smartx_large', vision_init_path=str(checkpoint))
    assert captured['img_size'] == 768
    assert captured['embed_dim'] == 192
    assert model.vision_encoder.weight.item() == 3.
    assert model.vision_encoder_name == 'smartx_large'


def test_resume_skips_initialization_download(monkeypatch):
    source = Path(__file__).resolve().parents[1] / 'pretrain/modules/ours_topk.py'
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'create_topk_vlp_model')
    class Vision(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
    stub = types.ModuleType('timm.models.swin_transformer')
    stub.SwinTransformer = Vision
    monkeypatch.setitem(sys.modules, 'timm.models.swin_transformer', stub)
    def forbidden(*args, **kwargs):
        raise AssertionError('Resume must not load initialization weights')
    monkeypatch.setattr(torch, 'load', forbidden)
    namespace = dict(torch=torch, os=os, print_rank_zero=lambda *a: None,
                     create_text_encoder=lambda **kw: (None, None),
                     XRayTopKVLP=lambda **kw: types.SimpleNamespace(**kw))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    model = namespace['create_topk_vlp_model'](
        vision_encoder_name='smartx_large', smartx_checkpoint='auto', resume_training=True)
    assert model.vision_encoder.weight.item() == 0
