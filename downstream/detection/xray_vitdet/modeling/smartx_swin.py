"""SmartX Swin-L backbone for Detectron2 FPN/VitDet training."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from detectron2.layers import ShapeSpec
from detectron2.modeling import Backbone

from downstream.modules import SMARTX_HF_REPO, SMARTX_LARGE_FILENAME, resolve_smartx_checkpoint

logger = logging.getLogger(__name__)

_OUT_FEATURES = ["res2", "res3", "res4", "res5"]
_OUT_CHANNELS = [192, 384, 768, 1536]
_OUT_STRIDES = [4, 8, 16, 32]


def _swin_stages(swin: nn.Module) -> list:
    return list(getattr(swin, "layers", None) or getattr(swin, "stages", []))


def _swin_patch_embed(swin: nn.Module, x: torch.Tensor):
    out = swin.patch_embed(x)
    if isinstance(out, tuple):
        tokens, (height, width) = out
    else:
        tokens = out
        height = width = int(tokens.shape[1] ** 0.5)
    if getattr(swin, "absolute_pos_embed", None) is not None:
        tokens = tokens + swin.absolute_pos_embed
    tokens = swin.pos_drop(tokens)
    return tokens, height, width


def _smartx_vision_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if "teacher" in state_dict and isinstance(state_dict["teacher"], dict):
        state_dict = state_dict["teacher"]

    cleaned = {}
    prefixes = (
        "model.vision_encoder.",
        "vision_encoder.",
        "vlp.vision_encoder.",
        "module.vision_encoder.",
    )
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        for prefix in prefixes:
            if key.startswith(prefix):
                cleaned[key[len(prefix) :]] = value
                break
        else:
            if key.startswith(("patch_embed.", "layers.", "stages.", "norm.")):
                cleaned[key] = value
    if not cleaned:
        raise ValueError(f"No SmartX vision-encoder weights found in {checkpoint_path}")
    return cleaned


class SmartXSwinBackbone(Backbone):
    """Swin-L encoder initialized from the SmartX checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        *,
        repo_id: str = SMARTX_HF_REPO,
        filename: str = SMARTX_LARGE_FILENAME,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        from timm.models.swin_transformer import SwinTransformer

        self.swin = SwinTransformer(
            img_size=768,
            patch_size=4,
            window_size=12,
            embed_dim=192,
            depths=(2, 2, 18, 2),
            num_heads=(6, 12, 24, 48),
        )
        self.swin.head = nn.Identity()

        resolved = resolve_smartx_checkpoint(checkpoint_path, repo_id=repo_id, filename=filename)
        missing, unexpected = self.swin.load_state_dict(
            _smartx_vision_state_dict(resolved),
            strict=False,
        )
        logger.info(
            "Loaded SmartX Swin backbone from %s with %d missing and %d unexpected keys",
            resolved,
            len(missing),
            len(unexpected),
        )
        self.checkpoint_path = resolved
        self._frozen = freeze
        if freeze:
            for parameter in self.swin.parameters():
                parameter.requires_grad = False
            self.swin.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._frozen:
            self.swin.eval()
        return self

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features: dict[str, torch.Tensor] = {}
        tokens, height, width = _swin_patch_embed(self.swin, x)
        batch, _, channels = tokens.shape
        features["res2"] = tokens.transpose(1, 2).reshape(batch, channels, height, width)

        stages = _swin_stages(self.swin)
        for stage, name in zip(stages[:3], ["res3", "res4", "res5"]):
            tokens = stage(tokens)
            if isinstance(tokens, tuple):
                tokens = tokens[0]
            batch, _, channels = tokens.shape
            height = width = int(tokens.shape[1] ** 0.5)
            features[name] = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        return features

    def output_shape(self) -> dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(channels=channels, stride=stride)
            for name, channels, stride in zip(_OUT_FEATURES, _OUT_CHANNELS, _OUT_STRIDES)
        }

    @property
    def size_divisibility(self) -> int:
        return 32
