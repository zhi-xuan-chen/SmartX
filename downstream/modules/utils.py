"""SmartX model loading helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

SMARTX_HF_REPO = "Trusure/SmartX"
SMARTX_LARGE_FILENAME = "smartx_large.ckpt"


class _ProjectedVisionWrapper(nn.Module):
    def __init__(self, vlp_model):
        super().__init__()
        self.vlp = vlp_model

    def forward(self, x):
        feat = self.vlp.vision_encoder.forward_features(x)
        if isinstance(feat, tuple):
            feat = feat[0]
        projected = self.vlp.proj_img(feat)
        return SimpleNamespace(pooler_output=F.normalize(projected, dim=-1))


class _ProjectedTextWrapper(nn.Module):
    def __init__(self, vlp_model):
        super().__init__()
        self.vlp = vlp_model

    def forward(self, input_ids, attention_mask=None, **kwargs):
        out = self.vlp.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        feat = getattr(out, "pooler_output", None)
        if feat is None:
            feat = out.last_hidden_state[:, 0]
        projected = self.vlp.proj_t(feat)
        return SimpleNamespace(pooler_output=F.normalize(projected, dim=-1))


def resolve_smartx_checkpoint(
    checkpoint_path: str | Path | None = None,
    repo_id: str = SMARTX_HF_REPO,
    filename: str = SMARTX_LARGE_FILENAME,
) -> str:
    """Return a local checkpoint path, downloading from Hugging Face if needed."""
    if checkpoint_path:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return str(path)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Install huggingface_hub or pass --checkpoint/path to a local SmartX ckpt."
        ) from exc

    return hf_hub_download(repo_id=repo_id, filename=filename)


def _checkpoint_state_dict(checkpoint_path: str, map_location="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        if key in cleaned:
            raise ValueError(f"Duplicate checkpoint key after prefix normalization: {key}")
        cleaned[key] = value
    return cleaned


def _load_compatible_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]):
    model_state = model.state_dict()
    compatible = {}
    skipped_shape = []
    skipped_missing = []

    for key, value in state_dict.items():
        if key not in model_state:
            skipped_missing.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        compatible[key] = value

    missing = sorted(set(model_state) - set(compatible))
    if missing or skipped_shape or skipped_missing:
        raise ValueError(
            "Incompatible SmartX checkpoint: "
            f"{len(missing)} missing, {len(skipped_shape)} shape mismatches, "
            f"{len(skipped_missing)} unexpected keys. Use a complete checkpoint "
            "with the matching model architecture."
        )
    message = model.load_state_dict(compatible, strict=True)
    return {
        "missing_keys": list(message.missing_keys),
        "unexpected_keys": list(message.unexpected_keys),
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
    }


def create_smartx_vlp_model(
    checkpoint_path: str | Path | None = None,
    *,
    repo_id: str = SMARTX_HF_REPO,
    filename: str = SMARTX_LARGE_FILENAME,
    text_encoder_name: str = "pubmedbert",
    output_dim: int = 512,
    map_location="cpu",
):
    """Create the SmartX large VLP model and load the official Hugging Face checkpoint."""
    from timm.models.swin_transformer import SwinTransformer

    from pretrain.modules.ours_topk import XRayTopKVLP
    from pretrain.modules.text_model import create_text_encoder

    resolved_checkpoint = resolve_smartx_checkpoint(checkpoint_path, repo_id, filename)

    vision_encoder = SwinTransformer(
        img_size=768,
        patch_size=4,
        window_size=12,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
    )
    if hasattr(vision_encoder, "head"):
        vision_encoder.head = nn.Identity()

    text_encoder, tokenizer = create_text_encoder(
        text_encoder_name=text_encoder_name, load_pretrained=False
    )
    model = XRayTopKVLP(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        output_dim=output_dim,
        vision_encoder_name="smartx_large",
    )

    load_report = _load_compatible_state_dict(
        model,
        _checkpoint_state_dict(resolved_checkpoint, map_location=map_location),
    )
    model.eval()
    model.load_report = load_report
    model.checkpoint_path = resolved_checkpoint
    return model, tokenizer, load_report


def create_clip_style_model_global(cfg: dict):
    """Load SmartX as CLIP-style image/text encoders for demos and evaluations."""
    model_name = str(cfg.get("model_name", "smartx_large")).lower()
    if model_name not in {"smartx", "smartx_large"}:
        raise ValueError(
            "This release loader supports SmartX large only. "
            "Use model_name='smartx_large'."
        )

    vlp_model, tokenizer, load_report = create_smartx_vlp_model(
        checkpoint_path=cfg.get("pretrained_path") or cfg.get("checkpoint"),
        repo_id=cfg.get("repo_id") or SMARTX_HF_REPO,
        filename=cfg.get("filename") or SMARTX_LARGE_FILENAME,
        text_encoder_name=cfg.get("text_encoder_name", "pubmedbert"),
        output_dim=int(cfg.get("output_dim", 512)),
    )

    model_info = {
        "model_name": model_name,
        "checkpoint_path": vlp_model.checkpoint_path,
        "repo_id": cfg.get("repo_id") or SMARTX_HF_REPO,
        "filename": cfg.get("filename") or SMARTX_LARGE_FILENAME,
        "image_embed_dim": int(vlp_model.proj_out),
        "text_embed_dim": int(vlp_model.proj_out),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "load_report": load_report,
    }
    return _ProjectedVisionWrapper(vlp_model), _ProjectedTextWrapper(vlp_model), tokenizer, model_info
