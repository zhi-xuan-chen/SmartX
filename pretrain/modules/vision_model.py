# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import timm.models.vision_transformer
from timm.models.vision_transformer import Attention, vit_base_patch16_224
import torch
import torch.nn as nn
try:
    from transformers import SiglipVisionModel, Siglip2VisionModel, AutoImageProcessor
except Exception:
    SiglipVisionModel = None
    Siglip2VisionModel = None
    AutoImageProcessor = None

try:  # works when repo root is on sys.path (downstream/demo)
    from pretrain.utils.utils import print_rank_zero
except ModuleNotFoundError:  # works when pretrain/ is the import root (training)
    from utils.utils import print_rank_zero


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling + attention capture via hooks (no custom Attention).
    
    Extended to support DINOv3-style dictionary returns with class tokens, patch tokens, and storage tokens.
    """

    def __init__(self, global_pool=False, n_storage_tokens=0, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.in_chans = kwargs["in_chans"]
        self.patch_size = kwargs["patch_size"]

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm

        # Storage tokens (register tokens) like in DINOv3
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.zeros(1, n_storage_tokens, self.embed_dim))
            nn.init.normal_(self.storage_tokens, std=0.02)

        # Hook-related attributes for attention map capture
        self._attention_hook = None
        self._captured_attention = None
        self._target_layer_idx = None
        self._target_head_idx = None
        self._saved_attn_drop_p = (
            None  # optional: to make maps deterministic during training
        )

    # ---------- Hook helpers ----------
    @torch.no_grad()
    def _reconstruct_attention(
        self, module: Attention, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct attention weights A = softmax((Q * scale) @ K^T)
        using the module's parameters and the input x (B, N, C).
        This mirrors timm Attention.forward without modifying it.
        """
        B, N, C = x.shape
        qkv = module.qkv(x)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # each is (B, H, N, C_head)

        attn = (q * module.scale) @ k.transpose(-2, -1)  # (B, H, N, N)

        # If the attention module has any additive bias (rare in plain ViTs), include it.
        # These guards keep it robust across timm variants.
        if hasattr(module, "rel_pos_bias") and module.rel_pos_bias is not None:
            attn = attn + module.rel_pos_bias()  # shapes should broadcast if present
        if hasattr(module, "attn_bias") and module.attn_bias is not None:
            attn = attn + module.attn_bias

        attn = attn.softmax(dim=-1)  # pure weights (before dropout)
        return attn

    def _attention_hook_fn(self, module, inputs, output):
        """
        Forward hook on timm Attention: reconstruct attention weights from the inputs and module params.
        inputs: tuple with x (B, N, C). output is the projected features, not used.
        """
        x = inputs[0]  # (B, N, C)
        attn = self._reconstruct_attention(module, x)
        if self._target_head_idx is not None:
            attn = attn[:, self._target_head_idx, :, :]  # (B, N, N)
        # Detach and move to CPU to avoid retaining graph / reduce GPU mem
        self._captured_attention = attn.detach().float().cpu()

    def _resolve_and_validate_indices(self, layer_idx, head_idx):
        # Resolve negative index
        layer = len(self.blocks) + layer_idx if layer_idx < 0 else layer_idx
        if not (0 <= layer < len(self.blocks)):
            raise ValueError(
                f"attn_layer_idx {layer_idx} out of range (num_layers={len(self.blocks)})"
            )
        if head_idx is not None:
            num_heads = self.blocks[layer].attn.num_heads
            if not (0 <= head_idx < num_heads):
                raise ValueError(
                    f"attn_head_idx {head_idx} out of range (num_heads={num_heads})"
                )
        return layer

    def _register_attention_hook(
        self, layer_idx, head_idx=None, make_deterministic=True
    ):
        """Register a forward hook to capture attention at specified layer without custom modules."""
        self._captured_attention = None
        layer = self._resolve_and_validate_indices(layer_idx, head_idx)
        self._target_layer_idx = layer
        self._target_head_idx = head_idx

        # Remove existing hook if any
        if self._attention_hook is not None:
            self._attention_hook.remove()
            self._attention_hook = None

        attn_module = self.blocks[layer].attn

        # TODO: I am a little confused about the code of this part
        # Optionally make maps deterministic during training by turning off attn dropout
        self._saved_attn_drop_p = None
        if (
            make_deterministic
            and hasattr(attn_module, "attn_drop")
            and isinstance(attn_module.attn_drop, nn.Dropout)
        ):
            self._saved_attn_drop_p = attn_module.attn_drop.p
            attn_module.attn_drop.p = 0.0

        self._attention_hook = attn_module.register_forward_hook(
            self._attention_hook_fn
        )

    def _remove_attention_hook(self):
        """Remove the attention hook and restore state."""
        # Restore dropout prob if we modified it
        if self._target_layer_idx is not None:
            attn_module = self.blocks[self._target_layer_idx].attn
            if self._saved_attn_drop_p is not None and hasattr(
                attn_module, "attn_drop"
            ):
                attn_module.attn_drop.p = self._saved_attn_drop_p
        self._saved_attn_drop_p = None

        if self._attention_hook is not None:
            self._attention_hook.remove()
            self._attention_hook = None

        self._target_layer_idx = None
        self._target_head_idx = None
        # do not clear _captured_attention here; caller may still need it after removal

    def forward_features(self, x, attn_mask=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        # Handle dynamic_img_size: patch_embed may return [B, C, H, W] or [B, H, W, C] instead of [B, N, C]
        if x.ndim == 4:
            # Check which format: timm may use [B, C, H, W] or [B, H, W, C]
            if x.shape[-1] == self.embed_dim:
                # Format: [B, H, W, C] -> flatten to [B, N, C]
                x = x.reshape(B, -1, self.embed_dim)
            else:
                # Format: [B, C, H, W] -> [B, N, C]
                x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # Add storage tokens if they exist
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens.expand(B, -1, -1)
            x = torch.cat((cls_tokens, storage_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Handle variable sequence lengths for multi-crop
        num_prefix_tokens = 1 + self.n_storage_tokens
        if x.shape[1] - num_prefix_tokens != self.pos_embed.shape[1] - 1:
            # Interpolate positional embeddings
            pos_embed = self._resize_pos_embed(self.pos_embed, x.shape[1] - num_prefix_tokens)
        else:
            pos_embed = self.pos_embed
            
        x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, num_prefix_tokens:, :].mean(dim=1)  # global pool without cls token
            cls_tkn = self.fc_norm(x)
        else:
            x = self.norm(x)
            cls_tkn = x[:, 0, :]
        
        return cls_tkn, x[:, num_prefix_tokens:, :]
    
    def dino_forward_features(self, x, masks=None):
        """DINOv3-compatible forward pass returning dictionary with all features.
        
        Args:
            x: Input tensor [B, C, H, W]
            masks: Optional mask tensor for masked image modeling [B, N_patches]
            
        Returns:
            Dictionary with keys:
                - x_norm_clstoken: [B, D] normalized class token
                - x_storage_tokens: [B, n_storage_tokens, D] storage/register tokens
                - x_norm_patchtokens: [B, N_patches, D] normalized patch tokens
                - x_prenorm: [B, 1+n_storage_tokens+N_patches, D] pre-normalization features
                - masks: input masks
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        
        # Handle dynamic_img_size: patch_embed may return [B, C, H, W] or [B, H, W, C] instead of [B, N, C]
        if x.ndim == 4:
            # Check which format: timm may use [B, C, H, W] or [B, H, W, C]
            if x.shape[-1] == self.embed_dim:
                # Format: [B, H, W, C] -> flatten to [B, N, C]
                x = x.reshape(B, -1, self.embed_dim)
            else:
                # Format: [B, C, H, W] -> [B, N, C]
                x = x.flatten(2).transpose(1, 2)

        # Add mask tokens if masks are provided (for masked image modeling)
        if masks is not None:
            # Assume masks is [B, N_patches] boolean tensor
            # Create mask token if not already created
            if not hasattr(self, 'mask_token'):
                self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dim))
                nn.init.zeros_(self.mask_token)
            # Ensure masks and mask_token are on the same device as x
            masks = masks.to(device=x.device)
            mask_token_tensor = self.mask_token.to(device=x.device, dtype=x.dtype)
            x = torch.where(masks.unsqueeze(-1), mask_token_tensor.unsqueeze(0), x)

        # Prepend class token and storage tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens.expand(B, -1, -1)
            x = torch.cat((cls_tokens, storage_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Handle variable sequence lengths for multi-crop
        num_prefix_tokens = 1 + self.n_storage_tokens
        if x.shape[1] - num_prefix_tokens != self.pos_embed.shape[1] - 1:
            # Interpolate positional embeddings
            pos_embed = self._resize_pos_embed(self.pos_embed, x.shape[1] - num_prefix_tokens)
        else:
            pos_embed = self.pos_embed
            
        x = x + pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Store pre-norm output
        x_prenorm = x

        if self.global_pool:
            x_patch = x[:, num_prefix_tokens:, :].mean(dim=1)  # global pool without cls token
            cls_tkn = self.fc_norm(x_patch)
            storage_tkn = torch.empty(B, 0, self.embed_dim, device=x.device)
            patch_tokens = x[:, num_prefix_tokens:, :]
        else:
            x = self.norm(x)
            cls_tkn = x[:, 0, :]
            storage_tkn = x[:, 1:num_prefix_tokens, :] if self.n_storage_tokens > 0 else torch.empty(B, 0, self.embed_dim, device=x.device)
            patch_tokens = x[:, num_prefix_tokens:, :]
        
        # DINOv3-style dictionary return
        return {
            "x_norm_clstoken": cls_tkn,
            "x_storage_tokens": storage_tkn,
            "x_norm_patchtokens": patch_tokens,
            "x_prenorm": x_prenorm,
            "masks": masks,
        }
    
    def _resize_pos_embed(self, pos_embed, num_patches):
        """Resize positional embeddings for different input sizes (multi-crop support)"""
        num_extra_tokens = 1  # cls token
        old_num_patches = pos_embed.shape[1] - num_extra_tokens
        
        if old_num_patches == num_patches:
            return pos_embed
        
        # Separate class token and patch embeddings
        pos_tokens = pos_embed[:, :num_extra_tokens]
        pos_embed_patches = pos_embed[:, num_extra_tokens:]
        
        # Calculate grid sizes
        old_h = old_w = int(old_num_patches ** 0.5)
        new_h = new_w = int(num_patches ** 0.5)
        
        # Interpolate patch embeddings
        embed_dim = pos_embed_patches.shape[-1]
        pos_embed_patches = pos_embed_patches.reshape(1, old_h, old_w, embed_dim).permute(0, 3, 1, 2)
        pos_embed_patches = torch.nn.functional.interpolate(
            pos_embed_patches,
            size=(new_h, new_w),
            mode='bicubic',
            align_corners=False,
        )
        pos_embed_patches = pos_embed_patches.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
        
        # Concatenate class token and resized patch embeddings
        new_pos_embed = torch.cat([pos_tokens, pos_embed_patches], dim=1)
        return new_pos_embed

    def forward_features_attn_map(
        self,
        x,
        attn_mask=None,
        return_attn_map=False,
        attn_layer_idx=-1,
        attn_head_idx=None,
        deterministic_maps=True,
    ):
        """
        Forward pass with attention map capture using a hook on timm Attention.
        Reconstructs attention inside the hook; single model forward; no custom layers.
        """
        if return_attn_map:
            self._register_attention_hook(
                attn_layer_idx, attn_head_idx, make_deterministic=deterministic_maps
            )

        cls_tkn, img_tokens = self.forward_features(x)

        attn_map = self._captured_attention if return_attn_map else None
        self._remove_attention_hook()
        return cls_tkn, img_tokens, attn_map

    def dino_forward(self, x, masks=None, is_training=False):
        """DINOv3-compatible forward pass.
        
        Args:
            x: Input tensor [B, C, H, W] or list of tensors for multi-crop
            masks: Optional mask tensor(s) for masked image modeling
            is_training: If True, return full dict. If False, return only class token
            
        Returns:
            If is_training=True: Dict with all features (DINOv3 format)
            If is_training=False: Class token only [B, D]
        """
        if isinstance(x, list):
            # Multi-crop support: process each crop separately
            if masks is None:
                masks = [None] * len(x)
            results = [self.dino_forward_features(x_i, masks=m_i) for x_i, m_i in zip(x, masks)]
            if is_training:
                return results
            else:
                # Return class tokens only
                return [r["x_norm_clstoken"] for r in results]
        else:
            # Single tensor input
            ret = self.dino_forward_features(x, masks=masks)
            if is_training:
                return ret
            else:
                return ret["x_norm_clstoken"]

    def unpatchify(self, patches):
        """
        patches: (N, L, patch_size**2 *C)
        imgs: (N, C, H, W)
        """
        p = self.patch_size
        c = self.in_chans
        h = w = int(patches.shape[1] ** 0.5)
        assert h * w == patches.shape[1]

        x = patches.reshape(shape=(patches.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(patches.shape[0], c, h * p, w * p))
        return imgs
    
class HFSiglipVisionWrapper(nn.Module):
    """
    适配 HuggingFace 的 SiglipVisionModel 到本项目期望的接口：
      - 属性：embed_dim
      - 方法：forward_features(x) -> (cls_token, patch_tokens)
    """

    def __init__(self, model_id: str = "google/siglip-base-patch16-224", device=None):
        super().__init__()
        model_id_lower = (model_id or "").lower()

        # 区分 SigLIP v1 与 SigLIP2
        if "siglip2" in model_id_lower:
            if Siglip2VisionModel is None:
                raise ImportError(
                    "当前 transformers 版本不支持 Siglip2VisionModel，"
                    "请升级 transformers 到 >= 4.49（或对应该模型的版本）后再使用 SigLIP2 视觉 encoder。"
                )
            self.model = Siglip2VisionModel.from_pretrained(model_id)
        else:
            if SiglipVisionModel is None:
                raise ImportError("transformers 未安装或版本不支持 SiglipVisionModel")
            self.model = SiglipVisionModel.from_pretrained(model_id)

        # 使用 HF AutoImageProcessor 做与预训练一致的归一化 / resize
        # 注意：dataloader 已经 ToTensor（[0,1]），这里关闭 rescale，只保留 normalize 等。
        if AutoImageProcessor is None:
            raise ImportError(
                "transformers 未提供 AutoImageProcessor，无法为 SigLIP 使用官方归一化配置。"
            )
        self.processor = AutoImageProcessor.from_pretrained(
            model_id,
            use_fast=False,
            do_rescale=False,  # 与 Rad-DINO 一致：ToTensor 已经完成 [0,255]->[0,1]
        )
        self.config = getattr(self.model, "config", None)
        self.embed_dim = int(getattr(self.config, "hidden_size", 768))
        self.is_hf_pretrained = True
        self._device_cache = device

        # 冻结未使用的 head 参数，避免 DDP 未用参数报错
        try:
            vm = getattr(self.model, "vision_model", None)
            if vm is not None and hasattr(vm, "head") and vm.head is not None:
                for p in vm.head.parameters():
                    p.requires_grad = False
        except Exception:
            pass

        # ViT 兼容信息：patch_size 与 num_patches（供上层读取）
        patch = int(getattr(self.config, "patch_size", 16))
        image_size = int(getattr(self.config, "image_size", 224))
        self._patch_size = (patch, patch)
        num_patches_side = image_size // patch
        self._num_patches = int(num_patches_side * num_patches_side)

        class _PatchShim:
            def __init__(self, patch_size, num_patches):
                self.patch_size = patch_size
                self.num_patches = num_patches

        self.patch_embed = _PatchShim(self._patch_size, self._num_patches)

    def forward_features(self, x, attn_mask=None):
        # x: [B, C, H, W]，dataloader 中已做 ToTensor / Resize，这里只负责：
        #   1) 通道数适配到 3
        #   2) 使用 HF AutoImageProcessor 做与预训练一致的 Normalize / CenterCrop 等
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input [B, C, H, W], got shape: {tuple(x.shape)}")

        # 使用官方 processor 做 SigLIP/SigLIP2 预处理（含 normalize、pixel_attention_mask、spatial_shapes 等）
        # SigLIP2 必须传入 processor 返回的完整 inputs（model(**inputs)），不能只传 pixel_values
        inputs = self.processor(
            images=x,
            return_tensors="pt",
        )
        inputs = {k: v.to(device=x.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        cls_tkn = outputs.pooler_output
        patch_tokens = outputs.last_hidden_state
        return cls_tkn, patch_tokens



def create_vision_encoder(
    img_size=512,
    patch_size=16,
    in_chans=1,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    pretrained=None,
    dynamic_img_size=True,
    **kwargs,
):
    if kwargs.get('hf_siglip_model_id', None):
        # 优先使用 HF 的 SigLIP 视觉模型
        model_id = kwargs['hf_siglip_model_id']
        model = HFSiglipVisionWrapper(model_id=model_id)
        print(f"Using HF SigLIP vision encoder: {model_id}")
        return model
    if kwargs.get('vision_encoder_name', None) == 'vit_base_patch16_224':
        model = vit_base_patch16_224(
            in_chans=in_chans,
        )
        print(f"Using {kwargs.get('vision_encoder_name', None)} as vision encoder")
    else:
        model = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            dynamic_img_size=dynamic_img_size,  # Enable flexible input sizes for multi-crop
            # **kwargs,
        )
        if pretrained is not None:
            weight = load_weight(pretrained)
            missing, unexpected = model.load_state_dict(weight, strict=False)
            print_rank_zero(f"Missing: {missing}")
            print_rank_zero(f"Unexpected: {unexpected}")
    return model


def vit_base_patch16_clip_224(
    img_size=224,
    patch_size=16,
    in_chans=1,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    **kwargs,
):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        norm_layer=norm_layer,
        **kwargs,
    )
    return model


def vit_base_patch16(
    img_size=512,
    patch_size=16,
    in_chans=1,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    **kwargs,
):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        norm_layer=norm_layer,
        dynamic_img_size=True,
        **kwargs,
    )
    return model


def vit_large_patch16(
    img_size=512,
    patch_size=16,
    in_chans=1,
    embed_dim=1024,
    depth=24,
    num_heads=16,
    mlp_ratio=4,
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    **kwargs,
):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        norm_layer=norm_layer,
        dynamic_img_size=True,  # Enable flexible input sizes for multi-crop
        **kwargs,
    )
    return model


def vit_huge_patch14(
    img_size=512,
    patch_size=14,
    in_chans=1,
    embed_dim=1280,
    depth=32,
    num_heads=16,
    mlp_ratio=4,
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    **kwargs,
):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        norm_layer=norm_layer,
        dynamic_img_size=True,  # Enable flexible input sizes for multi-crop
        **kwargs,
    )
    return model


def load_weight(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]

    # Replace 'model.' with ''
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    # Check if keys contain 'vision_encoder' or 'encoder' and handle accordingly
    has_vision_encoder = any("vision_encoder" in k for k in state_dict.keys())
    has_encoder = any("encoder" in k for k in state_dict.keys())

    # Replace 'vision_encoder.' or 'encoder.' with ''
    if has_vision_encoder:
        state_dict = {
            k.replace("vision_encoder.", ""): v
            for k, v in state_dict.items()
            if "vision_encoder" in k
        }
    elif has_encoder:
        state_dict = {
            k.replace("encoder.", ""): v
            for k, v in state_dict.items()
            if "encoder" in k
        }

    # Remove head.bias and head.weight from the state dict if exists
    keys_to_remove = ["head.bias", "head.weight"]
    for key in keys_to_remove:
        if key in state_dict:
            del state_dict[key]

    return state_dict
