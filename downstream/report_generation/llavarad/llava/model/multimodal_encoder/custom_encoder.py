import sys
import os

import torch
import torch.nn as nn

_SMARTX_VISION_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
if _SMARTX_VISION_ROOT not in sys.path:
    sys.path.insert(0, _SMARTX_VISION_ROOT)

from vision_encoders.utils import VIS_ENCODER_CONFIG, create_vision_encoder, get_extract_features_fn


class CustomVisionTower(nn.Module):
    """
    Wraps a custom radiology vision encoder (smartx_large, rad_dino, etc.) behind the
    same interface that LLaVA expects from CLIPVisionTower:
      - forward(images) -> patch_tokens [B, N, vis_dim]
      - hidden_size property
      - is_loaded / load_model()
      - image_processor property -> CustomImageProcessor
      - dummy_feature property
    """

    def __init__(self, encoder_name: str, args, delay_load: bool = False, **kwargs):
        super().__init__()
        self.encoder_name = encoder_name

        cfg = VIS_ENCODER_CONFIG[encoder_name]
        self._hidden_size = cfg["vis_dim"]
        self._is_loaded = False

        # pretrained_path: custom field first, then OpenCLIP-style checkpoint field
        self.pretrained_path = (
            getattr(args, 'custom_vision_pretrained_path', None)
            or getattr(args, 'mm_vision_tower_checkpoint', None)
        )

        if not delay_load:
            self.load_model()

    def load_model(self):
        cfg = {
            "model_name": self.encoder_name,
            "pretrained_path": self.pretrained_path,
        }
        self.encoder = create_vision_encoder(cfg)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.extract_fn = get_extract_features_fn(self.encoder_name)
        self._is_loaded = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.is_loaded:
            self.encoder.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W] preprocessed by CustomImageProcessor
        Returns:
            patch_tokens: [B, N, vis_dim]
        """
        patch_tokens, _ = self.extract_fn(self.encoder, images)
        return patch_tokens

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def dummy_feature(self) -> torch.Tensor:
        return torch.zeros(1, self._hidden_size)

    @property
    def image_processor(self):
        # Import here to avoid circular imports at module load time
        _llava_data = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        )
        if _llava_data not in sys.path:
            sys.path.insert(0, _llava_data)
        from data_processor import CustomImageProcessor
        return CustomImageProcessor(self.encoder_name)
