import sys
import os

import torch
from PIL import Image

_SMARTX_VISION_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
if _SMARTX_VISION_ROOT not in sys.path:
    sys.path.insert(0, _SMARTX_VISION_ROOT)

from vision_encoders.utils import VIS_ENCODER_CONFIG
from vision_encoders.transforms import get_custom_image_transform


class CustomImageProcessor:
    """
    Wraps a per-encoder torchvision transform with the same .preprocess() API
    as HuggingFace CLIPImageProcessor. Fields provided:
      - .preprocess(pil_img, return_tensors) -> {'pixel_values': [1,C,H,W]}
      - .image_mean  (for fallback blank-image creation in LazySupervisedDataset)
      - .crop_size   (for dummy zero-image in text-only samples)
    """

    def __init__(self, encoder_name: str):
        cfg = VIS_ENCODER_CONFIG[encoder_name]
        self._transform = get_custom_image_transform(encoder_name)
        self.crop_size = {"height": cfg["input_size"], "width": cfg["input_size"]}
        self.image_mean = [0.5, 0.5, 0.5]

    def preprocess(self, image: Image.Image, return_tensors: str = "pt"):
        tensor = self._transform(image)      # [C, H, W]
        return {"pixel_values": tensor.unsqueeze(0)}  # [1, C, H, W]
