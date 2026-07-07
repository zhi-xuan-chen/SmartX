"""Detection transforms used by the SmartX VitDet config."""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from detectron2.data.transforms import Augmentation, CropTransform, PadTransform, TransformList


class CenterCropFixedSize(Augmentation):
    """Deterministic fixed-size center crop, with optional centered padding."""

    def __init__(
        self,
        crop_size: Union[int, Tuple[int, int]],
        pad: bool = True,
        pad_value: float = 128.0,
        seg_pad_value: int = 255,
    ) -> None:
        super().__init__()
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.crop_size = (int(crop_size[0]), int(crop_size[1]))
        self.pad = pad
        self.pad_value = pad_value
        self.seg_pad_value = seg_pad_value

    def get_transform(self, image: np.ndarray) -> TransformList:
        input_h, input_w = image.shape[:2]
        crop_h, crop_w = self.crop_size
        output_h = min(input_h, crop_h)
        output_w = min(input_w, crop_w)
        offset_y = max((input_h - crop_h) // 2, 0)
        offset_x = max((input_w - crop_w) // 2, 0)
        transforms = [
            CropTransform(offset_x, offset_y, output_w, output_h, input_w, input_h)
        ]
        if self.pad:
            pad_h = max(crop_h - output_h, 0)
            pad_w = max(crop_w - output_w, 0)
            if pad_h or pad_w:
                pad_top = pad_h // 2
                pad_left = pad_w // 2
                transforms.append(
                    PadTransform(
                        pad_left,
                        pad_top,
                        pad_w - pad_left,
                        pad_h - pad_top,
                        output_w,
                        output_h,
                        self.pad_value,
                        self.seg_pad_value,
                    )
                )
        return TransformList(transforms)
