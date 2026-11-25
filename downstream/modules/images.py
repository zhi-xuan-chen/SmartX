"""Image decoding shared by classification and retrieval."""
from pathlib import Path

import numpy as np
from PIL import Image


def array_to_pil(array, in_channels=3):
    array = np.asarray(array)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise ValueError("Expected a 2D radiograph")
    array = array.astype(np.float32)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("Image pixels must be nonempty and finite")
    # Match the standard radiograph loading and normalization pipeline.
    array = (array - array.min()) / (array.max() - array.min() + 1e-8) * 255
    return Image.fromarray(array.astype(np.uint8)).convert("L" if in_channels == 1 else "RGB")


def load_image(path, in_channels=3):
    if in_channels not in (1, 3):
        raise ValueError("in_channels must be 1 or 3")
    path = Path(path)
    suffix = path.suffix.lower()
    mode = "L" if in_channels == 1 else "RGB"
    if suffix in {".dcm", ".dicom"}:
        import pydicom
        dataset = pydicom.dcmread(str(path))
        pixels = dataset.pixel_array
        if int(getattr(dataset, "NumberOfFrames", 1)) > 1:
            pixels = pixels[0]
        return array_to_pil(pixels, in_channels)
    if suffix in {".mha", ".mhd", ".nii"} or str(path).lower().endswith(".nii.gz"):
        import SimpleITK as sitk
        return array_to_pil(sitk.GetArrayFromImage(sitk.ReadImage(str(path))), in_channels)
    with Image.open(path) as image:
        return image.convert(mode)
