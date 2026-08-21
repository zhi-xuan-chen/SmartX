from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from PIL import Image


BASE_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "pretrain" / "datasets" / "base_dataset.py"
)
SPEC = spec_from_file_location("smartx_base_dataset", BASE_DATASET_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ImageOnlyDataset(MODULE.BaseDataset):
    def __init__(self, paths):
        super().__init__(img_size=8)
        self.text = False
        self._flat_image_paths = [str(path) for path in paths]
        self.in_channels = 1
        self.transform = lambda image: image


def test_image_only_dataset_handles_optional_masks_and_bad_images(tmp_path):
    image_path = tmp_path / "valid.png"
    Image.new("L", (8, 8)).save(image_path)

    sample = ImageOnlyDataset([image_path])[0]
    assert sample["image_path"] == str(image_path)
    assert "mask" not in sample

