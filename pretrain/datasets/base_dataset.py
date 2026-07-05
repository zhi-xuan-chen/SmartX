import json
import os
import re
import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


def load_image_with_dynamic_range(path, in_chans=1):
    """
    Load PNG/JPEG images with automatic handling for 8-bit and 16-bit depth.
    Scales pixel values linearly into [0, 1] before converting back to a PIL Image.
    """
    img = Image.open(path)
    arr = np.array(img)

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        divisor = float(info.max)
        arr = arr.astype(np.float32) / divisor
    else:
        arr = arr.astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)

    arr = np.clip(arr, 0.0, 1.0)

    if arr.ndim == 2 and in_chans == 3:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1 and in_chans == 3:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.ndim == 2 and in_chans == 1:
        arr = arr[..., None]

    pil_img = Image.fromarray((arr * 255).astype(np.uint8))
    if in_chans == 3:
        pil_img = pil_img.convert("RGB")
    else:
        pil_img = pil_img.convert("L")

    return pil_img


class BaseDataset(Dataset):
    """
    Modes:
      - text=False: Dataset indexes over a *flattened* list of all image paths (base dir fixed).
                    __getitem__(i) -> str (image_path)
      - text=True : Dataset indexes row-wise; returns dict with list of image paths and the text.
                    __getitem__(i) -> {"image_paths": [str, ...], "text": str}
    """

    def __init__(
        self,
        img_size,
        mask_patch_size=32,
        model_patch_size=4,
        mask_ratio=0.9,
        mask_generator=False,
        expand_samples=False,
    ):  
        self.expand_samples = expand_samples
        if mask_generator:
            self.mask_generator = MaskGenerator(
                input_size=img_size,
                mask_patch_size=mask_patch_size,
                model_patch_size=model_patch_size,
                mask_ratio=mask_ratio,
            )

    def __len__(self):
        if self.text and self.expand_samples:
            return len(self._expanded_data)
        elif self.text:
            return len(self.data_list)
        return len(self._flat_image_paths)

    def __getitem__(self, idx):
        if self.text and self.expand_samples:
            # Handle expanded samples mode
            expanded_item = self._expanded_data[idx]
            row = expanded_item['row_data']
            single_image_path = expanded_item['image_path']
            
            # Process text with padding
            report = row.get(self.text_column_name, "")
            max_retries = 10
            next_idx = idx
            retries = 0
            while report == "" and retries < max_retries:
                next_idx += 1
                if next_idx >= len(self._expanded_data):
                    raise IndexError(f"No valid report found after index {idx}")
                expanded_item = self._expanded_data[next_idx]
                row = expanded_item['row_data']
                report = row.get(self.text_column_name, "")
                retries += 1
            if report == "":
                raise ValueError(f"Could not find a valid report after {max_retries} retries starting from index {idx}")
            
            region_reports, text_attention_mask = self._split_report_into_regions(
                report=report, max_regions=self.max_regions
            )
            
            # Load single image
            try:
                image = load_image_with_dynamic_range(single_image_path, in_chans=self.in_channels)
                image = self.transform(image)
            except Exception as e:
                print(f"Warning: Could not load image {single_image_path}: {e}")
                # Create dummy image if loading fails
                image = torch.zeros(self.in_channels, self.img_size, self.img_size)
            
            # For expanded samples, we only have one image, so create a single-view tensor
            # Shape: [1, C, H, W] where 1 is the view dimension
            if image.dim() == 2:  # [H, W]
                image = image.unsqueeze(0)  # [1, H, W]
            elif image.dim() == 3:  # [C, H, W]
                image = image.unsqueeze(0)  # [1, C, H, W]
            
            # Create attention mask for single image
            image_attention_mask = torch.ones(1, dtype=torch.bool)
            
            return {
                "image": image,  # Single image tensor [1, C, H, W]
                "image_attention_mask": image_attention_mask,  # Mask tensor [1]
                "text": region_reports,  # List of text strings padded to max_regions
                "text_attention_mask": text_attention_mask,  # Mask tensor [max_regions]
                "image_paths": [single_image_path],
            }
        elif self.text:
            row = self.data_list.iloc[idx]
            image_paths = self._rows_image_paths[idx]

            # Process text with padding
            report = row.get(self.text_column_name, "")
            max_retries = 10  # Prevent infinite loop; adjust as needed
            next_idx = idx
            retries = 0
            while report == "" and retries < max_retries:
                next_idx += 1
                if next_idx >= len(self.data_list):
                    raise IndexError(f"No valid report found after index {idx}")
                row = self.data_list.iloc[next_idx]
                image_paths = self._rows_image_paths[next_idx]
                report = row.get(self.text_column_name, "")
                retries += 1
            if report == "":
                raise ValueError(f"Could not find a valid report after {max_retries} retries starting from index {idx}")
            if self.split_report == True:
                region_reports, text_attention_mask = self._split_report_into_regions(
                    report=report, max_regions=self.max_regions
                )
            else:
                region_reports = report
                text_attention_mask = None
            # Open all images from the image paths
            images = []
            for img_path in image_paths:
                try:
                    image = load_image_with_dynamic_range(img_path, in_chans=self.in_channels)
                    image = self.transform(image)  # Transform individual images
                    images.append(image)
                except Exception as e:
                    print(f"Warning: Could not load image {img_path}: {e}")
                    continue

            # Stack multiple images and create attention mask
            stacked_images, attention_mask = self._pad_and_stack_images(
                images=images, max_views=self.max_views, img_size=self.img_size
            )

            return {
                "image": stacked_images,  # Stacked images tensor [max_views, 1, H, W]
                "image_attention_mask": attention_mask,  # Mask tensor [max_views]
                "text": region_reports,  # List of text strings padded to max_regions
                "text_attention_mask": text_attention_mask,  # Mask tensor [max_regions]
                "image_paths": image_paths,
            }

        else:
            # image-only mode → return just a single fixed image path (str)
            image_path = self._flat_image_paths[idx]
            try:
                image = load_image_with_dynamic_range(image_path, in_chans=self.in_channels)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                return self[idx + 1]
            image = self.transform(image)
            return {
                "image": image,
                "image_path": image_path,
                "mask": self.mask_generator(),
            }

    def _is_valid_text(self, value):
        """
        Return True if value is a valid non-empty text.
        Filters out NaN/None/empty/placeholder tokens (e.g., 'nan', 'none', 'null', 'n/a', '-').
        """
        if value is None:
            return False
        if pd.isna(value):
            return False
        s = str(value).strip()
        if s == "":
            return False
        lowered = s.lower()
        invalid_tokens = {
            "nan",
            "none",
            "null",
            "n/a",
            "na",
            "n\\a",
            "-",
            "--",
            "unknown",
        }
        return lowered not in invalid_tokens

    def _filter_invalid_text_rows(self):
        """
        Filter in-place self.data_list to keep only rows whose text column has valid content.
        No-op if not in text mode or if required attributes are missing.
        """
        if not getattr(self, "text", False):
            return
        if not hasattr(self, "data_list") or self.data_list is None:
            return
        if not hasattr(self, "text_column_name"):
            return
        if self.text_column_name not in self.data_list.columns:
            return
        self.data_list = (
            self.data_list[
                self.data_list[self.text_column_name].apply(self._is_valid_text)
            ]
            .reset_index(drop=True)
        )

    def _parse_and_fix_paths(self, cell):
        """
        Accepts:
          - JSON list string: '["a.png","b.png"]'
          - Python-like list string: "['a.png', 'b.png']"
          - Delimited string: 'a.png;b.png' or 'a.png,b.png'
          - Single path string: 'a.png'
        Returns list[str] with base directory fixed and '../deid_png' removed. # TODO: Remove if not needed
        """
        if pd.isna(cell):
            return []

        s = str(cell).strip()

        # Try JSON
        paths = None
        if s.startswith("[") and s.endswith("]"):
            try:
                paths = json.loads(s)
            except Exception:
                s2 = s.replace("'", '"')
                try:
                    paths = json.loads(s2)
                except Exception:
                    paths = None

        # Delimiter fallback
        if paths is None:
            if ";" in s:
                paths = [p.strip() for p in s.split(";") if p.strip()]
            elif "," in s:
                paths = [p.strip() for p in s.split(",") if p.strip()]
            else:
                paths = [s]

        # Inline fix (replace '../deid_png', strip, join with base dir)
        fixed = [
            os.path.join(
                self.base_img_dir,
                str(p)
                .replace("../deid_png", "")
                .lstrip("/")
                .strip(),  # TODO: Remove if not needed
            )
            for p in paths
        ]
        return fixed

    def _pad_and_stack_images(self, images, max_views=4, img_size=512):
        """
        Stack multiple images along a new dimension while preserving channel dimension.
        Always pads to max_views and returns attention masks.

        Args:
            images: List of torch.Tensor images, each of shape [C, H, W] or [1, H, W]
            max_views: Maximum number of views to stack
            img_size: Image size for dummy images

        Returns:
            tuple: (stacked_images, attention_mask)
                - stacked_images: torch.Tensor of shape [max_views, C, H, W]
                - attention_mask: torch.Tensor of shape [max_views] where True=real, False=padded
        """
        # Take at most max_views real images
        if len(images) > max_views: # random sample max_views images
            real_images = random.sample(images, max_views)
        else:
            real_images = images[:max_views]
        num_real = len(real_images)

        # Create attention mask: True for real images, False for padded
        attention_mask = torch.zeros(max_views, dtype=torch.bool)
        attention_mask[:num_real] = True

        # Prepare final list of images (real + zero padding)
        final_images = []

        # Add real images
        for img in real_images:
            # Ensure img has shape [C, H, W]
            if img.dim() == 2:  # [H, W]
                img = img.unsqueeze(0)  # [1, H, W]
            final_images.append(img)

        # Pad with zero images if needed
        if num_real < max_views:
            # Get image dimensions from first real image, or use default
            if num_real > 0:
                c, h, w = final_images[0].shape
            else:
                c, h, w = 1, img_size, img_size

            # Create zero images with same channel dimension
            zero_image = torch.zeros(c, h, w)
            for _ in range(max_views - num_real):
                final_images.append(zero_image)

        # Stack all images along new dimension (dim=0) while preserving channels
        stacked_images = torch.stack(final_images, dim=0)  # [max_views, C, H, W]
        return stacked_images, attention_mask

    def _split_report_into_regions(self, report, max_regions=4):
        """
        Process text content to match model's expected nested list format with padding.
        The model expects [batch_size, max_regions] where each element is a text string.

        Args:
            report: Raw text content (string)
            max_regions: Maximum number of text regions (for padding)

        Returns:
            tuple: (text_regions, text_attention_mask)
                - text_regions: List of text strings padded to max_regions
                - text_attention_mask: torch.Tensor of shape [max_regions] where True=real, False=padded
        """
        if report and isinstance(report, str):
            # Split report into sections/regions if it contains multiple parts
            # TODO: This is a simple approach - you might want to use more sophisticated parsing
            # Split report into regions based on periods, stripping whitespace
            def simple_clean(text):
                return re.sub(r'\n\d+\.?\s*|\n', ' ', text).strip() if pd.notna(text) else ""
            
            text_regions = [
                section.strip()
                for section in simple_clean(report).split(".")
                if section.strip()
            ]
            if not text_regions:  # fallback if no sections found
                text_regions = [report]
        else:
            text_regions = [""]  # empty text if no report available

        # Truncate to max_regions if necessary
        text_regions = text_regions[:max_regions]
        num_real_regions = len(text_regions)

        # Create attention mask: True for real regions, False for padding
        text_attention_mask = torch.zeros(max_regions, dtype=torch.bool)
        text_attention_mask[:num_real_regions] = True

        # Pad with empty strings to reach max_regions
        text_regions = text_regions + [""] * (max_regions - num_real_regions)

        return text_regions, text_attention_mask

    def _create_expanded_data(self):
        """
        Create expanded data structure for sample expansion.
        This method should be called by subclasses after setting up data_list and _rows_image_paths.
        """
        if self.expand_samples and self.text:
            # Create expanded data structure for text mode
            self._expanded_data = []
            # Use enumerate to get position index, not DataFrame index
            for pos_idx, (df_idx, row) in enumerate(self.data_list.iterrows()):
                image_paths = self._rows_image_paths[pos_idx]  # Use position index
                for img_path in image_paths:
                    self._expanded_data.append({
                        'row_idx': df_idx,  # Store original DataFrame index
                        'image_path': img_path,
                        'row_data': row
                    })


def collate_fn(batch):
    """
    Custom collate function to handle batching for datasets with multi-view images and text.
    Converts individual samples into proper batch format expected by the model.

    Args:
        batch: List of samples from dataset __getitem__

    Returns:
        Batched data in model-expected format
    """
    # Stack images: [B, V, C, H, W]
    images = torch.stack([item["image"] for item in batch], dim=0)

    # Stack image attention masks: [B, V]
    image_attention_masks = torch.stack(
        [item["image_attention_mask"] for item in batch], dim=0
    )

    # Collect text as nested list: [B, max_regions]
    texts = [item["text"] for item in batch]

    # Stack text attention masks: [B, max_regions]
    text_attention_masks = torch.stack(
        [item["text_attention_mask"] for item in batch], dim=0
    )

    # Collect image paths: [B, max_views]
    image_paths = [item["image_paths"] for item in batch]

    result = {
        "image": images,
        "image_attention_mask": image_attention_masks,
        "text": texts,
        "text_attention_mask": text_attention_masks,
        "image_paths": image_paths,
    }

    return result

class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask