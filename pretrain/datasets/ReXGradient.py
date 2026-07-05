import os
import re

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import print_rank_zero
from .base_dataset import BaseDataset, collate_fn, load_image_with_dynamic_range


class ReXGradientDataset(BaseDataset):
    """
    Modes:
      - text=False: Dataset indexes over a *flattened* list of all image paths (base dir fixed).
                    __getitem__(i) -> str (image_path)
      - text=True : Dataset indexes row-wise; returns dict with list of image paths and the text.
                    __getitem__(i) -> {"image_paths": [str, ...], "text": str}
    """

    def __init__(
        self,
        data_dir,
        split,
        img_size,
        in_channels=1,
        max_views=2,
        max_regions=4,
        transform=None,
        text=False,
        split_report=True,
        expand_samples=False,
    ):
        super().__init__(img_size, expand_samples=expand_samples)
        self.text_column_name = "finding"
        self.data_dir = data_dir
        self.img_size = img_size
        self.in_channels = in_channels
        self.transform = transform  # TODO Need to be fixed for text=True
        self.split = split
        self.text = text
        self.max_views = max_views
        self.max_regions = max_regions
        self.split_report = split_report

        if self.split == "train":
            labels_csv = os.path.join(self.data_dir, "labels", "train_metadata.csv")
        elif self.split == "validate":
            labels_csv = os.path.join(self.data_dir, "labels", "valid_metadata.csv")
        elif self.split == "test":
            labels_csv = os.path.join(self.data_dir, "labels", "test_metadata.csv")
        else:
            raise ValueError(f"Invalid split: {self.split}")
        if not os.path.exists(labels_csv):
            raise FileNotFoundError(f"Metadata CSV not found: {labels_csv}")

        self.data_list = pd.read_csv(labels_csv)
        if self.text:
            self._filter_invalid_text_rows()

        # Pre-parse and fix paths once.
        img_dir_1024 = os.path.join(self.data_dir, "imgs-1024")
        self.base_img_dir = img_dir_1024 if os.path.isdir(img_dir_1024) else os.path.join(self.data_dir, "imgs-512")
        self._rows_image_paths = [
            self._parse_and_fix_paths(x) for x in self.data_list["image_paths"].tolist()
        ]
        
        # Create expanded data if needed
        self._create_expanded_data()

        if not self.text:
            # Flatten for image-only mode
            self._flat_image_paths = []
            for paths in self._rows_image_paths:
                self._flat_image_paths.extend(paths)
            if self.split == "train":
                assert len(self._flat_image_paths) == 238968
            elif self.split == "validate":
                assert len(self._flat_image_paths) == 17007
            elif self.split == "test":
                assert len(self._flat_image_paths) == 17029
            print_rank_zero(
                f"{self.split.upper()} ReXGradient: {len(self._flat_image_paths):,} Images"
            )
        elif self.expand_samples:
            print_rank_zero(
                f"{self.split.upper()} ReXGradient: {len(self._expanded_data):,} Expanded Samples (from {len(self._rows_image_paths):,} Studies)"
            )
        else:
            if self.split == "train":
                assert len(self._rows_image_paths) == 140000
            elif self.split == "validate":
                assert len(self._rows_image_paths) == 10000
            elif self.split == "test":
                assert len(self._rows_image_paths) == 10000
            print_rank_zero(
                f"{self.split.upper()} ReXGradient: {len(self._rows_image_paths):,} Studies"
            )

    def _filter_invalid_text_rows(self):
        """
        重写过滤方法，检查 finding 或 impression 是否有效
        """
        if not getattr(self, "text", False):
            return
        if not hasattr(self, "data_list") or self.data_list is None:
            return
        
        # 检查 finding 或 impression 是否至少有一个有效
        def has_valid_text(row):
            finding = row.get("finding", "")
            impression = row.get("impression", "")
            return (finding and self._is_valid_text(finding)) or (impression and self._is_valid_text(impression))
        
        if "finding" in self.data_list.columns or "impression" in self.data_list.columns:
            self.data_list = (
                self.data_list[
                    self.data_list.apply(has_valid_text, axis=1)
                ]
                .reset_index(drop=True)
            )

    def _get_report_from_row(self, row):
        """
        从行数据中获取报告文本。
        优先使用 finding，如果为空则使用 impression。
        
        Returns:
            tuple: (report_text, is_impression)
                - report_text: 报告文本字符串
                - is_impression: 布尔值，True表示使用的是impression
        """
        finding = row.get("finding", "")
        if finding and self._is_valid_text(finding):
            return str(finding).strip(), False
        
        impression = row.get("impression", "")
        if impression and self._is_valid_text(impression):
            return str(impression).strip(), True
        
        return "", False

    def _split_report_into_regions(self, report, max_regions=4, is_impression=False):
        """
        将报告拆分为多个区域。
        对于 finding 和 impression，都使用句号拆分（不需要数字前缀拆分）。
        
        Args:
            report: 报告文本
            max_regions: 最大区域数量
            is_impression: 是否为 impression 格式（目前与 finding 使用相同的拆分方式）
            
        Returns:
            tuple: (text_regions, text_attention_mask)
        """
        # 对于 finding 和 impression，都使用句号拆分
        if report and isinstance(report, str):
            text_regions = [
                section.strip()
                for section in report.split(".")
                if section.strip()
            ]
        else:
            text_regions = []
        
        if not text_regions:  # fallback if no sections found
            text_regions = [report] if report else [""]
        
        # Truncate to max_regions if necessary
        text_regions = text_regions[:max_regions]
        num_real_regions = len(text_regions)
        
        # Create attention mask: True for real regions, False for padding
        text_attention_mask = torch.zeros(max_regions, dtype=torch.bool)
        text_attention_mask[:num_real_regions] = True
        
        # Pad with empty strings to reach max_regions
        text_regions = text_regions + [""] * (max_regions - num_real_regions)
        
        return text_regions, text_attention_mask

    def __getitem__(self, idx):
        """重写 __getitem__ 方法以使用新的报告获取和拆分逻辑"""
        if self.text and self.expand_samples:
            # Handle expanded samples mode
            expanded_item = self._expanded_data[idx]
            row = expanded_item['row_data']
            single_image_path = expanded_item['image_path']
            
            # Process text with padding - 使用新的获取方法
            report, is_impression = self._get_report_from_row(row)
            max_retries = 10
            next_idx = idx
            retries = 0
            while report == "" and retries < max_retries:
                next_idx += 1
                if next_idx >= len(self._expanded_data):
                    raise IndexError(f"No valid report found after index {idx}")
                expanded_item = self._expanded_data[next_idx]
                row = expanded_item['row_data']
                report, is_impression = self._get_report_from_row(row)
                retries += 1
            if report == "":
                raise ValueError(f"Could not find a valid report after {max_retries} retries starting from index {idx}")
            
            region_reports, text_attention_mask = self._split_report_into_regions(
                report=report, max_regions=self.max_regions, is_impression=is_impression
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

            # Process text with padding - 使用新的获取方法
            report, is_impression = self._get_report_from_row(row)
            max_retries = 10  # Prevent infinite loop; adjust as needed
            next_idx = idx
            retries = 0
            while report == "" and retries < max_retries:
                next_idx += 1
                if next_idx >= len(self.data_list):
                    raise IndexError(f"No valid report found after index {idx}")
                row = self.data_list.iloc[next_idx]
                image_paths = self._rows_image_paths[next_idx]
                report, is_impression = self._get_report_from_row(row)
                retries += 1
            if report == "":
                raise ValueError(f"Could not find a valid report after {max_retries} retries starting from index {idx}")
            if self.split_report == True:
                region_reports, text_attention_mask = self._split_report_into_regions(
                    report=report, max_regions=self.max_regions, is_impression=is_impression
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
            # image-only mode → 调用基类方法
            return super().__getitem__(idx)


def build_rexgradient_dataloader(
    data_dir, split, img_size, batch_size, in_channels=1, text=False, max_views=2, max_regions=4, split_report=True, expand_samples=False
):
    if split == "train":
        transform = transforms.Compose(
            [
                transforms.Resize(
                    int(img_size * 1.1),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomCrop(img_size),
                transforms.ToTensor(),
            ]
        )
    else:  # validation
        transform = transforms.Compose(
            [
                transforms.Resize(
                    img_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
            ]
        )

    dataset = ReXGradientDataset(
        data_dir=data_dir,
        split=split,
        img_size=img_size,
        in_channels=in_channels,
        transform=transform,
        text=text,
        max_views=max_views,
        max_regions=max_regions,
        split_report=split_report,
        expand_samples=expand_samples,
    )
    loader = DataLoader(
        dataset=dataset,
        shuffle=(split == "train"),
        num_workers=8,
        pin_memory=True,
        batch_size=batch_size,
        collate_fn=collate_fn
        if text
        else None,  # Use custom collate only for text mode
    )
    return loader


if __name__ == "__main__":
    data_dir = "${SMARTX_DATA_ROOT}/ReXGradient-160K"
    dataset = ReXGradientDataset(
        data_dir=data_dir,
        split="train",
        img_size=512,
        in_channels=1,
        transform=transforms.Compose(
            [
                transforms.Resize(
                    int(512 * 1.1),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomCrop(512),
                transforms.ToTensor(),
            ]
        ),
        text=True,
        max_views=2,
        max_regions=4,
    )
    print(f"ReXGradient Dataset: {len(dataset):,}")
    print(dataset[0]["text"])
    print(dataset[0]["text_attention_mask"])
    print(dataset[0]["image"].shape)
    print(dataset[1]["image"].shape)
