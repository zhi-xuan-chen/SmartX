import os
import re

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import print_rank_zero
from .base_dataset import BaseDataset, collate_fn, load_image_with_dynamic_range


class MIMICDataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        split,
        img_size,
        in_channels=1,
        max_views=2,
        text=False,
        max_regions=4,
        transform=None,
        split_report=True,
        expand_samples=False,
    ):
        super().__init__(img_size, expand_samples=expand_samples)
        self.text_column_name = "findings_section"
        self.data_dir = data_dir
        self.img_size = img_size
        self.in_channels = in_channels
        self.transform = transform  # TODO Need to be fixed for text=True
        self.split = split
        self.text = text
        self.max_views = max_views
        self.max_regions = max_regions
        self.split_report = split_report
        labels_csv = os.path.join(
            self.data_dir, "labels/mimic_engineered_merge_with_split.csv"
        )
        self.data_list = pd.read_csv(labels_csv)
        self.data_list = self.data_list[self.data_list["split"] == self.split]
        if self.text:
            self._filter_invalid_text_rows()
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
            print_rank_zero(
                f"{self.split.upper()} MIMIC: {len(self._flat_image_paths):,} Images"
            )
        elif self.expand_samples:
            print_rank_zero(
                f"{self.split.upper()} MIMIC: {len(self._expanded_data):,} Expanded Samples (from {len(self._rows_image_paths):,} Studies)"
            )
        else:
            print_rank_zero(
                f"{self.split.upper()} MIMIC: {len(self._rows_image_paths):,} Studies"
            )

    def _filter_invalid_text_rows(self):
        """
        重写过滤方法，检查 findings_section 或 impression_section 是否有效
        """
        if not getattr(self, "text", False):
            return
        if not hasattr(self, "data_list") or self.data_list is None:
            return
        
        # 检查 findings_section 或 impression_section 是否至少有一个有效
        def has_valid_text(row):
            findings = row.get("findings_section", "")
            impression = row.get("impression_section", "")
            return (findings and self._is_valid_text(findings)) or (impression and self._is_valid_text(impression))
        
        self.data_list = (
            self.data_list[
                self.data_list.apply(has_valid_text, axis=1)
            ]
            .reset_index(drop=True)
        )

    def _get_report_from_row(self, row):
        """
        从行数据中获取报告文本。
        优先使用 findings_section，如果为空则使用 impression_section。
        
        Returns:
            tuple: (report_text, is_impression)
                - report_text: 报告文本字符串
                - is_impression: 布尔值，True表示使用的是impression_section
        """
        findings = row.get("findings_section", "")
        if findings and self._is_valid_text(findings):
            return str(findings).strip(), False
        
        impression = row.get("impression_section", "")
        if impression and self._is_valid_text(impression):
            return str(impression).strip(), True
        
        return "", False

    def _split_impression_report(self, report, max_regions=4):
        """
        按数字前缀（如 "1. ", "2. "）拆分 impression 格式的报告。
        
        Args:
            report: Impression 报告文本
            max_regions: 最大区域数量
            
        Returns:
            list: 拆分后的文本区域列表
        """
        if not report or not isinstance(report, str):
            return []
        
        # 使用正则表达式匹配数字加点的模式（如 "1. ", "2. ", "10. "）
        # 模式：行首的数字，后跟点号和可选的空格
        pattern = r'^\d+\.\s*'
        
        # 按行分割
        lines = report.split('\n')
        regions = []
        current_region = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是新的数字前缀开头
            if re.match(pattern, line):
                # 如果当前区域有内容，先保存
                if current_region:
                    regions.append(' '.join(current_region).strip())
                    current_region = []
                # 开始新区域（去掉数字前缀）
                line_without_prefix = re.sub(pattern, '', line, count=1)
                current_region.append(line_without_prefix)
            else:
                # 继续当前区域
                if current_region:
                    current_region.append(line)
                else:
                    # 如果没有当前区域，创建一个（处理没有数字前缀的第一行）
                    current_region.append(line)
        
        # 添加最后一个区域
        if current_region:
            regions.append(' '.join(current_region).strip())
        
        # 如果按数字前缀拆分没有结果，尝试按句号拆分作为后备
        if not regions:
            regions = [
                section.strip()
                for section in report.split(".")
                if section.strip()
            ]
        
        return regions

    def _split_report_into_regions(self, report, max_regions=4, is_impression=False):
        """
        将报告拆分为多个区域。
        如果 is_impression=True，使用数字前缀拆分；否则使用句号拆分。
        
        Args:
            report: 报告文本
            max_regions: 最大区域数量
            is_impression: 是否为 impression 格式
            
        Returns:
            tuple: (text_regions, text_attention_mask)
        """
        if is_impression:
            # 使用数字前缀拆分 impression
            text_regions = self._split_impression_report(report, max_regions)
        else:
            # 使用句号拆分 findings
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



def build_mimic_dataloader(
    data_dir, split, img_size, batch_size, in_channels=1, text=False, max_views=4, max_regions=4, split_report=True, expand_samples=False
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

    dataset = MIMICDataset(
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
    data_dir = "${SMARTX_DATA_ROOT}/MIMIC-CXR"
    dataset = MIMICDataset(
        data_dir=data_dir,
        split="train",
        img_size=512,
        in_channels=1,
        transform=transforms.ToTensor(),
        text=True,
        max_views=4,
        max_regions=4,
    )
    print(len(dataset))
    print(dataset[0])
    print(dataset[0]["text_attention_mask"])
    print(dataset[0]["image"].shape)
    print(dataset[1]["image"].shape)
