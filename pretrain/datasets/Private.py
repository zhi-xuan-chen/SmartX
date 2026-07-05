import os
import re

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import print_rank_zero
from .base_dataset import BaseDataset, collate_fn, load_image_with_dynamic_range


class PrivateDataset(BaseDataset):
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
        text=False,
        max_regions=4,
        transform=None,
        split_report=True,
        expand_samples=False,
        # 过滤每个 region CSV 中 Num_image 最高的 top pct（例如 0.05 表示过滤掉 top5%）
        # 设为 None/0 可关闭过滤
        filter_top_num_image_pct: float | None = None,
        max_num_image: int | None = 10,
        region=[
            "Chest",
            "Pelvis",
            "Wrist",
            "Metacarpal",
            "Lumbar_spine",
            "Cervical_spine",
            "Foot",
            "Ankle",
            "Abdominal",
            "Knee",
        ],
    ):
        super().__init__(img_size, expand_samples=expand_samples)
        self.text_column_name = "Translated_Findings"
        self.data_dir = data_dir
        self.img_size = img_size
        self.in_channels = in_channels
        self.transform = transform  # TODO Need to be fixed for text=True
        self.split = split
        self.text = text
        self.region = region
        self.max_views = max_views
        self.max_regions = max_regions
        self.split_report = split_report
        self.data_list = pd.DataFrame()
        img_dir_1024 = os.path.join(self.data_dir, "imgs-1024")
        self.base_img_dir = img_dir_1024 if os.path.isdir(img_dir_1024) else os.path.join(self.data_dir, "imgs-512")
        self._rows_image_paths = []
        self._flat_image_paths = [] if not self.text else None
        self._region_stats = {}  # Track stats per region
        self.filter_top_num_image_pct = filter_top_num_image_pct
        self.max_num_image = max_num_image
        print_rank_zero(f"{self.split.upper()} Private A: {len(self.region)} Regions")
        
        for region in self.region:
            csv_name = "Wrist_fixed.csv" if region == "Wrist" else f"{region}.csv"
            labels_csv = os.path.join(
                self.data_dir, "labels/XR_labels_step10", csv_name
            )
            # Try multiple encodings to handle different file encodings
            if not os.path.exists(labels_csv):
                print_rank_zero(f"WARNING: CSV file not found: {labels_csv}, skipping region {region}")
                continue
            
            # Try reading with different encodings
            df = None
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'iso-8859-1']
            for encoding in encodings:
                try:
                    df = pd.read_csv(labels_csv, encoding=encoding)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
                except Exception as e:
                    print_rank_zero(f"Error reading {labels_csv} with {encoding}: {e}")
                    continue
            
            if df is None:
                raise ValueError(f"Failed to read CSV file {labels_csv} with any encoding. File may be corrupted.")
            df = df[df["split"] == self.split]

            # --- 过滤：按每个 region 的 Num_image 列最大张数硬性阈值 (Hard Threshold) ---
            if self.max_num_image is not None and self.max_num_image > 0:
                if "Num_image" in df.columns:
                    num_img = pd.to_numeric(df["Num_image"], errors="coerce")
                    valid = num_img.notna()
                    if valid.any():
                        before = len(df)
                        keep_mask = (num_img <= self.max_num_image) | num_img.isna()
                        df = df[keep_mask]
                        removed = before - len(df)
                        if removed > 0:
                            print_rank_zero(
                                f"  {region}: filter Num_image > {self.max_num_image} removed {removed:,}/{before:,} cases"
                            )

            # --- 过滤：按每个 region 的 Num_image 列移除 top pct case ---
            # 注意：这里在解析 Image_Paths 之前过滤，保证后续路径列表与 df 行对齐
            pct = self.filter_top_num_image_pct
            if pct is not None and pct > 0:
                if "Num_image" in df.columns:
                    num_img = pd.to_numeric(df["Num_image"], errors="coerce")
                    valid = num_img.notna()
                    if valid.any():
                        q = float(1.0 - float(pct))
                        thr = float(num_img[valid].quantile(q))
                        before = len(df)
                        keep_mask = (num_img < thr) | num_img.isna()
                        df = df[keep_mask]
                        removed = before - len(df)
                        print_rank_zero(
                            f"  {region}: filter Num_image top{int(pct*100)}% (q{q:.2f} thr={thr:.3f}) removed {removed:,}/{before:,} cases"
                        )
                    else:
                        print_rank_zero(
                            f"  {region}: Num_image 全为空/不可解析，跳过 top{int(pct*100)}% 过滤"
                        )
                else:
                    print_rank_zero(
                        f"  {region}: 缺少 Num_image 列，跳过 top{int(pct*100)}% 过滤"
                    )

            if self.text:
                # 检查 Translated_Findings 或 Translated_Diagnosis 是否至少有一个有效
                def has_valid_text(row):
                    findings = row.get("Translated_Findings", "")
                    diagnosis = row.get("Translated_Diagnosis", "")
                    return (findings and self._is_valid_text(findings)) or (diagnosis and self._is_valid_text(diagnosis))
                
                if "Translated_Findings" in df.columns or "Translated_Diagnosis" in df.columns:
                    df = df[df.apply(has_valid_text, axis=1)]

            # Parse and fix paths for this region's data
            region_image_paths = [
                self._parse_and_fix_paths(x) for x in df["Image_Paths"].tolist()
            ]
            self._rows_image_paths.extend(region_image_paths)

            # Store region stats
            self._region_stats[region] = {
                'studies': len(df),
                'image_paths': region_image_paths,
                'total_images': sum(len(paths) for paths in region_image_paths)
            }

            if self.text:
                # Text mode: count rows
                print_rank_zero(f"  {region}: {len(df):,} Studies")
            else:
                # Image-only mode: count and flatten images
                region_image_count = sum(len(paths) for paths in region_image_paths)
                for paths in region_image_paths:
                    self._flat_image_paths.extend(paths)
                print_rank_zero(
                    f"  {region}: {region_image_count:,} Images"
                )

            self.data_list = pd.concat([self.data_list, df])

        self.data_list = self.data_list.reset_index(drop=True)
        
        # Create expanded data if needed
        self._create_expanded_data()
        
        # Print dataset statistics
        if not self.text:
            # Flatten for image-only mode
            self._flat_image_paths = []
            for paths in self._rows_image_paths:
                self._flat_image_paths.extend(paths)
            print_rank_zero(
                f"{self.split.upper()} Private: {len(self._flat_image_paths):,} Images"
            )
        elif self.expand_samples:
            # Print per-region expanded statistics
            print_rank_zero(f"{self.split.upper()} Private: {len(self._expanded_data):,} Expanded Samples (from {len(self._rows_image_paths):,} Studies)")
            
            # Print per-region expanded counts using pre-calculated data
            for region in self.region:
                region_studies = self._region_stats[region]['studies']
                region_expanded_count = self._region_stats[region]['total_images']  # Already calculated!
                
                print_rank_zero(f"  {region}: {region_expanded_count:,} Expanded Samples (from {region_studies:,} Studies)")
        else:
            print_rank_zero(
                f"{self.split.upper()} Private: {len(self._rows_image_paths):,} Studies"
            )

    def _get_report_from_row(self, row):
        """
        从行数据中获取报告文本。
        优先使用 Translated_Findings，如果为空则使用 Translated_Diagnosis。
        
        Returns:
            tuple: (report_text, is_diagnosis)
                - report_text: 报告文本字符串
                - is_diagnosis: 布尔值，True表示使用的是Translated_Diagnosis
        """
        findings = row.get("Translated_Findings", "")
        if findings and self._is_valid_text(findings):
            return str(findings).strip(), False
        
        diagnosis = row.get("Translated_Diagnosis", "")
        if diagnosis and self._is_valid_text(diagnosis):
            return str(diagnosis).strip(), True
        
        return "", False

    def _split_report_into_regions(self, report, max_regions=4, is_diagnosis=False):
        """
        将报告拆分为多个区域。
        对于 diagnosis 和 findings，都使用句号拆分（diagnosis 不需要数字前缀拆分）。
        
        Args:
            report: 报告文本
            max_regions: 最大区域数量
            is_diagnosis: 是否为 diagnosis 格式（目前与 findings 使用相同的拆分方式）
            
        Returns:
            tuple: (text_regions, text_attention_mask)
        """
        # 对于 findings 和 diagnosis，都使用句号拆分
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
            report, is_diagnosis = self._get_report_from_row(row)
            max_retries = 10
            next_idx = idx
            retries = 0
            while report == "" and retries < max_retries:
                next_idx += 1
                if next_idx >= len(self._expanded_data):
                    raise IndexError(f"No valid report found after index {idx}")
                expanded_item = self._expanded_data[next_idx]
                row = expanded_item['row_data']
                report, is_diagnosis = self._get_report_from_row(row)
                retries += 1
            if report == "":
                raise ValueError(f"Could not find a valid report after {max_retries} retries starting from index {idx}")
            
            region_reports, text_attention_mask = self._split_report_into_regions(
                report=report, max_regions=self.max_regions, is_diagnosis=is_diagnosis
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
            report, is_diagnosis = self._get_report_from_row(row)
            max_retries = 10  # Prevent infinite loop; adjust as needed
            next_idx = idx
            retries = 0
            while report == "" and retries < max_retries:
                next_idx += 1
                if next_idx >= len(self.data_list):
                    raise IndexError(f"No valid report found after index {idx}")
                row = self.data_list.iloc[next_idx]
                image_paths = self._rows_image_paths[next_idx]
                report, is_diagnosis = self._get_report_from_row(row)
                retries += 1
            if report == "":
                raise ValueError(f"Could not find a valid report after {max_retries} retries starting from index {idx}")
            if self.split_report == True:
                region_reports, text_attention_mask = self._split_report_into_regions(
                    report=report, max_regions=self.max_regions, is_diagnosis=is_diagnosis
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


def build_private_dataloader(
    data_dir,
    split,
    img_size,
    batch_size,
    text=False,
    in_channels=1,
    region="Chest",
    max_views=2,
    max_regions=4,
    split_report=True,
    expand_samples=False,
    filter_top_num_image_pct: float | None = None,
    max_num_image: int | None = 10,
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

    dataset = PrivateDataset(
        data_dir=data_dir,
        split=split,
        img_size=img_size,
        in_channels=in_channels,
        transform=transform,
        region=region,
        text=text,
        max_views=max_views,
        max_regions=max_regions,
        split_report=split_report,
        expand_samples=expand_samples,
        filter_top_num_image_pct=filter_top_num_image_pct,
        max_num_image=max_num_image,
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
    data_dir = "${SMARTX_DATA_ROOT}/Private_a"
    dataset = PrivateDataset(
        data_dir=data_dir,
        split="train",
        region=["Abdominal", "Knee"],
        img_size=512,
        in_channels=1,
        transform=transforms.ToTensor(),
        text=True,
        max_views=2,
        max_regions=4,
    )
    print(len(dataset))
    print(dataset[0]["text"])
    print(dataset[0]["text_attention_mask"])
    print(dataset[0]["image"].shape)
    print(dataset[0]["image_attention_mask"].shape)
