import os

import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from .base_dataset import BaseDataset, collate_fn


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
        expand_samples=False,  # Deprecated, kept for compatibility but not used
        filter_by_max_views: bool = False,
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
        # MIL classification: don't expand samples, work with cases
        super().__init__(img_size, expand_samples=False)
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
        self.base_img_dir = os.path.join(self.data_dir, "imgs-1024")
        self._rows_image_paths = []
        self._flat_image_paths = [] if not self.text else None
        self._region_stats = {}  # Track stats per region
        self.filter_by_max_views = filter_by_max_views
        print(f"{self.split.upper()} Private A: {len(self.region)} Regions")
        
        # Disease labels (last 46 columns in CSV)
        DISEASE_LIST = [
            "退行性变", "主动脉硬化", "骨质疏松", "椎间盘病变", "颈椎病", "椎间隙狭窄", "椎体滑脱",
            "脊柱侧凸", "骨折", "终板炎", "关节炎", "强直性脊柱炎", "先天发育异常", "钙化", "骨质增生",
            "峡部裂", "关节脱位", "肺炎", "纤维灶", "心影增大", "肺水肿", "肺实变", "肺不张", 
            "气胸", "胸膜增厚", "胸腔积液", "纵隔增宽", "肾结石", "尿路结石", "肠梗阻", 
            "盆腔静脉石", "隐裂", "骶化", "软组织损伤", "骨软骨瘤", "退行性骨关节病", "副骨",
            "踇外翻", "扁平足", "骨刺", "骨化", "颈椎失稳", "颈椎反弓", "髋关节撞击综合征",
            "金属异物", "固定术后",
        ]
        
        for region in self.region:
            csv_name = "Wrist_fixed.csv" if region == "Wrist" else f"{region}.csv"
            # Try step11 first (with disease labels), fallback to step10
            labels_csv = os.path.join(
                self.data_dir, "labels/XR_labels_step11", csv_name
            )
            if not os.path.exists(labels_csv):
                labels_csv = os.path.join(
                    self.data_dir, "labels/XR_labels_step10", csv_name
                )
            
            # Try multiple encodings to handle different file encodings
            if not os.path.exists(labels_csv):
                print(f"WARNING: CSV file not found: {labels_csv}, skipping region {region}")
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
                    print(f"Error reading {labels_csv} with {encoding}: {e}")
                    continue
            
            if df is None:
                raise ValueError(f"Failed to read CSV file {labels_csv} with any encoding. File may be corrupted.")
            df = df[df["split"] == self.split]
            initial_count = len(df)
            
            # Filter based on disease labels: keep only rows where at least one disease label is not NA
            # Check if disease columns exist
            disease_cols = [col for col in DISEASE_LIST if col in df.columns]
            if len(disease_cols) > 0:
                # Filter: keep rows where at least one disease label is not NA
                has_label = df[disease_cols].notna().any(axis=1)
                df = df[has_label].reset_index(drop=True)
                print(f"  {region}: Filtered to {len(df)} cases with disease labels (from {initial_count} total)")

            # 标记所属部位，便于后续按 region 统计与保存结果
            df["Region"] = region
            
            if self.text and self.text_column_name in df.columns:
                df = df[df[self.text_column_name].apply(self._is_valid_text)]

            # Parse and fix paths for this region's data
            region_image_paths = [
                self._parse_and_fix_paths(x) for x in df["Image_Paths"].tolist()
            ]

            # 可选：根据 max_views 过滤病例（只保留视角数 <= max_views 的病例）
            if self.filter_by_max_views and not self.text:
                keep_indices = [
                    i for i, paths in enumerate(region_image_paths) if len(paths) <= self.max_views
                ]
                if len(keep_indices) < len(region_image_paths):
                    dropped = len(region_image_paths) - len(keep_indices)
                    print(
                        f"  {region}: 丢弃 {dropped} 个视角数大于 {self.max_views} 的病例 "
                        f"(保留 {len(keep_indices)}/{len(region_image_paths)})"
                    )
                region_image_paths = [region_image_paths[i] for i in keep_indices]
                df = df.iloc[keep_indices].reset_index(drop=True)

            self._rows_image_paths.extend(region_image_paths)

            # Store region stats
            self._region_stats[region] = {
                'studies': len(df),
                'image_paths': region_image_paths,
                'total_images': sum(len(paths) for paths in region_image_paths)
            }

            if self.text:
                # Text mode: count rows
                print(f"  {region}: {len(df):,} Studies")
            else:
                # Image-only mode: count and flatten images
                region_image_count = sum(len(paths) for paths in region_image_paths)
                for paths in region_image_paths:
                    self._flat_image_paths.extend(paths)
                print(f"  {region}: {region_image_count:,} Images")

            self.data_list = pd.concat([self.data_list, df])

        self.data_list = self.data_list.reset_index(drop=True)
        
        # MIL classification: don't expand samples, work with cases
        # Each case can have multiple images, but we don't flatten them
        
        # Print dataset statistics
        if not self.text:
            # For image-only mode in MIL, we still need to flatten for feature extraction
            # But the dataset should return case-level information
            self._flat_image_paths = []
            for paths in self._rows_image_paths:
                self._flat_image_paths.extend(paths)
            print(f"{self.split.upper()} Private: {len(self._rows_image_paths):,} Cases, {len(self._flat_image_paths):,} Images")
        else:
            # Text mode: case-based MIL
            print(f"{self.split.upper()} Private: {len(self._rows_image_paths):,} Cases (MIL mode)")


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
    normalization="our_ark_base",
    filter_by_max_views: bool = False,
    preprocess: str = "default",
    shuffle=None,
    num_workers=8,
    prefetch_factor: int = 4,
):
    # 延后导入，避免循环依赖
    from .utils import get_mil_transforms

    transform = get_mil_transforms(
        split=split,
        img_size=img_size,
        in_channels=in_channels,
        normalization=normalization,
        preprocess=preprocess,
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
        filter_by_max_views=filter_by_max_views,
    )
    if shuffle is None:
        shuffle = split == "train"

    loader_kwargs = dict(
        dataset=dataset,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        batch_size=batch_size,
        collate_fn=collate_fn if text else None,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(**loader_kwargs)
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
