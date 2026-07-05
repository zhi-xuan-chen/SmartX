import os
from typing import List, Optional

import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from .base_dataset import BaseDataset, load_image_with_dynamic_range


class NYFSDataset(BaseDataset):
    """
    NYFS 数据集：
      - 图像目录: {data_dir}/png_images
      - 标注 CSV: {data_dir}/nyfs_prospective_with_labels_en_updated.csv
      - CSV 中 `图片路径` 列给出病例目录的相对路径，例如:
            掌指骨正斜位/1240762_7494088
        实际图像为该目录下的所有 PNG 文件:
            {data_dir}/png_images/掌指骨正斜位/1240762_7494088/*.png

    本数据集类同时支持：
      - text=False: 图像-only 模式（逐图索引，__getitem__ 返回单张图像）
      - text=True : 预留文本模式（当前 MIL 流程中未使用）

    对于 MIL，我们使用病例级信息：
      - data_list: 一行一个病例
      - _rows_image_paths: 每个病例对应的图像路径列表（病例目录下所有 PNG）
      - Region: 由 `图片路径` 的第一个路径段推断，例如 "掌指骨正斜位"
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        img_size: int,
        in_channels: int = 1,
        max_views: int = 1,
        text: bool = False,
        max_regions: int = 4,
        transform=None,
        split_report: bool = True,
        expand_samples: bool = False,
        filter_by_max_views: bool = False,
        region_col: Optional[str] = None,
        region_filter: Optional[str] = None,
        image_path_col: str = "图片路径",
        csv_name: str = "nyfs_prospective_with_labels_en_updated.csv",
    ):
        # MIL 中不需要展开样本，直接使用病例级
        super().__init__(img_size=img_size, expand_samples=expand_samples)

        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.in_channels = in_channels
        self.text = text
        self.max_views = max_views
        self.max_regions = max_regions
        self.split_report = split_report
        self.filter_by_max_views = filter_by_max_views

        # 与 PrivateDataset 对齐的属性命名，方便下游复用
        self.text_column_name = "Findings"
        self.base_img_dir = os.path.join(self.data_dir, "png_images")
        self.transform = transform or transforms.Compose([transforms.ToTensor()])

        self.data_list = pd.DataFrame()
        self._rows_image_paths: List[List[str]] = []
        self._flat_image_paths: List[str] = [] if not self.text else None
        self._flat_regions: List[str] = [] if not self.text else None

        csv_path = os.path.join(self.data_dir, csv_name)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"NYFS CSV 文件不存在: {csv_path}")

        df = pd.read_csv(csv_path)

        if image_path_col not in df.columns:
            raise ValueError(
                f"NYFS CSV 中未找到 `{image_path_col}` 列，请检查文件格式。"
            )

        # region 列优先级：
        # 1) 显式传入 region_col
        # 2) 自动匹配常见列名
        # 3) 回退为从图片路径第一段推断
        resolved_region_col = region_col
        if resolved_region_col is None:
            for c in ["region", "Region", "REGION"]:
                if c in df.columns:
                    resolved_region_col = c
                    break

        # 目前 NYFS 没有官方的 train/val/test 划分，这里简单地全部样本用于指定 split
        # 如需后续支持划分，可以在 CSV 中新增 split 列并在此处按列筛选

        image_rel_paths = df[image_path_col].astype(str).tolist()
        regions: List[str] = []
        rows_image_paths: List[List[str]] = []
        missing_image_samples = 0
        missing_image_total = 0
        skipped_no_dir = 0

        # 这里会做大量小目录访问（NAS 上可能很慢），加进度条避免看起来“卡死”
        for idx, rel in enumerate(
            tqdm(
                image_rel_paths,
                desc=f"Indexing NYFS {split} cases",
                unit="case",
                mininterval=1.0,
            )
        ):
            rel = rel.lstrip("/").strip()
            # region 优先使用 CSV 列；否则回退到路径第一段
            if resolved_region_col is not None:
                region = str(df.iloc[idx][resolved_region_col]).strip()
                if region == "" or region.lower() == "nan":
                    parts = rel.split("/")
                    region = parts[0] if len(parts) > 1 and parts[0] else "NYFS"
            else:
                parts = rel.split("/")
                region = parts[0] if len(parts) > 1 and parts[0] else "NYFS"
            regions.append(region)

            # 病例目录：{base_img_dir}/{图片路径}/，目录下所有 PNG 视为该病例的多视角图像
            case_dir = os.path.join(self.base_img_dir, rel)
            img_paths: List[str] = []

            if os.path.isdir(case_dir):
                # os.scandir 比 listdir + isfile 更快，尤其在网络盘上
                entries: List[str] = []
                try:
                    with os.scandir(case_dir) as it:
                        for e in it:
                            if e.is_file() and e.name.lower().endswith(".png"):
                                entries.append(e.path)
                except OSError:
                    entries = []
                if entries:
                    img_paths = sorted(entries)
            else:
                skipped_no_dir += 1

            # 严格存在性过滤：只保留磁盘上确实存在的 PNG
            existing_img_paths = [p for p in img_paths if os.path.isfile(p)]
            missing_for_sample = len(img_paths) - len(existing_img_paths)
            if missing_for_sample > 0:
                missing_image_samples += 1
                missing_image_total += missing_for_sample

            rows_image_paths.append(existing_img_paths)

        df["Region"] = regions

        # 一次性输出存在性统计（不逐条打印）
        if skipped_no_dir > 0:
            print(f"  NYFS: 有 {skipped_no_dir} 个病例的目录不存在（将作为空病例被丢弃）")
        if missing_image_samples > 0:
            print(
                f"  NYFS: 有 {missing_image_samples} 个病例存在缺失图片，"
                f"共忽略 {missing_image_total} 张缺失图片。"
            )

        # 先剔除没有任何可用 PNG 的空病例，避免它们继续留在 data_list 中导致 Case 数虚高，
        # 也避免在特征抽取阶段构造出不存在的路径而被静默替换。
        keep_indices = [i for i, paths in enumerate(rows_image_paths) if len(paths) > 0]
        if len(keep_indices) < len(rows_image_paths):
            dropped_empty = len(rows_image_paths) - len(keep_indices)
            print(
                f"  NYFS: 丢弃 {dropped_empty} 个未解析到可用 PNG 的病例 "
                f"(保留 {len(keep_indices)}/{len(rows_image_paths)})"
            )
            rows_image_paths = [rows_image_paths[i] for i in keep_indices]
            df = df.iloc[keep_indices].reset_index(drop=True)

        # 按 region 过滤（用于“每个 region 对应一个模型”测试）
        if region_filter is not None:
            keep_indices = df.index[df["Region"] == region_filter].tolist()
            df = df.iloc[keep_indices].reset_index(drop=True)
            rows_image_paths = [rows_image_paths[i] for i in keep_indices]

        # 可选：根据 max_views 过滤病例（只保留视角数 <= max_views 的病例）
        if self.filter_by_max_views and not self.text:
            keep_indices = [
                i for i, paths in enumerate(rows_image_paths) if len(paths) <= self.max_views
            ]
            if len(keep_indices) < len(rows_image_paths):
                dropped = len(rows_image_paths) - len(keep_indices)
                print(
                    f"  NYFS: 丢弃 {dropped} 个视角数大于 {self.max_views} 的病例 "
                    f"(保留 {len(keep_indices)}/{len(rows_image_paths)})"
                )
            rows_image_paths = [rows_image_paths[i] for i in keep_indices]
            df = df.iloc[keep_indices].reset_index(drop=True)

        self.data_list = df.reset_index(drop=True)
        self._rows_image_paths = rows_image_paths

        if not self.text:
            # 图像-only 模式：扁平化所有图像路径，用于逐图像迭代（特征提取）
            # 使用 data_list["Region"] 与过滤后的 _rows_image_paths 对齐（filter 后 regions 未同步，会错位）
            self._flat_image_paths = []
            self._flat_regions = []
            for r, paths in zip(self.data_list["Region"], self._rows_image_paths):
                for p in paths:
                    self._flat_image_paths.append(p)
                    self._flat_regions.append(r)
            print(
                f"{self.split.upper()} NYFS: {len(self._rows_image_paths):,} Cases, "
                f"{len(self._flat_image_paths):,} Images"
            )
        else:
            print(f"{self.split.upper()} NYFS: {len(self._rows_image_paths):,} Cases (text mode)")

    def __getitem__(self, idx):
        """
        与 BaseDataset 的文本模式保持兼容；
        当前特征提取和 MIL 都使用 text=False（图像-only 模式）。
        """
        if self.text:
            # 预留：如未来需要文本 + 图像联合建模，可在此扩展
            raise NotImplementedError("NYFSDataset 暂未实现 text=True 模式。")

        # image-only 模式 → 返回单张图像及其路径
        # 严格模式：不允许在 idx 失败时跳到其他图（否则会造成 path<->feature 错位）。
        # __init__ 已经过滤掉不存在的文件；这里若加载失败，直接抛错。
        if idx < 0 or idx >= len(self._flat_image_paths):
            raise IndexError(f"NYFS idx out of range: {idx} (size={len(self._flat_image_paths)})")
        image_path = self._flat_image_paths[idx]
        try:
            image = load_image_with_dynamic_range(
                image_path, in_chans=self.in_channels
            )
        except Exception as e:
            raise RuntimeError(
                f"NYFS 图像加载失败（不再静默替换为其他图，请检查文件是否损坏或路径错误）: "
                f"{image_path}\n原始错误: {e}"
            ) from e
        image = self.transform(image)
        return {
            "image": image,
            "image_path": image_path,
            "region": (self._flat_regions[idx] if self._flat_regions is not None else "NYFS"),
        }

