import os
from typing import List, Optional

import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from .base_dataset import BaseDataset, load_image_with_dynamic_range


class NCFYDataset(BaseDataset):
    """
    NCFY 数据集：
      - 根目录: {data_dir}（默认 ${SMARTX_DATA_ROOT}/NCFY）
      - 图像目录: {data_dir}/images
      - 标注 CSV: {data_dir}/ncfy_prospective_en_with_labels.csv
      - CSV 中 `image-paths` 列给出病例目录或图像相对路径，例如:
            Chest/00123456
        实际图像为该目录下的所有 PNG/JPG 文件:
            {data_dir}/images/Chest/00123456/*.png(jpg)

    本数据集类同时支持：
      - text=False: 图像-only 模式（逐图索引，__getitem__ 返回单张图像）
      - text=True : 预留文本模式（当前 MIL 流程中未使用）

    对于 MIL，我们使用病例级信息：
      - data_list: 一行一个病例
      - _rows_image_paths: 每个病例对应的图像路径列表（病例目录下所有图像）
      - Region: 由 `image-paths` 的第一个路径段或 CSV 中列推断
    """

    @staticmethod
    def _split_ncfy_raw_paths(raw_rel: str) -> List[str]:
        """将 CSV 单元格中的多路径（逗号分隔）拆成列表。"""
        raw = str(raw_rel).strip()
        if raw == "" or raw.lower() == "nan":
            return []
        return [c.strip() for c in raw.replace("，", ",").split(",") if c.strip()]

    @staticmethod
    def _normalize_ncfy_rel_path(single_rel: str) -> str:
        """
        规范化 NCFY 的单个路径：
        1) 统一路径分隔符为 /
        2) 去掉“南昌大学第一附属医院”及其之前的路径段
        3) 若是绝对路径且包含 /images/，仅保留其后的相对部分
        4) 去掉开头的 images/（避免与 base_img_dir 重复）
        5) 若以 .dcm 结尾，映射为 .png
        """
        rel = str(single_rel).strip().replace("\\", "/").lstrip("/")

        # 绝对路径时，若包含 /images/，只保留其后的路径
        marker = "/images/"
        rel_lower = rel.lower()
        marker_idx = rel_lower.find(marker)
        if marker_idx != -1:
            rel = rel[marker_idx + len(marker) :]

        parts = [p for p in rel.split("/") if p not in ("", ".")]

        hospital_token = "南昌大学第一附属医院"
        if hospital_token in parts:
            parts = parts[parts.index(hospital_token) + 1 :]

        if parts and parts[0].lower() == "images":
            parts = parts[1:]

        rel = "/".join(parts)
        if rel.lower().endswith(".dcm"):
            rel = os.path.splitext(rel)[0] + ".png"
        return rel

    def _resolve_single_rel_to_paths(self, rel: str) -> List[str]:
        """严格按 CSV 路径解析：仅作为图片文件路径处理。"""
        if not rel:
            return []
        # 文件路径：严格使用 CSV 指定路径（仅经过显式的 .dcm -> .png 规范化）
        return [os.path.join(self.base_img_dir, rel)]

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
        image_path_col: str = "image-paths",
        csv_name: str = "ncfy_prospective_en_with_labels.csv",
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
        self.base_img_dir = os.path.join(self.data_dir, "images")
        self.transform = transform or transforms.Compose([transforms.ToTensor()])

        self.data_list = pd.DataFrame()
        self._rows_image_paths: List[List[str]] = []
        self._flat_image_paths: List[str] = [] if not self.text else None
        self._flat_regions: List[str] = [] if not self.text else None

        csv_path = os.path.join(self.data_dir, csv_name)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"NCFY CSV 文件不存在: {csv_path}")

        df = pd.read_csv(csv_path)

        if image_path_col not in df.columns:
            raise ValueError(
                f"NCFY CSV 中未找到 `{image_path_col}` 列，请检查文件格式。"
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

        # NCFY 同样全部作为外部测试集使用，目前不区分 train/val/test 划分。
        # 如需后续支持划分，可以在 CSV 中新增 split 列并在此处按列筛选

        image_rel_paths = df[image_path_col].astype(str).tolist()
        regions: List[str] = []
        rows_image_paths: List[List[str]] = []
        missing_image_samples = 0
        missing_image_total = 0

        # 这里会做大量小目录访问（NAS 上可能很慢），加进度条避免看起来“卡死”
        for idx, raw_rel in enumerate(
            tqdm(
                image_rel_paths,
                desc=f"Indexing NCFY {split} cases",
                unit="case",
                mininterval=1.0,
            )
        ):
            raw_candidates = self._split_ncfy_raw_paths(raw_rel)
            if not raw_candidates:
                raw_candidates = [raw_rel]

            rel_candidates: List[str] = []
            for raw_path in raw_candidates:
                normalized = self._normalize_ncfy_rel_path(raw_path)
                if normalized:
                    rel_candidates.append(normalized)

            if not rel_candidates:
                rel_candidates = [""]

            # region 优先使用 CSV 列；否则回退到路径第一段
            if resolved_region_col is not None:
                region = str(df.iloc[idx][resolved_region_col]).strip()
                if region == "" or region.lower() == "nan":
                    parts = rel_candidates[0].split("/")
                    region = parts[0] if len(parts) > 1 and parts[0] else "NCFY"
            else:
                parts = rel_candidates[0].split("/")
                region = parts[0] if len(parts) > 1 and parts[0] else "NCFY"
            regions.append(region)

            # 同一样本可包含多个路径（逗号分隔），逐路径解析后合并去重
            img_paths: List[str] = []
            seen = set()
            for rel in rel_candidates:
                for p in self._resolve_single_rel_to_paths(rel):
                    if p not in seen:
                        seen.add(p)
                        img_paths.append(p)

            existing_img_paths = [p for p in img_paths if os.path.exists(p)]
            missing_for_sample = len(img_paths) - len(existing_img_paths)
            if missing_for_sample > 0:
                missing_image_samples += 1
                missing_image_total += missing_for_sample

            # 允许部分缺失：只保留实际存在的图片。
            # 但如果该样本一张都找不到，直接报错终止。
            if not existing_img_paths:
                raise FileNotFoundError(
                    "NCFY 样本无可用图片，停止运行。\n"
                    f"row_index={idx}\n"
                    f"raw_image_paths={raw_rel}\n"
                    f"normalized_paths={rel_candidates}\n"
                    f"resolved_paths={img_paths}"
                )

            rows_image_paths.append(existing_img_paths)

        if missing_image_samples > 0:
            print(
                f"  NCFY: 有 {missing_image_samples} 个样本存在缺失图片，"
                f"共忽略 {missing_image_total} 张缺失图片。"
            )

        df["Region"] = regions

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
                    f"  NCFY: 丢弃 {dropped} 个视角数大于 {self.max_views} 的病例 "
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
                f"{self.split.upper()} NCFY: {len(self._rows_image_paths):,} Cases, "
                f"{len(self._flat_image_paths):,} Images"
            )
        else:
            print(f"{self.split.upper()} NCFY: {len(self._rows_image_paths):,} Cases (text mode)")

    def __getitem__(self, idx):
        """
        与 BaseDataset 的文本模式保持兼容；
        当前特征提取和 MIL 都使用 text=False（图像-only 模式）。
        """
        if self.text:
            # 预留：如未来需要文本 + 图像联合建模，可在此扩展
            raise NotImplementedError("NCFYDataset 暂未实现 text=True 模式。")

        # image-only 模式 → 返回单张图像及其路径
        # 严格模式：不允许在 idx 失败时跳到其他图（否则会造成 path<->feature 错位）。
        # __init__ 已经过滤掉不存在的文件；这里若加载失败，直接抛错。
        if idx < 0 or idx >= len(self._flat_image_paths):
            raise IndexError(f"NCFY idx out of range: {idx} (size={len(self._flat_image_paths)})")
        image_path = self._flat_image_paths[idx]
        try:
            image = load_image_with_dynamic_range(
                image_path, in_chans=self.in_channels
            )
        except Exception as e:
            raise RuntimeError(
                f"NCFY 图像加载失败（不再静默替换为其他图，请检查文件是否损坏或路径错误）: "
                f"{image_path}\n原始错误: {e}"
            ) from e
        image = self.transform(image)
        return {
            "image": image,
            "image_path": image_path,
            "region": (self._flat_regions[idx] if self._flat_regions is not None else "NCFY"),
        }

