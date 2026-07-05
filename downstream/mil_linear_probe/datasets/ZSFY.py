import os
import ast
from typing import List, Optional

import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from .base_dataset import BaseDataset, load_image_with_dynamic_range


class ZSFYDataset(BaseDataset):
    """
    ZSFY 数据集：
      - 根目录: {data_dir}（默认 ${SMARTX_DATA_ROOT}/ZSFY）
      - 图像目录: {data_dir}/images
      - 标注 CSV: {data_dir}/zhognshanfuyi_with_labels_qwen3-30b_single_region_updated.csv
      - CSV 中 `image_paths` 列给出病例目录的相对路径，例如:
            Chest/00123456
        实际图像为该目录下的所有 PNG/JPG 文件:
            {data_dir}/images/Chest/00123456/*.png(jpg)

    本数据集类同时支持：
      - text=False: 图像-only 模式（逐图索引，__getitem__ 返回单张图像）
      - text=True : 预留文本模式（当前 MIL 流程中未使用）

    对于 MIL，我们使用病例级信息：
      - data_list: 一行一个病例
      - _rows_image_paths: 每个病例对应的图像路径列表（病例目录下所有图像）
      - Region: 由 `image_paths` 的第一个路径段或 CSV 中列推断
    """

    @staticmethod
    def _split_zsfy_image_list(raw_rel: str) -> List[str]:
        """
        将 CSV 中 `image_paths` 字段解析为图片相对路径列表。
        当前 ZSFY 提供的是一个列表字符串，例如:
            "['zsfy/xxx.jpg', 'zsfy/yyy.jpg']"
        或逗号分隔的多路径。
        """
        raw = str(raw_rel).strip()
        if raw == "" or raw.lower() == "nan":
            return []

        # 优先按 Python list 字面量解析
        if raw.startswith("[") and raw.endswith("]"):
            try:
                lst = ast.literal_eval(raw)
                return [str(x).strip() for x in lst if str(x).strip()]
            except Exception:
                pass

        # 回退：按逗号分隔
        parts = [p.strip() for p in raw.replace("，", ",").split(",") if p.strip()]
        return parts

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
        image_path_col: str = "image_paths",
        csv_name: str = "zhognshanfuyi_with_labels_qwen3-30b_single_region_updated.csv",
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
            raise FileNotFoundError(f"ZSFY CSV 文件不存在: {csv_path}")

        df = pd.read_csv(csv_path)

        if image_path_col not in df.columns:
            raise ValueError(
                f"ZSFY CSV 中未找到 `{image_path_col}` 列，请检查文件格式。"
            )

        # region 列优先级：
        # 1) 显式传入 region_col
        # 2) 自动匹配常见列名
        # 3) 回退为从图片目录第一段推断
        resolved_region_col = region_col
        if resolved_region_col is None:
            for c in ["region", "Region", "REGION"]:
                if c in df.columns:
                    resolved_region_col = c
                    break

        # ZSFY 同样全部作为外部测试集使用，目前不区分 train/val/test 划分。
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
                desc=f"Indexing ZSFY {split} cases",
                unit="case",
                mininterval=1.0,
            )
        ):
            rel_list = self._split_zsfy_image_list(raw_rel)
            # region 优先使用 CSV 列；否则回退到路径第一段
            if resolved_region_col is not None:
                region = str(df.iloc[idx][resolved_region_col]).strip()
                if region == "" or region.lower() == "nan":
                    if rel_list:
                        first = str(rel_list[0]).strip().lstrip("/").replace("\\", "/")
                        parts = first.split("/")
                        region = parts[0] if len(parts) > 1 and parts[0] else "ZSFY"
                    else:
                        region = "ZSFY"
            else:
                if rel_list:
                    first = str(rel_list[0]).strip().lstrip("/").replace("\\", "/")
                    parts = first.split("/")
                    region = parts[0] if len(parts) > 1 and parts[0] else "ZSFY"
                else:
                    region = "ZSFY"
            regions.append(region)

            img_paths: List[str] = []

            # 当前 ZSFY 的 image_paths 字段是“多个文件路径”的列表字符串，
            # 每个元素都是一张图片的相对路径（通常以 'zsfy/' 开头）。
            for rel in rel_list:
                rel_norm = str(rel).strip().replace("\\", "/").lstrip("/")
                # 去掉前缀 zsfy/（若存在），再与 images 根目录拼接
                lower = rel_norm.lower()
                if lower.startswith("zsfy/"):
                    rel_norm = rel_norm.split("/", 1)[1]
                full_path = os.path.join(self.base_img_dir, rel_norm)
                img_paths.append(full_path)

            # 严格存在性过滤：只保留磁盘上确实存在的图片，避免 __getitem__ 阶段
            # 拿到不存在的路径再静默替换成其他图。
            existing_img_paths = [p for p in img_paths if os.path.isfile(p)]
            missing_for_sample = len(img_paths) - len(existing_img_paths)
            if missing_for_sample > 0:
                missing_image_samples += 1
                missing_image_total += missing_for_sample

            rows_image_paths.append(existing_img_paths)

        df["Region"] = regions

        if missing_image_samples > 0:
            print(
                f"  ZSFY: 有 {missing_image_samples} 个病例存在缺失图片，"
                f"共忽略 {missing_image_total} 张缺失图片。"
            )

        # 先剔除没有任何可用图片的空病例，保持 data_list 与 _rows_image_paths 一致。
        keep_indices = [i for i, paths in enumerate(rows_image_paths) if len(paths) > 0]
        if len(keep_indices) < len(rows_image_paths):
            dropped_empty = len(rows_image_paths) - len(keep_indices)
            print(
                f"  ZSFY: 丢弃 {dropped_empty} 个未解析到可用图片的病例 "
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
                    f"  ZSFY: 丢弃 {dropped} 个视角数大于 {self.max_views} 的病例 "
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
                f"{self.split.upper()} ZSFY: {len(self._rows_image_paths):,} Cases, "
                f"{len(self._flat_image_paths):,} Images"
            )
        else:
            print(f"{self.split.upper()} ZSFY: {len(self._rows_image_paths):,} Cases (text mode)")

    def __getitem__(self, idx):
        """
        与 BaseDataset 的文本模式保持兼容；
        当前特征提取和 MIL 都使用 text=False（图像-only 模式）。
        """
        if self.text:
            # 预留：如未来需要文本 + 图像联合建模，可在此扩展
            raise NotImplementedError("ZSFYDataset 暂未实现 text=True 模式。")

        # image-only 模式 → 返回单张图像及其路径
        # 严格模式：不允许在 idx 失败时跳到其他图（否则会造成 path<->feature 错位）。
        # __init__ 已经过滤掉不存在的文件；这里若加载失败，直接抛错。
        if idx < 0 or idx >= len(self._flat_image_paths):
            raise IndexError(f"ZSFY idx out of range: {idx} (size={len(self._flat_image_paths)})")
        image_path = self._flat_image_paths[idx]
        try:
            image = load_image_with_dynamic_range(
                image_path, in_chans=self.in_channels
            )
        except Exception as e:
            raise RuntimeError(
                f"ZSFY 图像加载失败（不再静默替换为其他图，请检查文件是否损坏或路径错误）: "
                f"{image_path}\n原始错误: {e}"
            ) from e
        image = self.transform(image)
        return {
            "image": image,
            "image_path": image_path,
            "region": (self._flat_regions[idx] if self._flat_regions is not None else "ZSFY"),
        }

