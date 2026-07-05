import os
from typing import List, Optional

import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from .base_dataset import BaseDataset, load_image_with_dynamic_range


class ZSFSDataset(BaseDataset):
    """
    ZSFS 数据集：
      - 根目录: {data_dir}（默认 ${SMARTX_DATA_ROOT}/ZSFS）
      - 图像目录: {data_dir}/png_images
      - 标注 CSV: {data_dir}/zsfs_prospective_with_labels_en.csv
      - CSV 中 `图片目录` 列给出病例目录的相对路径，例如:
            掌指骨正斜位/1240762_7494088
        实际图像为该目录下的所有 PNG 文件:
            {data_dir}/png_images/掌指骨正斜位/1240762_7494088/*.png

    本数据集类同时支持：
      - text=False: 图像-only 模式（逐图索引，__getitem__ 返回单张图像）
      - text=True : 预留文本模式（当前 MIL 流程中未使用）

    对于 MIL，我们使用病例级信息：
      - data_list: 一行一个病例
      - _rows_image_paths: 每个病例对应的图像路径列表（病例目录下所有 PNG）
      - Region: 由 `图片目录` 的第一个路径段或 CSV 中列推断
    """

    @staticmethod
    def _normalize_zsfs_rel_path(raw_rel: str) -> str:
        """
        规范化 ZSFS CSV 中的相对路径：
        1) 统一路径分隔符为 /
        2) 去掉开头的 png_images/（避免与 base_img_dir 重复）
        3) 去掉开头的 / 与空白
        """
        rel = str(raw_rel).strip().replace("\\", "/").lstrip("/")
        # 如果包含 /png_images/，只保留其后的部分
        marker = "/png_images/"
        rel_lower = rel.lower()
        idx = rel_lower.find(marker)
        if idx != -1:
            rel = rel[idx + len(marker) :]
        # 如果以 png_images/ 开头，也去掉
        if rel.lower().startswith("png_images/"):
            rel = rel[len("png_images/") :]
        return rel.strip("/")

    @staticmethod
    def _collect_case_png_paths(case_dir: str) -> List[str]:
        """
        收集病例目录中的 PNG：
        1) 先尝试当前目录下的一层文件（快）；
        2) 若为空，再递归子目录（兼容如 ExamImage 这类嵌套结构）。
        """
        entries: List[str] = []
        try:
            with os.scandir(case_dir) as it:
                for e in it:
                    if e.is_file() and e.name.lower().endswith(".png"):
                        entries.append(e.path)
        except OSError:
            entries = []
        if entries:
            return sorted(entries)

        recursive_entries: List[str] = []
        try:
            for root, _, files in os.walk(case_dir):
                for name in files:
                    if name.lower().endswith(".png"):
                        recursive_entries.append(os.path.join(root, name))
        except OSError:
            recursive_entries = []
        return sorted(recursive_entries)

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
        image_path_col: str = "图片目录",
        csv_name: str = "zsfs_retrospective_with_labels_en_dedup_updated.csv",
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
        # 仅 ZSFS 在 MIL 读特征阶段需要严格按 CSV 逻辑路径定位 feature。
        self.use_feature_image_paths_for_mil = True

        self.data_list = pd.DataFrame()
        self._rows_image_paths: List[List[str]] = []
        # 与 _rows_image_paths 对齐的“逻辑路径”（基于 CSV 的相对路径，而非真实磁盘目录），
        # 仅用于特征保存时构造特征文件路径，不参与实际图像加载。
        self._rows_feature_image_paths: List[List[str]] = []
        self._flat_image_paths: List[str] = [] if not self.text else None
        self._flat_feature_image_paths: List[str] = [] if not self.text else None
        self._flat_regions: List[str] = [] if not self.text else None

        csv_path = os.path.join(self.data_dir, csv_name)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"ZSFS CSV 文件不存在: {csv_path}")

        df = pd.read_csv(csv_path)

        if image_path_col not in df.columns:
            raise ValueError(
                f"ZSFS CSV 中未找到 `{image_path_col}` 列，请检查文件格式。"
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

        # ZSFS 同样全部作为外部测试集使用，目前不区分 train/val/test 划分。
        # 如需后续支持划分，可以在 CSV 中新增 split 列并在此处按列筛选

        image_rel_paths = df[image_path_col].astype(str).tolist()
        regions: List[str] = []
        rows_image_paths: List[List[str]] = []
        rows_feature_image_paths: List[List[str]] = []
        # 跳过统计（避免逐条打印刷屏）
        skipped_multi_prefix = 0
        skipped_unresolved = 0
        # 命中统计：用于评估 CSV 与磁盘路径系统的一致性
        hit_direct_dir = 0       # 直接拼出的目录就存在
        hit_strip_ext = 0        # CSV 写了带扩展名的"文件路径"，去扩展名后命中目录
        hit_prefix_match = 0     # 走"姓名+日期"前缀匹配 fallback 命中

        # 这里会做大量小目录访问（NAS 上可能很慢），加进度条避免看起来“卡死”
        for idx, rel in enumerate(
            tqdm(
                image_rel_paths,
                desc=f"Indexing ZSFS {split} cases",
                unit="case",
                mininterval=1.0,
            )
        ):
            rel = self._normalize_zsfs_rel_path(rel)
            # region 优先使用 CSV 列；否则回退到路径第一段
            if resolved_region_col is not None:
                region = str(df.iloc[idx][resolved_region_col]).strip()
                if region == "" or region.lower() == "nan":
                    parts = rel.split("/")
                    region = parts[0] if len(parts) > 1 and parts[0] else "ZSFS"
            else:
                parts = rel.split("/")
                region = parts[0] if len(parts) > 1 and parts[0] else "ZSFS"
            regions.append(region)

            # 病例目录：{base_img_dir}/{图片路径}/，目录下所有 PNG 视为该病例的多视角图像。
            # 注意：CSV 中的 `图片目录` 有时会包含 png_images/ 前缀，或写成具体文件路径；
            # 这里统一先用规范化后的 rel 构造病例目录。
            case_dir = os.path.join(self.base_img_dir, rel)
            img_paths: List[str] = []
            feature_img_paths: List[str] = []
            skip_case = False

            if os.path.isdir(case_dir):
                entries = self._collect_case_png_paths(case_dir)
                if entries:
                    img_paths = entries
                    hit_direct_dir += 1

            # 若直接拼出的目录不存在，尝试将 rel 视为“带扩展名的文件路径”，
            # 使用去掉扩展名后的路径作为病例目录。
            if not img_paths and not os.path.isdir(case_dir):
                rel_no_ext, rel_ext = os.path.splitext(rel)
                if rel_ext:
                    alt_case_dir = os.path.join(self.base_img_dir, rel_no_ext)
                    if os.path.isdir(alt_case_dir):
                        entries = self._collect_case_png_paths(alt_case_dir)
                        if entries:
                            img_paths = entries
                            hit_strip_ext += 1

            # 若目录不存在或没有 PNG，进一步尝试“去掉末尾 ID 后的前缀匹配”逻辑：
            #   例如 CSV 给出:
            #       png_images/颈椎正侧位25年/20250623_梁幸梅_4189200
            #   若该目录不存在，则在
            #       png_images/颈椎正侧位25年/
            #   下查找所有以
            #       20250623_梁幸梅
            #   开头的子目录；如果恰好找到 1 个，则使用该目录作为病例目录；
            #   如果找到多个，则报错退出，方便人工检查。
            if not img_paths and not os.path.isdir(case_dir):
                parts = rel.split("/")
                if parts:
                    last_seg = parts[-1]
                    if "_" in last_seg:
                        # 去掉最后一个 '_' 及其之后的 ID
                        prefix_last = last_seg.rsplit("_", 1)[0]
                        prefix_rel = "/".join(parts[:-1] + [prefix_last])
                        prefix_abs = os.path.join(self.base_img_dir, prefix_rel)
                        parent_dir = os.path.dirname(prefix_abs)
                        base_prefix = os.path.basename(prefix_abs)

                        if os.path.isdir(parent_dir):
                            matched_dirs: List[str] = []
                            try:
                                with os.scandir(parent_dir) as it:
                                    for e in it:
                                        if e.is_dir() and e.name.startswith(base_prefix):
                                            matched_dirs.append(e.path)
                            except OSError:
                                matched_dirs = []

                            if len(matched_dirs) > 1:
                                # 多个候选目录：记录并跳过该病例（不加载任何图像，不保存任何特征）
                                skipped_multi_prefix += 1
                                skip_case = True
                            elif len(matched_dirs) == 1:
                                # 使用前缀唯一匹配到的目录作为病例目录
                                entries = self._collect_case_png_paths(matched_dirs[0])
                                if entries:
                                    img_paths = entries
                                    hit_prefix_match += 1

            # 为该病例构造“基于 CSV 的逻辑图像路径”，仅用于特征保存：
            #   逻辑路径形式：{base_img_dir}/{规范化后的 CSV 相对路径}/{文件名}
            # 即使物理目录是通过前缀匹配到的其他目录，这里也仍然使用 CSV 提供的路径前缀。
            if img_paths and not skip_case:
                for p in img_paths:
                    fname = os.path.basename(p)
                    feature_img_paths.append(os.path.join(self.base_img_dir, rel, fname))

            # 若目录与前缀匹配后仍没有可用 PNG，则直接跳过该病例。
            # 不再做单文件 fallback（{base_img_dir}/{rel}.png），避免错误路径混入。
            if not img_paths and not skip_case:
                skipped_unresolved += 1

            rows_image_paths.append(img_paths)
            rows_feature_image_paths.append(feature_img_paths)

        df["Region"] = regions

        # 先剔除没有任何可用 PNG 的空病例，避免它们继续留在 data_list 中导致 Case 数虚高。
        keep_indices = [i for i, paths in enumerate(rows_image_paths) if len(paths) > 0]
        if len(keep_indices) < len(rows_image_paths):
            dropped_empty = len(rows_image_paths) - len(keep_indices)
            print(
                f"  ZSFS: 丢弃 {dropped_empty} 个未解析到可用 PNG 的病例 "
                f"(保留 {len(keep_indices)}/{len(rows_image_paths)})"
            )
            rows_image_paths = [rows_image_paths[i] for i in keep_indices]
            rows_feature_image_paths = [rows_feature_image_paths[i] for i in keep_indices]
            df = df.iloc[keep_indices].reset_index(drop=True)

        # 按 region 过滤（用于“每个 region 对应一个模型”测试）
        if region_filter is not None:
            keep_indices = df.index[df["Region"] == region_filter].tolist()
            df = df.iloc[keep_indices].reset_index(drop=True)
            rows_image_paths = [rows_image_paths[i] for i in keep_indices]
            rows_feature_image_paths = [rows_feature_image_paths[i] for i in keep_indices]

        # 可选：根据 max_views 过滤病例（只保留视角数 <= max_views 的病例）
        if self.filter_by_max_views and not self.text:
            keep_indices = [
                i for i, paths in enumerate(rows_image_paths) if len(paths) <= self.max_views
            ]
            if len(keep_indices) < len(rows_image_paths):
                dropped = len(rows_image_paths) - len(keep_indices)
                print(
                    f"  ZSFS: 丢弃 {dropped} 个视角数大于 {self.max_views} 的病例 "
                    f"(保留 {len(keep_indices)}/{len(rows_image_paths)})"
                )
            rows_image_paths = [rows_image_paths[i] for i in keep_indices]
            rows_feature_image_paths = [rows_feature_image_paths[i] for i in keep_indices]
            df = df.iloc[keep_indices].reset_index(drop=True)

        self.data_list = df.reset_index(drop=True)
        self._rows_image_paths = rows_image_paths
        self._rows_feature_image_paths = rows_feature_image_paths

        # 一次性输出跳过统计（不逐条打印）
        total_skipped = skipped_multi_prefix + skipped_unresolved
        if total_skipped > 0:
            print(
                "ZSFS 跳过病例统计: "
                f"总计 {total_skipped}，"
                f"前缀多候选 {skipped_multi_prefix}，"
                f"目录/前缀匹配后仍无可用PNG {skipped_unresolved}"
            )
        # 一次性输出命中统计（CSV→磁盘的路径解析方式分布）
        total_hits = hit_direct_dir + hit_strip_ext + hit_prefix_match
        if total_hits > 0:
            print(
                "ZSFS 路径命中统计: "
                f"总计 {total_hits}，"
                f"直拼目录 {hit_direct_dir}，"
                f"去扩展名后命中 {hit_strip_ext}，"
                f"前缀匹配 fallback {hit_prefix_match}"
            )

        if not self.text:
            # 图像-only 模式：扁平化所有图像路径，用于逐图像迭代（特征提取）
            # 使用 data_list["Region"] 与过滤后的 _rows_image_paths 对齐（filter 后 regions 未同步，会错位）
            self._flat_image_paths = []
            self._flat_feature_image_paths = []
            self._flat_regions = []
            for r, paths, feat_paths in zip(
                self.data_list["Region"], self._rows_image_paths, self._rows_feature_image_paths
            ):
                for p, fp in zip(paths, feat_paths):
                    self._flat_image_paths.append(p)
                    self._flat_feature_image_paths.append(fp)
                    self._flat_regions.append(r)
            print(
                f"{self.split.upper()} ZSFS: {len(self._rows_image_paths):,} Cases, "
                f"{len(self._flat_image_paths):,} Images"
            )
        else:
            print(f"{self.split.upper()} ZSFS: {len(self._rows_image_paths):,} Cases (text mode)")

    def __getitem__(self, idx):
        """
        与 BaseDataset 的文本模式保持兼容；
        当前特征提取和 MIL 都使用 text=False（图像-only 模式）。
        """
        if self.text:
            # 预留：如未来需要文本 + 图像联合建模，可在此扩展
            raise NotImplementedError("ZSFSDataset 暂未实现 text=True 模式。")

        # image-only 模式 → 返回单张图像及其路径
        # 严格模式：不允许在 idx 失败时跳到其他图（否则会造成 path<->feature 错位）。
        # __init__ 已经丢弃了无可用 PNG 的病例；这里若加载失败，直接抛错。
        if idx < 0 or idx >= len(self._flat_image_paths):
            raise IndexError(f"ZSFS idx out of range: {idx} (size={len(self._flat_image_paths)})")
        image_path = self._flat_image_paths[idx]
        feature_image_path = (
            self._flat_feature_image_paths[idx]
            if self._flat_feature_image_paths is not None
            else image_path
        )
        try:
            image = load_image_with_dynamic_range(
                image_path, in_chans=self.in_channels
            )
        except Exception as e:
            raise RuntimeError(
                f"ZSFS 图像加载失败（不再静默替换为其他图，请检查文件是否损坏或路径错误）: "
                f"{image_path}\n原始错误: {e}"
            ) from e
        image = self.transform(image)
        return {
            "image": image,
            "image_path": image_path,
            # 用于特征保存时构造特征文件路径（基于 CSV 的逻辑路径）
            "feature_image_path": feature_image_path,
            "region": (self._flat_regions[idx] if self._flat_regions is not None else "ZSFS"),
        }

