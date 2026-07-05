import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from datasets.Private import PrivateDataset

# 46 个疾病标签（顺序固定，全局并集）
DISEASE_LIST = [
    "退行性变",
    "主动脉硬化",
    "骨质疏松",
    "椎间盘病变",
    "颈椎病",
    "椎间隙狭窄",
    "椎体滑脱",
    "脊柱侧凸",
    "骨折",
    "终板炎",
    "关节炎",
    "强直性脊柱炎",
    "先天发育异常",
    "钙化",
    "骨质增生",
    "峡部裂",
    "关节脱位",
    "肺炎",
    "纤维灶",
    "心影增大",
    "肺水肿",
    "肺实变",
    "肺不张",
    "气胸",
    "胸膜增厚",
    "胸腔积液",
    "纵隔增宽",
    "肾结石",
    "尿路结石",
    "肠梗阻",
    "盆腔静脉石",
    "隐裂",
    "骶化",
    "软组织损伤",
    "骨软骨瘤",
    "退行性骨关节病",
    "副骨",
    "踇外翻",
    "扁平足",
    "骨刺",
    "骨化",
    "颈椎失稳",
    "颈椎反弓",
    "髋关节撞击综合征",
    "金属异物",
    "固定术后",
]

# 每个 Region 合理的候选标签子集（均为 DISEASE_LIST 的子集）
# 策略：解剖合理 + test split 阳性数 >= 5（保证 AUC 可信）
# 临床危急/招牌病（如 Chest 肺水肿/肺实变、Cervical 颈椎病、Lumbar 强直性脊柱炎、Foot 扁平足、Wrist 骨化）
# 即使 test 阳性 3~4 也保留
REGION_DISEASES = {
    # 胸部
    "Chest": [
        "退行性变",         # 胸椎退行性变
        "主动脉硬化",
        "骨质疏松",
        "脊柱侧凸",         # 胸椎正位高频
        "骨折",             # 肋骨/胸椎骨折
        "钙化",
        "骨质增生",
        "肺炎",
        "纤维灶",
        "心影增大",
        "肺水肿",           # test=3 但为危急值，保留
        "肺实变",           # test=3 但为危急值，保留
        "肺不张",
        "气胸",
        "胸膜增厚",
        "胸腔积液",
        "纵隔增宽",
        "金属异物",         # 起搏器/导丝等
        "固定术后",         # 锁骨/胸椎/胸骨术后
    ],
    # 腰椎
    "Lumbar_spine": [
        "退行性变",
        "主动脉硬化",       # 腰椎侧位的腹主钙化，test=169
        "骨质疏松",
        "椎间盘病变",
        "椎间隙狭窄",
        "椎体滑脱",
        "脊柱侧凸",
        "骨折",
        "终板炎",
        "关节炎",
        "强直性脊柱炎",     # test=2 但 AS 累及腰段是关键诊断，保留
        "钙化",
        "骨质增生",
        "峡部裂",
        "隐裂",
        "骶化",
        "金属异物",
        "固定术后",
    ],
    # 颈椎
    "Cervical_spine": [
        "退行性变",
        "骨质疏松",
        "椎间盘病变",
        "颈椎病",           # test=3 但为颈椎招牌病，保留
        "椎间隙狭窄",
        "椎体滑脱",
        "脊柱侧凸",
        "骨折",
        "终板炎",
        "钙化",
        "骨质增生",
        "颈椎失稳",
        "颈椎反弓",
        "固定术后",
    ],
    # 骨盆 / 髋
    "Pelvis": [
        "退行性变",
        "骨质疏松",
        "椎间盘病变",       # L5-S1 层面
        "椎间隙狭窄",       # L5-S1 层面
        "椎体滑脱",         # L5-S1 滑脱
        "脊柱侧凸",         # 含腰骶段
        "骨折",
        "关节炎",
        "强直性脊柱炎",
        "先天发育异常",
        "钙化",
        "骨质增生",
        "关节脱位",
        "骨软骨瘤",
        "退行性骨关节病",
        "骶化",
        "髋关节撞击综合征",
        "尿路结石",
        "盆腔静脉石",
        "金属异物",
        "固定术后",
    ],
    # 膝关节
    "Knee": [
        "退行性变",
        "骨质疏松",
        "骨折",
        "关节炎",
        "钙化",
        "骨质增生",
        "关节脱位",
        "软组织损伤",
        "骨软骨瘤",
        "退行性骨关节病",
        "骨化",
        "金属异物",
        "固定术后",
    ],
    # 踝关节
    "Ankle": [
        "退行性变",
        "骨质疏松",
        "骨折",
        "关节炎",
        "钙化",
        "骨质增生",
        "关节脱位",
        "软组织损伤",
        "骨软骨瘤",
        "退行性骨关节病",
        "副骨",
        "骨刺",
        "骨化",
        "金属异物",
        "固定术后",
    ],
    # 足部
    "Foot": [
        "退行性变",
        "骨质疏松",
        "骨折",
        "关节炎",
        "先天发育异常",
        "钙化",
        "骨质增生",
        "关节脱位",
        "软组织损伤",
        "骨软骨瘤",
        "副骨",
        "踇外翻",
        "扁平足",           # test=2 但临床有意义，保留
        "骨刺",
        "骨化",
        "金属异物",
        "固定术后",
    ],
    # 腕关节
    "Wrist": [
        "退行性变",
        "骨质疏松",
        "骨折",
        "关节炎",
        "钙化",
        "骨质增生",
        "关节脱位",
        "软组织损伤",
        "骨软骨瘤",
        "副骨",
        "骨化",             # test=4 边缘，保留
        "金属异物",
        "固定术后",
    ],
    # 掌骨 / 手
    "Metacarpal": [
        "退行性变",
        "骨质疏松",
        "骨折",
        "关节炎",
        "先天发育异常",
        "钙化",
        "骨质增生",
        "关节脱位",
        "软组织损伤",
        "金属异物",
        "固定术后",
    ],
    # 腹部
    "Abdominal": [
        "退行性变",         # 腹部平片报告中腰椎退变描述常见
        "主动脉硬化",
        "骨质疏松",         # 腰椎椎体密度
        "脊柱侧凸",         # 含腰椎部分
        "骨折",             # 腰椎/盆骨/低位肋
        "钙化",
        "骨质增生",         # 腰椎缘骨质增生
        "肾结石",
        "尿路结石",
        "肠梗阻",
        "盆腔静脉石",
        "金属异物",
        "固定术后",         # 腹部平片可见腰椎/骨盆内固定
    ],
}


def load_metadata(features_dir: Path) -> Optional[dict]:
    metadata_path = Path(features_dir) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            return json.load(f)
    return None


def infer_model_id(features_root: Path, regions: List[str]) -> Optional[str]:
    """
    从特征目录自动推断 model_id（region 下的子目录）。
    """
    features_root = Path(features_root)
    for region in regions:
        region_dir = features_root / region
        if region_dir.exists():
            subdirs = [d.name for d in region_dir.iterdir() if d.is_dir()]
            if subdirs:
                return subdirs[0]
    # 兜底：直接找第一层子目录
    subdirs = [d.name for d in features_root.iterdir() if d.is_dir()]
    return subdirs[0] if subdirs else None


def image_path_to_feature_path(
    img_path: str | Path,
    base_img_dir: str,
    features_root: Path,
    model_id: str,
    region_override: Optional[str] = None,
) -> Path:
    """
    将原始图片路径映射到特征文件路径。
    Private 数据集路径结构：
      imgs-512/{Region}/{Year}/{StudyID}/{filename}.png
    特征结构：
      {features_root}/{Region}/{model_id}/{Year}/{StudyID}/{filename}.pt
    """
    img_path = str(img_path)
    base_img_dir = str(base_img_dir)
    # 去掉 base_img_dir 与开头的 '/'
    if base_img_dir in img_path:
        relative_path = img_path.replace(base_img_dir, "").lstrip("/")
    else:
        relative_path = img_path.lstrip("/")
    parts = relative_path.split("/")
    if region_override is not None:
        region = str(region_override)
        # 若相对路径第一段本身就是 region，则去掉第一段；否则保留完整相对路径
        if len(parts) > 1 and parts[0] == region:
            relative_without_region = "/".join(parts[1:])
        else:
            relative_without_region = "/".join(parts)
    else:
        if len(parts) <= 1:
            relative_without_region = "/".join(parts)
            region = "unknown"
        else:
            region = parts[0]
            relative_without_region = "/".join(parts[1:])
    feature_root_dir = Path(features_root) / region / model_id
    feature_path = feature_root_dir / relative_without_region
    # 优先 .npy（提取脚本已改为存 .npy），加载时若无则回退 .pt
    feature_path = feature_path.with_suffix(".npy")
    # print(f"feature_path: {feature_path}")
    return feature_path


def _load_feature_tensor(feat_path: Path) -> torch.Tensor:
    """加载单个特征文件。优先 .npy（更快），不存在则尝试 .pt（兼容旧特征）。"""
    feat_path = Path(feat_path)
    load_path = feat_path if feat_path.exists() else feat_path.with_suffix(".pt")
    if not load_path.exists():
        raise FileNotFoundError(f"特征文件不存在: {feat_path} 或 {feat_path.with_suffix('.pt')}")
    if load_path.suffix == ".npy":
        arr = np.load(load_path)
        return torch.from_numpy(arr).float()
    try:
        obj = torch.load(str(load_path), map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(
            f"无法加载特征文件（可能已损坏）: {load_path}\n原始错误: {e}"
        ) from e
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, dict) and "tensor" in obj:
        return obj["tensor"].float()
    raise RuntimeError(f"特征文件内容不是 Tensor: {load_path}, type={type(obj)}")


class MILFeatureDataset(Dataset):
    """
    将 PrivateDataset 的病例级图片路径映射到特征向量，生成 bag。
    返回:
        bag: [max_views, feature_dim]
        mask: [max_views]，True 表示有效视角
        labels: [num_classes]
        region: str
    """

    def __init__(
        self,
        base_dataset: PrivateDataset,
        features_root: Path,
        model_id: str,
        max_views: int,
        disease_cols: Optional[List[str]] = None,
        feature_dim: Optional[int] = None,
        base_img_dir: Optional[str] = None,
        preload_features: bool = True,
        split: Optional[str] = None,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.features_root = Path(features_root)
        self.model_id = model_id
        self.max_views = max_views
        self.rows_image_paths = base_dataset._rows_image_paths
        self.use_feature_image_paths = bool(
            getattr(base_dataset, "use_feature_image_paths_for_mil", False)
        )
        self.rows_feature_image_paths = (
            getattr(base_dataset, "_rows_feature_image_paths", None)
            if self.use_feature_image_paths
            else None
        )
        self.data_list = base_dataset.data_list.reset_index(drop=True)
        self.base_img_dir = base_img_dir or getattr(base_dataset, "base_img_dir", "")
        self.disease_cols = disease_cols or DISEASE_LIST
        self.feature_dim = feature_dim or self._infer_feature_dim()
        self.preload_features = preload_features
        self.split = split or getattr(base_dataset, "split", "unknown")
        
        # 预加载所有特征到内存
        self._preloaded_features = None
        if self.preload_features:
            # 先尝试从文件加载，如果不存在则从磁盘加载并保存
            if not self._load_preloaded_features():
                self._preload_all_features()
                self._save_preloaded_features()

    def _region_for_index(self, idx: int) -> Optional[str]:
        if "Region" in self.data_list.columns:
            return str(self.data_list.loc[idx, "Region"])
        return None

    def _feature_source_path(self, img_path: str | Path, idx: Optional[int] = None) -> str | Path:
        if idx is None or self.rows_feature_image_paths is None:
            return img_path
        if idx >= len(self.rows_feature_image_paths):
            return img_path

        raw_paths = self.rows_image_paths[idx]
        feature_paths = self.rows_feature_image_paths[idx]
        if len(raw_paths) != len(feature_paths):
            return img_path

        img_path_str = str(img_path)
        for raw_path, feature_path in zip(raw_paths, feature_paths):
            if str(raw_path) == img_path_str:
                return feature_path
        return img_path

    def _feature_source_paths_for_index(self, idx: int) -> List[str | Path]:
        if (
            self.rows_feature_image_paths is not None
            and idx < len(self.rows_feature_image_paths)
            and len(self.rows_feature_image_paths[idx]) == len(self.rows_image_paths[idx])
        ):
            return self.rows_feature_image_paths[idx]
        return self.rows_image_paths[idx]

    def _to_feature_path(self, img_path: str | Path, idx: Optional[int] = None) -> Path:
        region_override = self._region_for_index(idx) if idx is not None else None
        feature_source_path = self._feature_source_path(img_path, idx=idx)
        return image_path_to_feature_path(
            feature_source_path,
            self.base_img_dir,
            self.features_root,
            self.model_id,
            region_override=region_override,
        )

    def _infer_feature_dim(self) -> int:
        for idx in range(len(self.rows_image_paths)):
            source_paths = self._feature_source_paths_for_index(idx)
            for img_path in source_paths:
                feat_path = self._to_feature_path(img_path, idx=idx)
                load_path = feat_path if feat_path.exists() else feat_path.with_suffix(".pt")
                if load_path.exists():
                    feat = _load_feature_tensor(feat_path)
                    if feat.dim() > 1:
                        feat = feat.view(-1)
                    return feat.shape[-1]
        raise FileNotFoundError("无法在特征目录中找到任意特征文件，请检查 features_dir 或 model_id。")

    def _get_preload_save_dir(self, region: str) -> Path:
        """获取预加载特征保存目录"""
        return self.features_root / region / self.model_id / "preload_features"
    
    def _get_preload_save_path(self, region: str) -> Path:
        """获取预加载特征保存路径"""
        save_dir = self._get_preload_save_dir(region)
        return save_dir / f"{self.split}_features.pt"
    
    def _load_preloaded_features(self) -> bool:
        """尝试从文件加载预加载的特征，返回是否成功
        支持单个 region 和多个 regions 的训练场景：
        - 单个 region: 只会加载该 region 的保存文件
        - 多个 regions: 会加载所有 regions 的保存文件并按原始索引顺序合并
        """
        if "Region" not in self.data_list.columns:
            return False
        
        # 获取所有 regions（支持单个或多个 regions）
        regions = self.data_list["Region"].unique().tolist()
        
        # 检查所有 regions 的保存文件是否存在
        region_data = {}
        for region in regions:
            save_path = self._get_preload_save_path(region)
            if not save_path.exists():
                return False
            region_data[region] = torch.load(save_path, map_location="cpu")
        
        # 所有文件都存在，按原始索引顺序合并数据
        print(f"从文件加载预加载特征 (split: {self.split}, regions: {regions})...")
        self._preloaded_features = [None] * len(self.rows_image_paths)
        
        # 按 region 分组建立索引映射
        for region in regions:
            region_indices = self.data_list[self.data_list["Region"] == region].index.tolist()
            region_features = region_data[region]["features"]
            
            if len(region_indices) != len(region_features):
                print(f"警告: region {region} 的数据数量不匹配，重新加载...")
                return False
            
            for local_idx, global_idx in enumerate(region_indices):
                self._preloaded_features[global_idx] = region_features[local_idx]
        
        # 检查是否有 None（表示加载不完整）
        if any(f is None for f in self._preloaded_features):
            return False
        
        total_features = sum(len(feats) for feats in self._preloaded_features)
        print(f"✓ 成功从文件加载预加载特征: {total_features:,} 个特征")
        print()
        return True
    
    def _save_preloaded_features(self):
        """将预加载的特征保存到文件（按 region 分组）
        支持单个 region 和多个 regions 的训练场景：
        - 单个 region: 只保存该 region 的文件
        - 多个 regions: 为每个 region 分别保存文件
        保存路径: {features_root}/{region}/{model_id}/preload_features/{split}_features.pt
        """
        if self._preloaded_features is None or len(self._preloaded_features) == 0:
            return
        
        if "Region" not in self.data_list.columns:
            print("警告: 数据中没有 Region 列，无法保存预加载特征")
            return
        
        # 按 region 分组（支持单个或多个 regions）
        regions = self.data_list["Region"].unique().tolist()
        region_groups = {region: [] for region in regions}
        
        for idx in range(len(self.rows_image_paths)):
            region = self.data_list.loc[idx, "Region"]
            region_groups[region].append(idx)
        
        # 保存每个 region 的数据
        for region, indices in region_groups.items():
            if len(indices) == 0:
                continue
            
            # 提取该 region 的特征
            region_features = [self._preloaded_features[idx] for idx in indices]
            
            # 提取该 region 的标签
            region_labels = []
            for idx in indices:
                row = self.data_list.loc[idx, self.disease_cols]
                labels = row.fillna(0).astype(float).values
                region_labels.append(torch.tensor(labels, dtype=torch.float32))
            
            # 保存到文件
            save_dir = self._get_preload_save_dir(region)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = self._get_preload_save_path(region)
            
            save_dict = {
                "features": region_features,
                "labels": region_labels,
                "feature_dim": self.feature_dim,
                "max_views": self.max_views,
                "split": self.split,
                "num_cases": len(region_features),
            }
            torch.save(save_dict, save_path)
            print(f"✓ 已保存预加载特征到: {save_path} ({len(region_features):,} 个病例)")
        
        print()

    def _preload_all_features(self):
        """预加载所有特征到内存中"""
        print(f"\n开始预加载特征到内存 (共 {len(self.rows_image_paths):,} 个病例)...")
        print(f"特征维度: {self.feature_dim}, 最大视角数: {self.max_views}")
        self._preloaded_features = []
        
        for idx in tqdm(range(len(self.rows_image_paths)), desc="预加载特征", unit="病例"):
            paths = self._feature_source_paths_for_index(idx)
            feats = []
            for p in paths:
                feat_path = self._to_feature_path(p, idx=idx)
                load_path = feat_path if feat_path.exists() else feat_path.with_suffix(".pt")
                if load_path.exists():
                    feat = _load_feature_tensor(feat_path)
                    if feat.dim() > 1:
                        feat = feat.view(-1)
                    feats.append(feat)
                if len(feats) >= self.max_views:
                    break
            self._preloaded_features.append(feats)
        
        # 计算内存占用
        total_features = sum(len(feats) for feats in self._preloaded_features)
        if total_features > 0:
            # 假设每个特征占 feature_dim * 4 bytes (float32)
            memory_mb = total_features * self.feature_dim * 4 / (1024 * 1024)
            memory_gb = memory_mb / 1024
            if memory_gb >= 1:
                print(f"✓ 特征预加载完成: {total_features:,} 个特征, 约 {memory_gb:.2f} GB ({memory_mb:.0f} MB)")
            else:
                print(f"✓ 特征预加载完成: {total_features:,} 个特征, 约 {memory_mb:.2f} MB")
        print()  # 空行分隔

    def __len__(self):
        return len(self.rows_image_paths)

    def __getitem__(self, idx):
        # 如果已预加载，直接从内存读取
        if self.preload_features and self._preloaded_features is not None:
            feats = self._preloaded_features[idx]
        else:
            # 否则从磁盘读取
            paths = self._feature_source_paths_for_index(idx)
            feats = []
            for p in paths:
                feat_path = self._to_feature_path(p, idx=idx)
                load_path = feat_path if feat_path.exists() else feat_path.with_suffix(".pt")
                if load_path.exists():
                    feat = _load_feature_tensor(feat_path)
                    if feat.dim() > 1:
                        feat = feat.view(-1)
                    feats.append(feat)
                if len(feats) >= self.max_views:
                    break

        num_real = len(feats)
        mask = torch.zeros(self.max_views, dtype=torch.bool)
        padded = []

        for i in range(self.max_views):
            if i < num_real:
                f = feats[i]
                if f.numel() != self.feature_dim:
                    raise ValueError(
                        f"特征维度不一致: 读取到 {f.numel()}，期望 {self.feature_dim}。"
                    )
                padded.append(f)
                mask[i] = True
            else:
                padded.append(torch.zeros(self.feature_dim))

        if num_real == 0:
            # 全零 bag，mask 为全 False
            padded[0] = torch.zeros(self.feature_dim)

        bag = torch.stack(padded, dim=0)

        # 取标签（缺失填 0）
        row = self.data_list.loc[idx, self.disease_cols]
        labels = row.fillna(0).astype(float).values
        labels = torch.tensor(labels, dtype=torch.float32)

        region = self.data_list.loc[idx].get("Region", "unknown")
        return bag, mask, labels, region


def mil_collate_fn(batch):
    bags, masks, labels, regions = zip(*batch)
    return (
        torch.stack(bags, dim=0),
        torch.stack(masks, dim=0),
        torch.stack(labels, dim=0),
        list(regions),
    )


def build_private_base_dataset(
    data_dir: str,
    split: str,
    img_size: int,
    in_channels: int,
    region: Optional[List[str]],
    max_views: int,
):
    """
    只用 PrivateDataset 读取 CSV 与路径信息，不会实际加载图片。
    """
    dummy_transform = transforms.Compose([transforms.ToTensor()])
    return PrivateDataset(
        data_dir=data_dir,
        split=split,
        img_size=img_size,
        in_channels=in_channels,
        transform=dummy_transform,
        region=region,
        text=False,
        max_views=max_views,
        # MIL 训练/测试：只保留视角数 <= max_views 的病例
        filter_by_max_views=True,
    )


def build_nyfs_base_dataset(
    data_dir: str,
    split: str,
    img_size: int,
    in_channels: int,
    max_views: int,
    region_filter: Optional[str] = None,
    csv_name: str = "nyfs_prospective_with_labels_en_updated.csv",
):
    """
    基于 NYFS 的病例级数据集，仅解析 CSV 与图像路径，不实际加载图像。
    """
    from datasets.NYFS import NYFSDataset

    dummy_transform = transforms.Compose([transforms.ToTensor()])
    return NYFSDataset(
        data_dir=data_dir,
        split=split,
        img_size=img_size,
        in_channels=in_channels,
        max_views=max_views,
        text=False,
        max_regions=4,
        transform=dummy_transform,
        split_report=True,
        expand_samples=False,
        region_filter=region_filter,
        filter_by_max_views=True,
        csv_name=csv_name,
    )


def build_zsfs_base_dataset(
    data_dir: str,
    split: str,
    img_size: int,
    in_channels: int,
    max_views: int,
    region_filter: Optional[str] = None,
    csv_name: str = "zsfs_prospective_with_labels_en_dedup_updated.csv",
):
    """
    基于 ZSFS 的病例级数据集，仅解析 CSV 与图像路径，不实际加载图像。
    """
    from datasets.ZSFS import ZSFSDataset

    dummy_transform = transforms.Compose([transforms.ToTensor()])
    return ZSFSDataset(
        data_dir=data_dir,
        split=split,
        img_size=img_size,
        in_channels=in_channels,
        max_views=max_views,
        text=False,
        max_regions=4,
        transform=dummy_transform,
        split_report=True,
        expand_samples=False,
        region_filter=region_filter,
        filter_by_max_views=True,
        csv_name=csv_name,
    )


def build_zsfy_base_dataset(
    data_dir: str,
    split: str,
    img_size: int,
    in_channels: int,
    max_views: int,
    region_filter: Optional[str] = None,
    csv_name: str = "zhognshanfuyi_with_labels_qwen3-30b_single_region_updated.csv",
):
    """
    基于 ZSFY 的病例级数据集，仅解析 CSV 与图像路径，不实际加载图像。
    """
    from datasets.ZSFY import ZSFYDataset

    dummy_transform = transforms.Compose([transforms.ToTensor()])
    return ZSFYDataset(
        data_dir=data_dir,
        split=split,
        img_size=img_size,
        in_channels=in_channels,
        max_views=max_views,
        text=False,
        max_regions=4,
        transform=dummy_transform,
        split_report=True,
        expand_samples=False,
        region_filter=region_filter,
        filter_by_max_views=True,
        csv_name=csv_name,
    )


def build_ncfy_base_dataset(
    data_dir: str,
    split: str,
    img_size: int,
    in_channels: int,
    max_views: int,
    region_filter: Optional[str] = None,
    csv_name: str = "ncfy_retrospective_en_with_labels.csv",
):
    """
    基于 NCFY 的病例级数据集，仅解析 CSV 与图像路径，不实际加载图像。
    """
    from datasets.NCFY import NCFYDataset

    dummy_transform = transforms.Compose([transforms.ToTensor()])
    return NCFYDataset(
        data_dir=data_dir,
        split=split,
        img_size=img_size,
        in_channels=in_channels,
        max_views=max_views,
        text=False,
        max_regions=4,
        transform=dummy_transform,
        split_report=True,
        expand_samples=False,
        region_filter=region_filter,
        filter_by_max_views=True,
        csv_name=csv_name,
    )

