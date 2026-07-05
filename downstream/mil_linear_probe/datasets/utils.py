from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader

from .Private import build_private_dataloader
from .NYFS import NYFSDataset
from .ZSFS import ZSFSDataset
from .ZSFY import ZSFYDataset
from .NCFY import NCFYDataset


class PadToSquare:
    """先把图像 0-padding 成正方形（短边补到等于长边），再交给后续 Resize。
    保证内容完整、无形变（letterbox padding），用于 our_ark*_lp 预处理对照实验。
    """

    def __init__(self, fill: int = 0):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        if w == h:
            return img
        if w > h:
            pad_top = (w - h) // 2
            pad_bottom = w - h - pad_top
            padding = (0, pad_top, 0, pad_bottom)
        else:
            pad_left = (h - w) // 2
            pad_right = h - w - pad_left
            padding = (pad_left, 0, pad_right, 0)
        return TF.pad(img, padding, fill=self.fill)

__all__ = ["create_dataloader"]


def _resolve_normalization_name(normalization: str) -> str:
    """Map model_name-style strings到规范的 normalization 名称。"""
    name = (normalization or "").lower()
    if name in ["our_ark_base", "our_ark_large"]:
        return "ark_base"
    if name in ["our_rad_dino"]:
        return "rad_dino"
    if name in ["our_siglip2"]:
        return "siglip2"
    return name


def _get_mean_std(normalization: str, in_channels: int):
    """
    根据模型名称返回对应的 mean/std。
    逻辑参考 downstream/linear_probe/datasets/utils.get_transforms。
    """
    norm = _resolve_normalization_name(normalization)
    if norm in ["ark_base", "ark_large"]:
        mean = (0.485, 0.456, 0.406)  # ImageNet
        std = (0.229, 0.224, 0.225)
    elif norm == "gloria":
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif norm in ["chexzero", "biomedclip"]:
        # BiomedCLIP (open_clip) 与 CheXzero 一样使用 OpenAI CLIP 的 mean/std
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    elif norm == "maco":
        mean = 0.4978
        std = 0.2449
    elif norm == "chess":
        mean = 0.485
        std = 0.229
    elif norm in ["rad_dino", "siglip2"]:
        # 这两种模型在各自 encoder 内部使用 AutoImageProcessor 做标准化，这里跳过
        return None, None
    else:
        # 未知模型：保守起见不做 Normalize
        return None, None

    return mean, std


def get_mil_transforms(
    split: str,
    img_size: int,
    in_channels: int,
    normalization: str,
    preprocess: str = "default",
):
    """
    MIL 下游特征提取用的图像变换（不做 Data Augmentation）：
      - 统一 Resize 到 (img_size, img_size)
      - 始终 ToTensor
      - 根据模型名称选择性添加 Normalize(mean,std)

    preprocess:
      - "default": 当前默认逻辑（our_ark* 用 short-side resize + center crop；其余直接 squeeze）
      - "lp":      our_ark* 改为 pad-to-square + resize(img_size, img_size)，
                   保留全部内容、无形变；用于对比两种预处理的差异。
    """
    mean, std = _get_mean_std(normalization, in_channels)

    transform_list = []

    if normalization.startswith("our_ark") and preprocess == "lp":
        # 新方案：先 padding 成正方形，再 resize 到目标大小（letterbox）
        transform_list.append(PadToSquare(fill=0))
        transform_list.append(
            transforms.Resize(
                (img_size, img_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        )
    elif normalization.startswith("our_ark") or normalization == "biomedclip":
        # 等比例 resize（short side = img_size）+ center crop，防止变形。
        # BiomedCLIP 也走这条:这正是 open_clip 官方 BiomedCLIP preprocess 的几何变换
        # （Resize(224, bicubic) + CenterCrop(224)），与 downstream/vqa 中
        # get_transforms 对 biomedclip 直接复用 open_clip preprocess 的行为一致。
        transform_list.append(
            transforms.Resize(
                img_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        )
        transform_list.append(transforms.CenterCrop(img_size))
    else:
        transform_list.append(
            transforms.Resize(
                (img_size, img_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        )

    transform_list.append(transforms.ToTensor())

    # Rad-DINO / SigLIP2 使用各自 encoder 内部的 processor 做标准化，这里跳过 Normalize
    if mean is not None and std is not None:
        transform_list.append(
            transforms.Normalize(
                mean=mean,
                std=std,
            )
        )

    return transforms.Compose(transform_list)

def create_dataloader(
    dataset_name,
    data_dir,
    split="train",
    batch_size=32,
    img_size=224,
    in_channels=1,
    normalization="our_ark_base",
    num_workers=8,
    pin_memory=True,
    region=None,
    text=False,
    max_views=6,
    max_regions=4,
    split_report=True,
    expand_samples=False,
    preprocess="default",
    csv_name=None,
    shuffle=None,
    prefetch_factor: int = 4,
):
    """
    Create a dataloader for MIL linear probe evaluation.
    
    Args:
        dataset_name: Name of the dataset (e.g., "private_a", "nyfs")
        data_dir: Root directory of the dataset
        split: Data split ("train", "validate", "test")
        batch_size: Batch size
        img_size: Image size
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        normalization: Normalization type (not used for MIL, kept for compatibility)
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        region: List of regions or single region string (for Private dataset)
        text: Whether to use text mode
        max_views: Maximum number of views per case
        max_regions: Maximum number of regions for text splitting
        split_report: Whether to split report into regions
        expand_samples: Whether to expand samples (deprecated, kept for compatibility)
    
    Returns:
        DataLoader instance
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "private_a" or dataset_name == "private":
        # Default regions if not specified
        if region is None:
            region = [
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
            ]
        elif isinstance(region, str):
            region = [region]
        
        # 在 MIL 线性探针流程中，为了保证与下游训练/测试一致，这里也根据 max_views
        # 丢弃视角数大于 max_views 的病例（filter_by_max_views=True）。
        return build_private_dataloader(
            data_dir=data_dir,
            split=split,
            img_size=img_size,
            batch_size=batch_size,
            text=text,
            in_channels=in_channels,
            region=region,
            max_views=max_views,
            max_regions=max_regions,
            split_report=split_report,
            expand_samples=expand_samples,
            normalization=normalization,
            filter_by_max_views=True,
            preprocess=preprocess,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
    elif dataset_name in ["nyfs", "zsfs", "zsfy", "ncfy"]:
        # NYFS / ZSFS / ZSFY / NCFY 当前主要作为外部测试集使用，不区分 train/validate/test 划分。
        # 为了与现有特征提取脚本兼容，我们对所有 split 使用同一整集数据。
        if split not in ["train", "validate", "test"]:
            raise ValueError(f"{dataset_name.upper()} 不支持 split={split}")

        if text:
            # 目前仅支持图像-only 模式
            raise NotImplementedError(f"{dataset_name.upper()} 暂未实现 text=True 模式。")

        transform = get_mil_transforms(
            split=split,
            img_size=img_size,
            in_channels=in_channels,
            normalization=normalization,
            preprocess=preprocess,
        )

        if dataset_name == "nyfs":
            DatasetCls = NYFSDataset
        elif dataset_name == "zsfs":
            DatasetCls = ZSFSDataset
        elif dataset_name == "zsfy":
            DatasetCls = ZSFYDataset
        else:
            DatasetCls = NCFYDataset
        # 这里同样开启 filter_by_max_views，以保证特征提取阶段与 MIL 训练/评估使用
        # 的病例集合一致（只保留视角数 <= max_views 的病例）。
        # csv_name=None 时由 DatasetCls 使用其默认 CSV；显式传入时覆盖。
        extra_kwargs = {"csv_name": csv_name} if csv_name else {}
        dataset = DatasetCls(
            data_dir=data_dir,
            split=split,
            img_size=img_size,
            in_channels=in_channels,
            max_views=max_views,
            text=False,
            max_regions=max_regions,
            transform=transform,
            split_report=split_report,
            expand_samples=expand_samples,
            filter_by_max_views=True,
            **extra_kwargs,
        )
        loader_shuffle = (split == "train") if shuffle is None else shuffle
        loader_kwargs = dict(
            dataset=dataset,
            shuffle=loader_shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            batch_size=batch_size,
        )
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = prefetch_factor
        loader = DataLoader(**loader_kwargs)
        return loader
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

