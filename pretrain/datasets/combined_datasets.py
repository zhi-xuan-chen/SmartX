import random

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import transforms
import torchvision.transforms.functional as TF

from .CheXpert import CheXpertDataset
from .MIMIC import MIMICDataset
from .PadChest import PadChestDataset
from .ReXGradient import ReXGradientDataset
from .base_dataset import collate_fn
from utils import print_rank_zero

def build_combined_concat_dataloader(
    chexpert_data_dir: str,
    padchest_data_dir: str,
    mimic_data_dir: str,
    rexgradient_data_dir: str,
    split: str,
    text: bool,
    img_size: int,
    batch_size: int,
    num_workers: int,
    max_views: int,
    max_regions: int,
    split_report: bool = True,
    in_channels=3,
    expand_samples=False,
    vision_encoder_name: str = None,
    prefetch_factor: int = 2,
    max_samples: int | None = None,
    data_fraction: float = 1.0,
    data_fraction_seed: int = 42,
):
    """
    Build a combined dataloader using ConcatDataset (simplest approach).
    
    Args:
        img_size: Image size for transforms
        batch_size: Batch size
        split: Dataset split ("train" or "val")
        vision_encoder_name: Name of vision encoder (e.g., "smartx_large", "rad-dino", "siglip2")
                          If "rad-dino" or "siglip2", normalization will be skipped in transforms
                          and handled inside the vision encoder / its processor instead.
        max_samples: If set, use only the first N samples (smoke test / quick pipeline check).
    """
    # Skip normalization for Rad-DINO and SigLIP2, which apply normalization
    # internally via their HuggingFace processors to avoid duplicate normalization.
    skip_normalize = vision_encoder_name in ("rad-dino", "siglip2")
    
    datasets = []
    if chexpert_data_dir is not None:
        chexpert_dataset = CheXpertDataset(
            data_dir=chexpert_data_dir,
            split=split,
            img_size=img_size,
            max_views=max_views,
            text=text,
            max_regions=max_regions,
            transform=_get_transforms(img_size, split, "chexpert", in_channels, skip_normalize),
            split_report=split_report,
            in_channels=in_channels,
            expand_samples=expand_samples,
        )
        datasets.append(chexpert_dataset)

    if padchest_data_dir is not None:
        padchest_dataset = PadChestDataset(
            data_dir=padchest_data_dir,
            split=split,
            img_size=img_size,
            max_views=max_views,
            text=text,
            max_regions=max_regions,
            transform=_get_transforms(img_size, split, "padchest", in_channels, skip_normalize),
            split_report=split_report,
            in_channels=in_channels,
            expand_samples=expand_samples,
        )
        datasets.append(padchest_dataset)

    if mimic_data_dir is not None:
        mimic_dataset = MIMICDataset(
            data_dir=mimic_data_dir,
            split=split,
            img_size=img_size,
            max_views=max_views,
            text=text,
            max_regions=max_regions,
            transform=_get_transforms(img_size, split, "mimic", in_channels, skip_normalize),
            split_report=split_report,
            in_channels=in_channels,
            expand_samples=expand_samples,
        )
        datasets.append(mimic_dataset)

    if rexgradient_data_dir is not None:
        rexgradient_dataset = ReXGradientDataset(
            data_dir=rexgradient_data_dir,
            split=split,
            img_size=img_size,
            max_views=max_views,
            text=text,
            max_regions=max_regions,
            transform=_get_transforms(img_size, split, "rexgradient", in_channels, skip_normalize),
            split_report=split_report,
            in_channels=in_channels,
            expand_samples=expand_samples,
        )
        datasets.append(rexgradient_dataset)

    # Subsample each training source/region with a fixed seed to preserve proportions.
    # Validation uses the full dataset.
    if split == "train" and 0.0 < float(data_fraction) < 1.0:
        rng = random.Random(int(data_fraction_seed))
        sub_datasets = []
        for i, ds in enumerate(datasets):
            n = len(ds)
            target = max(1, int(round(n * float(data_fraction))))
            indices = list(range(n))
            rng.shuffle(indices)
            indices = sorted(indices[:target])
            sub_datasets.append(Subset(ds, indices))
            print_rank_zero(
                f"[data_fraction={data_fraction}] dataset[{i}] subsampled "
                f"{target}/{n} ({100.0 * target / max(1, n):.1f}%)"
            )
        datasets = sub_datasets

    if not datasets or not any(len(dataset) for dataset in datasets):
        raise ValueError(f"No usable {split} data; configure at least one nonempty dataset")
    # Combine datasets
    combined_dataset = ConcatDataset(datasets)

    if max_samples is not None and max_samples > 0:
        n = min(int(max_samples), len(combined_dataset))
        combined_dataset = Subset(combined_dataset, range(n))
        print_rank_zero(
            f"Combined {split} dataset truncated to first {n} samples (max_samples={max_samples})."
        )

    # Print combined dataset length for debugging
    print_rank_zero(f"Combined {split} dataset length: {len(combined_dataset):,}")

    # Normal training drops incomplete last batch; smoke-test subsets may be smaller than
    # batch_size and must keep the final partial batch.
    drop_last = (split == "train") and (max_samples is None) and len(combined_dataset) >= batch_size

    loader = DataLoader(
        dataset=combined_dataset,
        shuffle=(split == "train"),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=True,
        batch_size=batch_size,
        collate_fn=collate_fn if text else None,
        drop_last=drop_last,
    )

    return loader


def pad_to_square(img):
    """Pad image to square with zero padding."""
    w, h = img.size
    max_wh = max(w, h)
    hp = int((max_wh - w) / 2)
    vp = int((max_wh - h) / 2)
    return TF.pad(img, (hp, vp, max_wh - w - hp, max_wh - h - vp), 0, 'constant')


def _get_transforms(img_size: int, split: str, dataset_name: str=None, in_channels: int=3, skip_normalize: bool=False):
    """Get transforms for the given split
    
    Args:
        img_size: Target image size
        split: Dataset split ("train" or "validate")
        dataset_name: Name of the dataset
        in_channels: Number of input channels
        skip_normalize: If True, skip normalization (e.g., for Rad-DINO which handles normalization internally)
    """
    dataset_name = dataset_name.lower()
    dataset_mean_std = {
        "chexpert": [0.503, 0.291],
        "padchest": [0.904, 0.288],
        "mimic": [0.472, 0.304],
        'rexgradient': [0.864, 0.338],
    }
    if in_channels == 1:
        mean, std = dataset_mean_std[dataset_name]
    else:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    
    if split == "train":
        transform_list = [
            transforms.Lambda(pad_to_square),
            transforms.Resize(
                int(img_size * 1.1),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomCrop(img_size),
            transforms.RandomRotation(
                degrees=10, 
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
        ]
        if not skip_normalize:
            transform_list.append(transforms.Normalize(mean=mean, std=std))
        return transforms.Compose(transform_list)
    else:  # validation
        transform_list = [
            transforms.Lambda(pad_to_square),
            transforms.Resize(
                int(img_size * 1.1),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
        if not skip_normalize:
            transform_list.append(transforms.Normalize(mean=mean, std=std))
        return transforms.Compose(transform_list)
