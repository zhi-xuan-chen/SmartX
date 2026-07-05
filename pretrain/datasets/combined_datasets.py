import random
from typing import List

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms
import torchvision.transforms.functional as TF

from .CheXpert import CheXpertDataset
from .MIMIC import MIMICDataset
from .PadChest import PadChestDataset
from .Private import PrivateDataset
from .ReXGradient import ReXGradientDataset
from .base_dataset import collate_fn
from utils import print_rank_zero

def pad_to_square(img):
    w, h = img.size
    max_wh = max(w, h)
    hp = int((max_wh - w) / 2)
    vp = int((max_wh - h) / 2)
    return TF.pad(img, (hp, vp, max_wh - w - hp, max_wh - h - vp), 0, 'constant')

class WeightedCombinedDataset(Dataset):
    """
    Combines multiple datasets with specified weights for sampling probability.
    Each dataset can have different sampling weights.
    """

    def __init__(self, datasets: List[Dataset], weights: List[float] = None):
        """
        Args:
            datasets: List of Dataset objects to combine
            weights: Can be:
                    - "uniform": Equal weights for all datasets
                    - "proportional": Weights based on dataset sizes
                    - "inverse_proportional": Weights inversely proportional to dataset sizes
                    - List[float]: Custom weights for each dataset (will be normalized)
                    - None: Same as "uniform"
        """
        self.datasets = datasets
        self.dataset_lengths = [len(d) for d in datasets]
        self.total_length = sum(self.dataset_lengths)

        # Calculate cumulative lengths for dataset selection
        self.cumulative_lengths = []
        cumsum = 0
        for length in self.dataset_lengths:
            cumsum += length
            self.cumulative_lengths.append(cumsum)

        # Set up weights based on input type
        if weights is None or weights == "uniform":
            # Equal weights for all datasets
            weights = [1.0] * len(datasets)
        elif weights == "proportional":
            # Weights based on dataset sizes
            weights = self.dataset_lengths
        elif weights == "inverse_proportional":
            # Weights inversely proportional to dataset sizes (smaller datasets get higher weights)
            weights = [1.0 / length if length > 0 else 1.0 for length in self.dataset_lengths]
        elif isinstance(weights, (list, tuple)):
            # Custom weights provided
            if len(weights) != len(datasets):
                raise ValueError(f"Number of weights ({len(weights)}) must match number of datasets ({len(datasets)})")
            weights = list(weights)
        else:
            raise ValueError(f"Invalid weights type: {type(weights)}. Must be 'uniform', 'proportional', 'inverse_proportional', or List[float]")

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Sum of weights cannot be zero")
        self.weights = [w / total_weight for w in weights]

        # Create weighted indices for each dataset
        self._create_weighted_indices()

    def _create_weighted_indices(self):
        """Create indices based on dataset weights"""
        self.weighted_indices = []

        for i, (dataset, weight) in enumerate(zip(self.datasets, self.weights)):
            # Number of samples from this dataset based on weight
            num_samples = int(weight * self.total_length)
            # Create indices for this dataset (with replacement if needed)
            dataset_indices = []
            while len(dataset_indices) < num_samples:
                remaining = num_samples - len(dataset_indices)
                available = len(dataset)
                if remaining >= available:
                    dataset_indices.extend(list(range(available)))
                else:
                    dataset_indices.extend(random.sample(range(available), remaining))

            # Store as (dataset_idx, sample_idx) pairs
            for idx in dataset_indices:
                self.weighted_indices.append((i, idx))

        # Shuffle the combined indices
        random.shuffle(self.weighted_indices)

    def __len__(self):
        return len(self.weighted_indices)

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.weighted_indices[idx]
        return self.datasets[dataset_idx][sample_idx]


class MultiDatasetLoader:
    """
    Alternates between multiple dataloaders, useful when you want to
    maintain separate batch sizes or sampling strategies for each dataset.
    """

    def __init__(self, dataloaders: List[DataLoader], mode: str = "round_robin"):
        """
        Args:
            dataloaders: List of DataLoader objects
            mode: 'round_robin' (alternate), 'random' (random selection),
                  'proportional' (based on dataset sizes)
        """
        self.dataloaders = dataloaders
        self.mode = mode
        self.iterators = []
        self.dataset_sizes = [len(dl.dataset) for dl in dataloaders]
        self.total_size = sum(self.dataset_sizes)

        if mode == "proportional":
            self.probabilities = [size / self.total_size for size in self.dataset_sizes]

    def __iter__(self):
        # Create fresh iterators
        self.iterators = [iter(dl) for dl in self.dataloaders]
        self.current_idx = 0
        self.exhausted = [False] * len(self.dataloaders)
        return self

    def __next__(self):
        if all(self.exhausted):
            raise StopIteration

        if self.mode == "round_robin":
            return self._next_round_robin()
        elif self.mode == "random":
            return self._next_random()
        elif self.mode == "proportional":
            return self._next_proportional()

    def _next_round_robin(self):
        """Alternate between dataloaders in order"""
        attempts = 0
        while attempts < len(self.dataloaders):
            try:
                batch = next(self.iterators[self.current_idx])
                self.current_idx = (self.current_idx + 1) % len(self.dataloaders)
                return batch, self.current_idx
            except StopIteration:
                self.exhausted[self.current_idx] = True
                self.current_idx = (self.current_idx + 1) % len(self.dataloaders)
                attempts += 1

        raise StopIteration

    def _next_random(self):
        """Randomly select from available dataloaders"""
        available = [i for i, exhausted in enumerate(self.exhausted) if not exhausted]
        if not available:
            raise StopIteration

        idx = random.choice(available)
        try:
            batch = next(self.iterators[idx])
            return batch, idx
        except StopIteration:
            self.exhausted[idx] = True
            return self._next_random()

    def _next_proportional(self):
        """Select based on dataset size proportions"""
        available = [i for i, exhausted in enumerate(self.exhausted) if not exhausted]
        if not available:
            raise StopIteration

        # Adjust probabilities for available dataloaders
        available_probs = [self.probabilities[i] for i in available]
        total_prob = sum(available_probs)
        normalized_probs = [p / total_prob for p in available_probs]

        idx = available[torch.multinomial(torch.tensor(normalized_probs), 1).item()]
        try:
            batch = next(self.iterators[idx])
            return batch, idx
        except StopIteration:
            self.exhausted[idx] = True
            return self._next_proportional()

    def __len__(self):
        return sum(len(dl) for dl in self.dataloaders)


def build_combined_concat_dataloader(
    chexpert_data_dir: str,
    padchest_data_dir: str,
    mimic_data_dir: str,
    private_data_dir: str,
    rexgradient_data_dir: str,
    region: list[str],
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
    filter_top_num_image_pct: float | None = None,
    max_num_image: int | None = 10,
    max_samples: int | None = None,
    data_fraction: float = 1.0,
    data_fraction_seed: int = 42,
):
    """
    Build a combined dataloader using ConcatDataset (simplest approach).
    
    Args:
        datasets_config: List of dicts with dataset configuration
            Example: [
                {"type": "chexpert", "data_dir": "/path/to/chexpert"},
                {"type": "xr", "data_dir": "/path/to/xr"}
            ]
        img_size: Image size for transforms
        batch_size: Batch size
        split: Dataset split ("train" or "val")
        vision_encoder_name: Name of vision encoder (e.g., "rad-dino", "ark_base", "siglip2")
                          If "rad-dino" or "siglip2", normalization will be skipped in transforms
                          and handled inside the vision encoder / its processor instead.
        max_samples: If set, use only the first N samples (smoke test / quick pipeline check).
    """
    # Skip normalization for Rad-DINO 和 SigLIP2，这两种模型在各自的 encoder 内部使用
    # HuggingFace 的 processor 做标准化，避免与 dataloader 中的 Normalize 重复或不一致。
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

    if private_data_dir is not None:
        for region_name in region:
            private_dataset = PrivateDataset(
                data_dir=private_data_dir,
                split=split,
                img_size=img_size,
                max_views=max_views,
                region=[region_name],
                text=text,
                max_regions=max_regions,
                transform=_get_transforms(img_size, split, f"private_a_{region_name.lower()}", in_channels, skip_normalize),
                split_report=split_report,
                in_channels=in_channels,
                expand_samples=expand_samples,
                filter_top_num_image_pct=filter_top_num_image_pct,
                max_num_image=max_num_image,
            )
            datasets.append(private_dataset)

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

    # Stratified subsampling for data-scale ablation (A4):
    # subsample each per-source / per-region dataset independently with a fixed seed
    # so that region proportions are preserved across the 25% / 50% / 100% sweep.
    # Only applied to the train split; val stays full-size.
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
    drop_last = (split == "train") and (max_samples is None)

    loader = DataLoader(
        dataset=combined_dataset,
        shuffle=(split == "train"),
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        batch_size=batch_size,
        collate_fn=collate_fn if text else None,
        drop_last=drop_last,
    )

    return loader


def build_weighted_combined_dataloader(
    chexpert_data_dir: str,
    padchest_data_dir: str,
    mimic_data_dir: str,
    private_data_dir: str,
    rexgradient_data_dir: str,
    region: list[str],
    weights,
    img_size: int,
    batch_size: int,
    split: str,
    text: bool,
    num_workers: int,
    max_views: int,
    max_regions: int,
    split_report: bool = True,
    in_channels=3,
    expand_samples=False,
    prefetch_factor: int = 2,
):
    """
    Build a combined dataloader with weighted sampling.

    Args:
        chexpert_data_dir: Path to CheXpert dataset
        padchest_data_dir: Path to PadChest dataset
        mimic_data_dir: Path to MIMIC dataset
        private_data_dir: Path to Private dataset
        rexgradient_data_dir: Path to ReXGradient dataset
        weights: Can be:
                - "uniform": Equal weights for all datasets
                - "proportional": Weights based on dataset sizes
                - "inverse_proportional": Weights inversely proportional to dataset sizes
                - List[float]: Custom weights for each dataset
        img_size: Image size for transforms
        batch_size: Batch size
        split: Dataset split ("train" or "val")
        text: Whether to use text data
        num_workers: Number of workers for dataloader
    """
    datasets = []
    cxr_datasets = []
    xr_datasets = []
    if chexpert_data_dir is not None:
        chexpert_dataset = CheXpertDataset(
            data_dir=chexpert_data_dir,
            split=split,
            img_size=img_size,
            max_views=max_views,
            text=text,
            max_regions=max_regions,
            transform=_get_transforms(img_size, split),
            split_report=split_report,
            in_channels=in_channels,
            expand_samples=expand_samples,
        )
        cxr_datasets.append(chexpert_dataset)

    if rexgradient_data_dir is not None:
        rexgradient_dataset = ReXGradientDataset(
            data_dir=rexgradient_data_dir,
            split=split,
            img_size=img_size,
            max_views=max_views,
            text=text,
            max_regions=max_regions,
            transform=_get_transforms(img_size, split),
            split_report=split_report,
            in_channels=in_channels,
            expand_samples=expand_samples,
        )
        cxr_datasets.append(rexgradient_dataset)

    if padchest_data_dir is not None:
        padchest_dataset = PadChestDataset(
            data_dir=padchest_data_dir,
            split=split,
            img_size=img_size,
            max_views=max_views,
            text=text,
            max_regions=max_regions,
            transform=_get_transforms(img_size, split),
            split_report=split_report,
            in_channels=in_channels,
            expand_samples=expand_samples,
        )
        cxr_datasets.append(padchest_dataset)

    if mimic_data_dir is not None:
        mimic_dataset = MIMICDataset(
            data_dir=mimic_data_dir,
            split=split,
            img_size=img_size,
            max_views=max_views,
            text=text,
            max_regions=max_regions,
            transform=_get_transforms(img_size, split),
            split_report=split_report,
            in_channels=in_channels,
            expand_samples=expand_samples,
        )
        cxr_datasets.append(mimic_dataset)

    if private_data_dir is not None:
        private_cxr_dataset = PrivateDataset(
            data_dir=private_data_dir,
            split=split,
            img_size=img_size,
            max_views=max_views,
            text=text,
            max_regions=max_regions,
            region=['Chest'] if 'Chest' in region else None,
            transform=_get_transforms(img_size, split),
            split_report=split_report,
            in_channels=in_channels,
            expand_samples=expand_samples,
        )
        cxr_datasets.append(private_cxr_dataset)
    if private_data_dir is not None:
        if 'Chest' in region:
            region.remove('Chest')
        private_xr_dataset = PrivateDataset(
            data_dir=private_data_dir,
            split=split,
            img_size=img_size,
            max_views=max_views,
            text=text,
            max_regions=max_regions,
            region=region,
            transform=_get_transforms(img_size, split),
            split_report=split_report,
            in_channels=in_channels,
            expand_samples=expand_samples,
        )
        xr_datasets.append(private_xr_dataset)

    # Combine datasets with weights
    datasets = [ConcatDataset(cxr_datasets), ConcatDataset(xr_datasets)]
    combined_dataset = WeightedCombinedDataset(datasets, weights)
    print(f"CXR Datasets: {len(ConcatDataset(cxr_datasets))}")
    print(f"XR Datasets: {len(ConcatDataset(xr_datasets))}")
    print(f"Combined Datasets: {len(combined_dataset)}")
    # Create dataloader
    loader = DataLoader(
        dataset=combined_dataset,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        batch_size=batch_size,
        drop_last=True,
        collate_fn=collate_fn if text else None,
    )
    return loader

def build_weighted_combined_dataloader_each(
    chexpert_data_dir: str,
    padchest_data_dir: str,
    mimic_data_dir: str,
    private_data_dir: str,
    rexgradient_data_dir: str,
    region: list[str],
    weights,
    img_size: int,
    batch_size: int,
    split: str,
    text: bool,
    num_workers: int,
    max_views: int,
    max_regions: int,
    split_report: bool = True,
    in_channels=3,
    prefetch_factor: int = 2,
):
    """
    Build a combined dataloader with weighted sampling.

    Args:
        chexpert_data_dir: Path to CheXpert dataset
        padchest_data_dir: Path to PadChest dataset
        mimic_data_dir: Path to MIMIC dataset
        private_data_dir: Path to Private dataset
        rexgradient_data_dir: Path to ReXGradient dataset
        weights: Can be:
                - "uniform": Equal weights for all datasets
                - "proportional": Weights based on dataset sizes
                - "inverse_proportional": Weights inversely proportional to dataset sizes
                - List[float]: Custom weights for each dataset
        img_size: Image size for transforms
        batch_size: Batch size
        split: Dataset split ("train" or "val")
        text: Whether to use text data
        num_workers: Number of workers for dataloader
    """
    datasets = []
    cxr_datasets = []
    xr_datasets = []
    if chexpert_data_dir is not None:
        chexpert_dataset = CheXpertDataset(
            data_dir=chexpert_data_dir,
            split=split,
            img_size=img_size,
            max_views=max_views,
            text=text,
            max_regions=max_regions,
            transform=_get_transforms(img_size, split),
            split_report=split_report,
            in_channels=in_channels,
        )
        cxr_datasets.append(chexpert_dataset)

    if rexgradient_data_dir is not None:
        rexgradient_dataset = ReXGradientDataset(
            data_dir=rexgradient_data_dir,
            split=split,
            img_size=img_size,
            max_views=max_views,
            text=text,
            max_regions=max_regions,
            transform=_get_transforms(img_size, split),
            split_report=split_report,
            in_channels=in_channels,
        )
        cxr_datasets.append(rexgradient_dataset)

    if padchest_data_dir is not None:
        padchest_dataset = PadChestDataset(
            data_dir=padchest_data_dir,
            split=split,
            img_size=img_size,
            max_views=max_views,
            text=text,
            max_regions=max_regions,
            transform=_get_transforms(img_size, split),
            split_report=split_report,
            in_channels=in_channels,
        )
        cxr_datasets.append(padchest_dataset)

    if mimic_data_dir is not None:
        mimic_dataset = MIMICDataset(
            data_dir=mimic_data_dir,
            split=split,
            img_size=img_size,
            max_views=max_views,
            text=text,
            max_regions=max_regions,
            transform=_get_transforms(img_size, split),
            split_report=split_report,
            in_channels=in_channels,
        )
        cxr_datasets.append(mimic_dataset)

    if private_data_dir is not None:
        print(f"Private CXR Datasets: {region}")
        private_cxr_dataset = PrivateDataset(
            data_dir=private_data_dir,
            split=split,
            img_size=img_size,
            max_views=max_views,
            text=text,
            max_regions=max_regions,
            region=['Chest'] if 'Chest' in region else None,
            transform=_get_transforms(img_size, split),
            split_report=split_report,
            in_channels=in_channels,
        )
        region.remove('Chest')
        cxr_datasets.append(private_cxr_dataset)
    
    datasets = [ConcatDataset(cxr_datasets)]
    if private_data_dir is not None:
        for region_name in region:
            private_xr_dataset = PrivateDataset(
                data_dir=private_data_dir,
                split=split,
                img_size=img_size,
                max_views=max_views,
                text=text,
                max_regions=max_regions,
                region=[region_name],
                transform=_get_transforms(img_size, split),
                split_report=split_report,
                in_channels=in_channels,
            )
            datasets.append(private_xr_dataset)

    # Combine datasets with weights
    combined_dataset = WeightedCombinedDataset(datasets, weights)
    # print(f"CXR Datasets: {len(ConcatDataset(cxr_datasets))}")
    # print(f"XR Datasets: {len(ConcatDataset(xr_datasets))}")
    print(f"Combined Datasets: {len(combined_dataset)}")
    # Create dataloader
    loader = DataLoader(
        dataset=combined_dataset,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        batch_size=batch_size,
        drop_last=True,
        collate_fn=collate_fn if text else None,
    )
    return loader

def build_multi_dataloader(
    chexpert_data_dir: str,
    padchest_data_dir: str,
    mimic_data_dir: str,
    private_data_dir: str,
    rexgradient_data_dir: str,
    region: list[str],
    mode: str,
    img_size: int,
    batch_size: int,
    split: str,
    text: bool,
    num_workers: int,
    max_views: int,
    max_regions: int,
    split_report: bool = True,
    in_channels=3,
):
    """
    Build a combined dataloader with weighted sampling.

    Args:
        chexpert_data_dir: Path to CheXpert dataset
        padchest_data_dir: Path to PadChest dataset
        mimic_data_dir: Path to MIMIC dataset
        private_data_dir: Path to Private dataset
        rexgradient_data_dir: Path to ReXGradient dataset
        region: List of regions to use
        mode: How to alternate between loaders ("round_robin", "random", "proportional")
        img_size: Image size for transforms
        batch_size: Batch size
        split: Dataset split ("train" or "val")
        text: Whether to use text data
        num_workers: Number of workers for dataloader
    """
    dataloaders = []
    cxr_datasets = []
    xr_datasets = []
    
    if chexpert_data_dir is not None:
        chexpert_dataset = CheXpertDataset(
            data_dir=chexpert_data_dir,
            split=split,
            img_size=img_size,
            max_views=max_views,
            text=text,
            max_regions=max_regions,
            transform=_get_transforms(img_size, split),
            split_report=split_report,
            in_channels=in_channels,
        )
        cxr_datasets.append(chexpert_dataset)
    if rexgradient_data_dir is not None:
        rexgradient_dataset = ReXGradientDataset(
            data_dir=rexgradient_data_dir,
            split=split,
            img_size=img_size,
            max_views=max_views,
            text=text,
            max_regions=max_regions,
            transform=_get_transforms(img_size, split),
            split_report=split_report,
            in_channels=in_channels,
        )
        cxr_datasets.append(rexgradient_dataset)

    if mimic_data_dir is not None:
        mimic_dataset = MIMICDataset(
            data_dir=mimic_data_dir,
            split=split,
            img_size=img_size,
            max_views=max_views,
            text=text,
            max_regions=max_regions,
            transform=_get_transforms(img_size, split),
            split_report=split_report,
            in_channels=in_channels,
        )
        cxr_datasets.append(mimic_dataset)

    if padchest_data_dir is not None:
        padchest_dataset = PadChestDataset(
            data_dir=padchest_data_dir,
            split=split,
            img_size=img_size,
            max_views=max_views,
            text=text,
            max_regions=max_regions,
            transform=_get_transforms(img_size, split),
            split_report=split_report,
            in_channels=in_channels,
        )
        cxr_datasets.append(padchest_dataset)

    if private_data_dir is not None and 'Chest' in region:
        private_dataset = PrivateDataset(
            data_dir=private_data_dir,
            split=split,
            img_size=img_size,
            max_views=max_views,
            text=text,
            max_regions=max_regions,
            region=region,
            transform=_get_transforms(img_size, split),
            split_report=split_report,
            in_channels=in_channels,
        )
        cxr_datasets.append(private_dataset)

    if private_data_dir is not None and 'XR' in region:
        private_dataset = PrivateDataset(
            data_dir=private_data_dir,
            split=split,
            img_size=img_size,
            max_views=max_views,
            text=text,
            max_regions=max_regions,
            region=region,
            transform=_get_transforms(img_size, split),
            split_report=split_report,
            in_channels=in_channels,
        )
        xr_datasets.append(private_dataset)

    cxr_dataloader = DataLoader(
        dataset=ConcatDataset(cxr_datasets),
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        batch_size=batch_size,
        drop_last=True,
        collate_fn=collate_fn if text else None,
    )
    xr_dataloader = DataLoader(
        dataset=ConcatDataset(xr_datasets),
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        batch_size=batch_size,
        drop_last=True,
        collate_fn=collate_fn if text else None,
    )
    dataloaders = [cxr_dataloader, xr_dataloader]
    return MultiDatasetLoader(dataloaders, mode=mode)


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
        "private_a_chest": [0.536, 0.266],
        "private_a_pelvis": [0.380, 0.236],
        "private_a_wrist": [0.293, 0.247],
        "private_a_metacarpal": [0.313, 0.234],
        "private_a_lumbar_spine": [0.361, 0.266],
        "private_a_cervical_spine": [0.423, 0.298],
        "private_a_foot": [0.333, 0.254],
        "private_a_ankle": [0.280, 0.249],
        "private_a_abdominal": [0.445, 0.220],
        "private_a_knee": [0.333, 0.258],
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
