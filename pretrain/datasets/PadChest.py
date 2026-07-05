import os

import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import print_rank_zero
from .base_dataset import BaseDataset, collate_fn


class PadChestDataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        split,
        img_size,
        in_channels=1,
        max_views=2,
        max_regions=4,
        transform=None,
        text=False,
        split_report=True,
        expand_samples=False,
    ):
        super().__init__(img_size, expand_samples=expand_samples)
        self.text_column_name = "Translated_Report"
        self.data_dir = data_dir
        self.img_size = img_size
        self.in_channels = in_channels
        self.transform = transform
        self.split = split
        self.text = text
        self.max_views = max_views
        self.max_regions = max_regions
        self.split_report = split_report
        labels_csv = os.path.join(self.data_dir, "labels/padchest_labels_grouped.csv")
        self.data_list = pd.read_csv(labels_csv)
        self.data_list = self.data_list[self.data_list["split"] == self.split]

        # if self.text:
        #     self._filter_invalid_text_rows()
        img_dir_1024 = os.path.join(self.data_dir, "imgs")
        self.base_img_dir = img_dir_1024 if os.path.isdir(img_dir_1024) else os.path.join(self.data_dir, "imgs-512")

        self._rows_image_paths = [
            self._parse_and_fix_paths(x) for x in self.data_list["ImageID"].tolist()
        ]
        
        # Create expanded data if needed
        self._create_expanded_data()
        
        if not self.text:
            # Flatten for image-only mode
            self._flat_image_paths = []
            for paths in self._rows_image_paths:
                self._flat_image_paths.extend(paths)
            print_rank_zero(
                f"{self.split.upper()} PadChest: {len(self._flat_image_paths):,} Images"
            )
        elif self.expand_samples:
            print_rank_zero(
                f"{self.split.upper()} PadChest: {len(self._expanded_data):,} Expanded Samples (from {len(self._rows_image_paths):,} Studies)"
            )
        else:
            print_rank_zero(
                f"{self.split.upper()} PadChest: {len(self._rows_image_paths):,} Studies"
            )


def build_padchest_dataloader(
    data_dir, split, img_size, batch_size, in_channels=1, text=False, max_views=2, max_regions=4, split_report=True, expand_samples=False
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

    dataset = PadChestDataset(
        data_dir=data_dir,
        split=split,
        img_size=img_size,
        in_channels=in_channels,
        transform=transform,
        text=text,
        max_views=max_views,
        max_regions=max_regions,
        split_report=split_report,
        expand_samples=expand_samples,
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
    data_dir = "${SMARTX_DATA_ROOT}/PadChest"
    dataset = PadChestDataset(
        data_dir=data_dir,
        split="train",
        img_size=512,
        in_channels=1,
        transform=transforms.ToTensor(),
        text=True,
        max_views=2,
        max_regions=4,
    )
    print(len(dataset))
    print(dataset[0]["text_attention_mask"])
    print(dataset[0]["image"].shape)
    print(dataset[1]["image"].shape)
    print(dataset[0]["text"])
