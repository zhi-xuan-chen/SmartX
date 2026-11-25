"""
Image transforms for custom vision encoders.
Use each encoder original preprocessing logic; no external image processor is required.
"""

from torchvision import transforms

from .utils import VIS_ENCODER_CONFIG

# Mapping from model_name to normalization parameters for get_transforms.
_NORM_MAP = {
    "gloria": "gloria",
    "chexzero": "chexzero",
    "maco": "maco",
    "rad_dino": "rad_dino",
    "chess": "chess",
    "ark_base": "ark_base",
    "ark_large": "ark_large",
    "smartx_base": "smartx_base",
    "smartx_large": "smartx_large",
    "biomedclip": "chexzero",
    "conceptclip": "chexzero",
}


def get_transforms(
    img_size: int = 224,
    in_channels: int = 1,
    normalization: str = "smartx_large",
    preserve_aspect: bool = False,
):
    if normalization in ["smartx_base", "smartx_large"]:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif normalization in ["ark_base", "ark_large"]:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif normalization == "gloria":
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif normalization == "chexzero":
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    elif normalization == "maco":
        mean = 0.4978
        std = 0.2449
    elif normalization == "chess":
        mean = 0.485
        std = 0.229
    elif normalization in ["rad_dino", "our_rad_dino"]:
        mean = None
        std = None
    else:
        raise ValueError(f"Invalid normalization: {normalization}")

    transform_list = []

    if normalization in ["maco", "chess"]:
        transform_list.append(transforms.Grayscale(num_output_channels=1))

    if preserve_aspect:
        transform_list.append(
            transforms.Resize(
                img_size,  # short side = img_size, preserve aspect ratio
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

    if normalization == "maco":
        transform_list.append(transforms.Grayscale(num_output_channels=3))

    transform_list.append(transforms.ToTensor())
    if mean is not None and std is not None:
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)


def get_custom_image_transform(model_name: str):
    """Return the image transform Compose for the given encoder name."""
    model_name = model_name.lower()
    if model_name not in VIS_ENCODER_CONFIG:
        raise ValueError(
            f"Unknown encoder '{model_name}'. Supported: {list(VIS_ENCODER_CONFIG.keys())}"
        )
    if model_name not in _NORM_MAP:
        raise ValueError(
            f"No normalization config for '{model_name}'. Supported: {list(_NORM_MAP.keys())}"
        )

    cfg = VIS_ENCODER_CONFIG[model_name]
    # Preserve aspect ratio + center crop for SmartX models
    preserve_aspect = model_name.startswith("smartx_")
    return get_transforms(
        img_size=cfg["input_size"],
        in_channels=cfg["channels"],
        normalization=_NORM_MAP[model_name],
        preserve_aspect=preserve_aspect,
    )
