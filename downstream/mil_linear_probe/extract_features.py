import os
import argparse
import sys
from pathlib import Path
import warnings

import torch
import numpy as np
from tqdm import tqdm
import yaml

warnings.filterwarnings("ignore")
# Ensure repo root is on sys.path so `downstream` can be imported.
sys.path.append(str(Path(__file__).resolve().parents[2]))
from datasets import create_dataloader
from downstream.modules import create_vision_encoder
sys.path.append(str(Path(__file__).parent))
from data_utils import image_path_to_feature_path, DISEASE_LIST


def parse_args():
    parser = argparse.ArgumentParser(description="Extract features for linear probing")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="configs/base_config.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        required=False,
        default="configs/dataset_dir.yaml",
        help="Path to the data configuration file",
    )
    parser.add_argument(
        "--model_name", type=str, required=False, help="Name of the model"
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=False,
        help="Path to the pretrained model",
    )
    parser.add_argument(
        "--dataset_name", type=str, required=False, help="Name of the dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory for features",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if features already exist",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume: skip images whose feature .npy already exists (size > 4KB). Ignored when --force is set.",
    )
    parser.add_argument(
        "--feature_root",
        type=str,
        default=None,
        help="Root directory for saving features (replaces image root, e.g., 'features-512')",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default=None,
        help=(
            "可选：覆盖数据集类的默认 csv_name（位于 dataset_dir 下）。"
            "用于一次抽特征同时处理 NCFY/ZSFS 的 prospective + retrospective 两份 CSV。"
            "未指定时使用 dataset 类的默认 csv_name。"
        ),
    )
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def _compute_feature_path(
    img_path,
    img_path_for_feature,
    region_hint,
    base_img_dir,
    feature_root_base,
    is_private_dataset,
    model_id,
):
    """Compute the on-disk .npy feature path for a single image.

    Mirrors the logic that used to live inline in the per-image save loop, so
    pre-forward skip checks and post-forward saves agree on the same path.
    Returns (feature_path, region) where region may be None for non-private.
    """
    region = region_hint if region_hint and region_hint.lower() != "nan" else None

    if is_private_dataset:
        if region is None:
            if base_img_dir and base_img_dir in img_path:
                relative_path = img_path.replace(base_img_dir, "").lstrip("/")
            else:
                relative_path = img_path
            parts = relative_path.split("/")
            if len(parts) > 0 and parts[0]:
                region = parts[0]
        if region and model_id:
            feature_root_dir = os.path.join(feature_root_base, region, model_id)
        else:
            feature_root_dir = (
                os.path.join(feature_root_base, "unknown", model_id)
                if model_id
                else feature_root_base
            )
    else:
        feature_root_dir = feature_root_base

    if base_img_dir and base_img_dir in img_path_for_feature:
        relative_path = img_path_for_feature.replace(base_img_dir, "").lstrip("/")
    else:
        relative_path = img_path_for_feature
    parts = relative_path.split("/")
    if is_private_dataset and region and len(parts) > 1 and parts[0] == region:
        relative_path_for_feature = "/".join(parts[1:])
    else:
        relative_path_for_feature = relative_path

    feature_path = os.path.join(feature_root_dir, relative_path_for_feature)
    feature_path = os.path.splitext(feature_path)[0] + ".npy"
    return feature_path, region


def _feature_file_complete(path, min_size=4096):
    """Treat a feature file as complete only if it exists AND is large enough.

    1536-dim float32 .npy is ~6.2KB; anything under 4KB is treated as truncated
    from a prior crash and will be re-extracted.
    """
    try:
        return os.path.getsize(path) > min_size
    except OSError:
        return False


def extract_features_from_dataloader(
    encoder,
    dataloader,
    device,
    base_img_dir,
    feature_root_base,
    desc="Extracting",
    is_private_dataset=False,
    model_id=None,
    resume=False,
):
    """
    Extract features from all samples in a dataloader and save each image's feature separately.

    Args:
        encoder: Vision encoder model
        dataloader: DataLoader to extract features from
        device: Device to run on
        base_img_dir: Base directory for images (e.g., "imgs-512")
        feature_root_base: Base directory where features should be saved (replaces base_img_dir)
                          For Private dataset: {feature_root_base}/{region}/{model_id}/
                          For other datasets: {feature_root_base}/
        desc: Description for progress bar
        is_private_dataset: Whether this is Private dataset (needs region-specific paths)
        model_id: Model ID (only needed for Private dataset)

    Returns:
        num_processed: Number of images processed
    """
    encoder.eval()
    num_processed = 0
    num_skipped = 0

    # Initialize progress bar
    pbar = tqdm(dataloader, desc=desc)
    current_region = None

    with torch.no_grad():
        for batch in pbar:
            # Handle different batch formats
            if isinstance(batch, dict):
                images = batch.get("image", batch.get("images", batch.get("imgs")))
                # Check for both image_paths (text mode) and image_path (image-only mode)
                image_paths = batch.get("image_paths", None)
                # 逻辑特征路径（例如 ZSFS 使用基于 CSV 的路径保存特征）
                feature_image_paths = batch.get("feature_image_path", None)
                batch_regions = batch.get("region", None)
                if image_paths is None:
                    # image-only mode: single image_path per item
                    image_path = batch.get("image_path", None)
                    if image_path is not None:
                        # Convert single path to list for consistent handling
                        image_paths = [image_path] if isinstance(image_path, str) else image_path
                    else:
                        image_paths = []
                if feature_image_paths is None:
                    # 若未提供单独的特征路径，则默认与 image_paths 对齐
                    feature_image_paths = image_paths
                if batch_regions is None:
                    batch_regions = []
            else:
                # Tuple format: (images, labels) or list of dicts
                if isinstance(batch, tuple):
                    images = batch[0]
                    image_paths = []
                    batch_regions = []
                elif isinstance(batch, list):
                    # List of dicts from dataset
                    images = torch.stack([item.get("image", item.get("images")) for item in batch])
                    image_paths_list = []
                    for item in batch:
                        if "image_paths" in item:
                            image_paths_list.append(item["image_paths"])
                        elif "image_path" in item:
                            # image-only mode: single path
                            image_paths_list.append([item["image_path"]])
                    image_paths = image_paths_list
                    batch_regions = []
                else:
                    images = batch
                    image_paths = []
                    feature_image_paths = None
                    batch_regions = []
            
            # Skip if no images
            if images is None:
                continue
            
            # Handle case where image_paths is nested (case-based MIL)
            # In text mode, image_paths from batch is a list: [[path1, path2], [path3]] (B cases, each with V paths)
            # In image-only mode, image_paths might be a list of single-item lists: [[path1], [path2]]
            flat_paths = []
            flat_regions = []
            flat_feature_paths = []
            if isinstance(image_paths, list) and len(image_paths) > 0:
                for path_item in image_paths:
                    if isinstance(path_item, list):
                        # Each case can have multiple images
                        flat_paths.extend(path_item)
                    elif isinstance(path_item, str):
                        # Single path
                        flat_paths.append(path_item)
            else:
                flat_paths = []

            # 展平特征使用的逻辑路径（若存在）
            if isinstance(feature_image_paths, list) and len(feature_image_paths) > 0:
                for path_item in feature_image_paths:
                    if isinstance(path_item, list):
                        flat_feature_paths.extend(path_item)
                    elif isinstance(path_item, str):
                        flat_feature_paths.append(path_item)
            else:
                flat_feature_paths = []

            if isinstance(batch_regions, list) and len(batch_regions) > 0:
                # image-only：通常是 [B]；text 模式若后续使用可扩展
                for r in batch_regions:
                    if isinstance(r, list):
                        flat_regions.extend([str(x) for x in r])
                    else:
                        flat_regions.append(str(r))
            
            # For Private dataset, extract region from first path and update progress bar
            if is_private_dataset and len(flat_paths) > 0:
                first_path = flat_paths[0]
                region = None
                if base_img_dir and base_img_dir in first_path:
                    relative_path = first_path.replace(base_img_dir, "").lstrip("/")
                else:
                    relative_path = first_path
                
                path_parts = relative_path.split("/")
                if len(path_parts) > 0 and path_parts[0]:
                    region = path_parts[0]
                
                # Update progress bar description if region changed
                if region and region != current_region:
                    current_region = region
                    pbar.set_description(f"{desc} [Region: {region}]")

            # Precompute feature_path for every image in this batch so we can
            # (a) skip the whole batch when every target file is already complete
            # and (b) skip individual saves for already-complete files.
            precomputed_feature_paths = []
            for idx, img_path in enumerate(flat_paths):
                img_path_for_feature = (
                    flat_feature_paths[idx]
                    if idx < len(flat_feature_paths) and flat_feature_paths[idx]
                    else img_path
                )
                region_hint = flat_regions[idx] if idx < len(flat_regions) else None
                fp, _ = _compute_feature_path(
                    img_path=img_path,
                    img_path_for_feature=img_path_for_feature,
                    region_hint=region_hint,
                    base_img_dir=base_img_dir,
                    feature_root_base=feature_root_base,
                    is_private_dataset=is_private_dataset,
                    model_id=model_id,
                )
                precomputed_feature_paths.append(fp)

            if resume and precomputed_feature_paths and all(
                _feature_file_complete(p) for p in precomputed_feature_paths
            ):
                num_skipped += len(precomputed_feature_paths)
                pbar.set_postfix(skipped=num_skipped, saved=num_processed)
                continue

            # Reshape images if needed
            # In text mode with collate_fn, images shape is [B, V, C, H, W] where V is max_views
            # In image-only mode, images shape is [B, C, H, W]
            original_shape = images.shape
            need_flatten = False
            if len(original_shape) == 5:
                # Text mode: [B, V, C, H, W] -> [B*V, C, H, W]
                B, V, C, H, W = original_shape
                images = images.view(B * V, C, H, W)
                need_flatten = True
            elif len(original_shape) == 4:
                # Image-only mode: [B, C, H, W] - already correct
                pass
            else:
                print(f"Warning: Unexpected image shape {original_shape}, skipping batch")
                continue

            images = images.to(device)
            
            # Debug: Check image device and shape (only for first batch)
            if num_processed == 0:
                print(f"First batch - Images device: {images.device}, shape: {images.shape}, dtype: {images.dtype}")
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
                    print(f"GPU memory after loading images: {memory_allocated:.2f} GB")
            
            # Extract features for each image in the batch
            batch_size = images.shape[0]
            if hasattr(encoder, "extract_cls_token"):
                # RadDino returns CLS embeddings via extract_cls_token
                features = encoder.extract_cls_token(images)
            elif "visiontransformer" in encoder.__class__.__name__.lower():
                features = encoder.forward_features(images)
                features = features[0]
            else:
                outputs = encoder.forward(images)
                # Some encoders return (cls, patches); keep CLS if tuple
                features = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            
            # Debug: Check feature device and shape (only for first batch)
            if num_processed == 0:
                print(f"First batch - Features device: {features.device}, shape: {features.shape}, dtype: {features.dtype}")
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
                    print(f"GPU memory after feature extraction: {memory_allocated:.2f} GB")
            
            # Save each feature to its corresponding path
            features_np = features.cpu().numpy()
            for i in range(batch_size):
                if i >= len(precomputed_feature_paths):
                    print(f"Warning: No path for image {i} in batch (total paths: {len(precomputed_feature_paths)}), skipping")
                    continue

                feature_path = precomputed_feature_paths[i]

                if resume and _feature_file_complete(feature_path):
                    num_skipped += 1
                    continue

                os.makedirs(os.path.dirname(feature_path), exist_ok=True)
                np.save(feature_path, features_np[i].astype(np.float32))
                num_processed += 1

            if resume:
                pbar.set_postfix(skipped=num_skipped, saved=num_processed)

    if resume:
        print(f"Resume summary: saved={num_processed}, skipped(already complete)={num_skipped}")
    return num_processed


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.model_name:
        cfg["model_name"] = args.model_name
    if args.pretrained_path:
        cfg["pretrained_path"] = args.pretrained_path
    if args.dataset_name:
        cfg["dataset_name"] = args.dataset_name
    cfg["dataset_name"] = cfg["dataset_name"].lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        print(f"GPU count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"Current GPU: {torch.cuda.current_device()}")
    data_config = load_config(args.data_config)

    # Determine feature root directory (replaces image root)
    dataset_name = cfg["dataset_name"]
    model_name_cfg = cfg["model_name"]
    preprocess = cfg.get("preprocess", "default")
    if preprocess not in ("default", "lp"):
        raise ValueError(f"Unsupported preprocess: {preprocess}. Use 'default' or 'lp'.")
    if model_name_cfg.startswith("our_") and cfg.get("pretrained_path"):
        # e.g. '${SMARTX_DATA_ROOT}/pretrained_weights/top1_v87_last.ckpt' -> 'top1_v87_last'
        model_id = Path(cfg["pretrained_path"]).stem
    else:
        # 其他模型仍然使用 model_name（去掉路径斜杠）
        model_id = model_name_cfg.replace("/", "_")
    # 不同预处理方案使用独立的 model_id（特征/checkpoint/结果目录全部独立）
    if preprocess == "lp":
        model_id = f"{model_id}_lp"
        print(f"[preprocess=lp] pad-to-square then resize; model_id suffixed with _lp -> {model_id}")
    
    # Check if this is Private-like dataset（需要基于 Region 的特征目录结构）
    # - private_a / private: 原始 Private A 数据
    # - nyfs / zsfs / zsfy / ncfy: 结构类似，使用 Region 作为子目录
    is_private_dataset = dataset_name in ["private_a", "private", "nyfs", "zsfs", "zsfy", "ncfy"]
    
    if args.feature_root:
        feature_root_base = args.feature_root
    else:
        if is_private_dataset:
            # For Private: features/{dataset_name}/ (region and model_id will be added later)
            feature_root_base = str(Path("${SMARTX_OUTPUT_ROOT}/features") / dataset_name)
        else:
            # For other datasets: features/{dataset_name}/{model_id}/
            feature_root_base = str(Path("${SMARTX_OUTPUT_ROOT}/features") / dataset_name / model_id)

    if is_private_dataset:
        print(f"\nFeatures will be saved to: {feature_root_base}/<region>/{model_id}/")
        print(f"  (Each region will have its own subdirectory, then model subdirectory)")
    else:
        print(f"\nFeatures will be saved to: {feature_root_base}")

    # Resolve resume vs. force semantics: --force takes precedence over --resume.
    resume = bool(getattr(args, "resume", False)) and not args.force
    if args.force and getattr(args, "resume", False):
        print("[--force] overrides --resume; existing features will be overwritten.")
    if resume:
        print("[--resume] 已开启增量模式：已存在且 >4KB 的 .npy 将被跳过。")

    # Check if features already exist (skip this gate when force or resume is set)
    if not args.force and not resume:
        features_exist = False
        if is_private_dataset:
            # For Private dataset, check if any region has features
            feature_root_base_path = Path(feature_root_base)
            # Check all possible regions
            possible_regions = [
                "Chest", "Pelvis", "Wrist", "Metacarpal", "Lumbar_spine",
                "Cervical_spine", "Foot", "Ankle", "Abdominal", "Knee"
            ]
            existing_regions = []
            for region in possible_regions:
                region_dir = feature_root_base_path / region / model_id
                # Check if there are any .npy or .pt files in this region's directory
                if region_dir.exists():
                    feature_files = list(region_dir.rglob("*.npy")) + list(region_dir.rglob("*.pt"))
                    if len(feature_files) > 0:
                        existing_regions.append(region)
                        features_exist = True
            
            if features_exist:
                print(f"\n特征文件已存在! 找到以下 region 的特征:")
                for region in existing_regions:
                    region_dir = feature_root_base_path / region / model_id
                    feature_count = len(list(region_dir.rglob("*.npy"))) + len(list(region_dir.rglob("*.pt")))
                    print(f"  - {region}: {feature_count} 个特征文件")
                print(f"\n使用 --force 参数可以重新提取并覆盖现有特征。")
                return
        else:
            # For other datasets, check if feature files exist
            feature_root_base_path = Path(feature_root_base)
            if feature_root_base_path.exists():
                feature_files = list(feature_root_base_path.glob("*.npy")) + list(feature_root_base_path.glob("*.pt"))
                if len(feature_files) > 0:
                    print(f"\n特征文件已存在! 找到 {len(feature_files)} 个特征文件:")
                    for f in feature_files[:10]:  # Show first 10
                        print(f"  - {f.name}")
                    if len(feature_files) > 10:
                        print(f"  ... 还有 {len(feature_files) - 10} 个文件")
                    print(f"\n使用 --force 参数可以重新提取并覆盖现有特征。")
                    return
    else:
        # Force mode: print warning
        print(f"\n[--force] 强制重新提取，将覆盖现有特征文件")

    # Create encoder
    print(f"\nCreating encoder: {cfg['model_name']}")
    encoder = create_vision_encoder(cfg)
    encoder = encoder.to(device)
    encoder.eval()
    
    # Verify model is on GPU and check memory usage
    if torch.cuda.is_available():
        # Check if encoder parameters are on GPU
        first_param = next(encoder.parameters())
        print(f"Encoder device: {first_param.device}")
        print(f"Encoder dtype: {first_param.dtype}")
        
        # Count parameters
        total_params = sum(p.numel() for p in encoder.parameters())
        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Check GPU memory before processing
        torch.cuda.empty_cache()
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
        print(f"GPU memory allocated: {memory_allocated:.2f} GB")
        print(f"GPU memory reserved: {memory_reserved:.2f} GB")

    # Initialize base_img_dir (will be set from dataset)
    base_img_dir_metadata = None

    # Extract features for each split
    # NYFS / ZSFS / ZSFY / NCFY 没有官方 split，这里只跑一次 test（全量数据）避免重复提取。
    split_list = ["test"] if dataset_name in ["nyfs", "zsfs", "zsfy", "ncfy"] else ["train", "validate", "test"]
    for split in split_list:
        print(f"\n{'=' * 50}")
        print(f"Extracting {split} features")
        print("=" * 50)

        # Create dataloader (image-only mode for feature extraction)
        # text=False means image-only mode, where each sample is a single image
        # text=True would be text mode, where each sample is a case with multiple images and text
        dataloader = create_dataloader(
            dataset_name=cfg["dataset_name"],
            data_dir=data_config[cfg["dataset_name"]]["dataset_dir"],
            split=split,
            batch_size=cfg["batch_size"],
            img_size=cfg["img_size"],
            in_channels=cfg["in_channels"],
            normalization=cfg["model_name"].lower(),
            num_workers=cfg["num_workers"],
            text=False,  # Image-only mode: extract features for individual images
            preprocess=preprocess,
            csv_name=args.csv_name,
            shuffle=False,  # Sequential NAS reads; does not affect saved features
            prefetch_factor=cfg.get("prefetch_factor", 4),
        )

        print(f"Dataset size: {len(dataloader.dataset)} samples")
        print(f"Mode: Image-only (each sample is a single image)")
        if len(dataloader.dataset) == 0:
            print(f"No samples found for {split} split")
            continue
        
        # Get base image directory from dataset
        dataset_obj = dataloader.dataset
        base_img_dir = getattr(dataset_obj, "base_img_dir", None)
        if base_img_dir is None:
            # Fallback: construct from data_dir
            base_img_dir = os.path.join(data_config[cfg["dataset_name"]]["dataset_dir"], "imgs-512")
        # Save for metadata
        if base_img_dir_metadata is None:
            base_img_dir_metadata = base_img_dir
        print(f"Base image directory: {base_img_dir}")
        if is_private_dataset:
            print(f"Feature root base: {feature_root_base} (region-specific subdirectories will be created)")
        else:
            print(f"Feature root directory: {feature_root_base}")
        
        # Extract features and save each image's feature separately
        num_processed = extract_features_from_dataloader(
            encoder, dataloader, device, base_img_dir, feature_root_base,
            desc=f"Extracting {split} features",
            is_private_dataset=is_private_dataset,
            model_id=model_id if is_private_dataset else None,
            resume=resume,
        )

        print(f"Processed {num_processed} images")
    
    # Fallback for base_img_dir if no dataset was processed
    if base_img_dir_metadata is None:
        base_img_dir_metadata = os.path.join(data_config[cfg["dataset_name"]]["dataset_dir"], "imgs-512")

    # Save metadata
    import json
    metadata = {
        "model_name": cfg["model_name"],
        "pretrained_path": cfg.get("pretrained_path", "None"),
        "preprocess": preprocess,
        "model_id": model_id,
        "img_size": cfg["img_size"],
        "base_img_dir": base_img_dir_metadata,
        "feature_root_base": feature_root_base,
        "is_private_dataset": is_private_dataset,
        "data_dir": data_config[cfg["dataset_name"]]["dataset_dir"],
    }

    # Save metadata in the base directory
    metadata_path = Path(feature_root_base) / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 50}")
    print("Feature extraction complete!")
    if is_private_dataset:
        print(f"All features saved to: {feature_root_base}/<region>/{model_id}/")
        print(f"  (Each region has its own subdirectory, then model subdirectory)")
    else:
        print(f"All features saved to: {feature_root_base}")
    print("=" * 50)


if __name__ == "__main__":
    main()
