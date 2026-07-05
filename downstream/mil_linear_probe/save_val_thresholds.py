"""One-shot script: compute per-class Youden's J val thresholds for **existing** MIL
checkpoints and write a sidecar ``val_thresholds.json`` next to each ckpt.

Run this once per region after training has already finished, so that
``trainer_mil_lp.py`` (internal test) and the 4 external test scripts can
load the val thresholds instead of falling back to 0.5 or re-thresholding
on the test set.

Example:
    python save_val_thresholds.py \
        --config configs/train_private_a_our_ark_base.yaml \
        --dataset_config configs/dataset_dir.yaml \
        --features_dir /path/to/features/private_a \
        --region Chest \
        --ckpt /path/to/.../checkpoints/best.ckpt
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from utils import load_config

from trainer_mil_lp import MILLinearProbe
from data_utils import (
    DISEASE_LIST,
    REGION_DISEASES,
    infer_model_id,
    build_private_base_dataset,
    MILFeatureDataset,
    mil_collate_fn,
)
from val_thresholds_utils import (
    compute_per_class_youden,
    write_thresholds_json,
    thresholds_path_for_ckpt,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recompute & save val_thresholds.json for existing ckpts")
    p.add_argument("--config", type=str, required=True, help="training-style config yaml")
    p.add_argument("--dataset_config", type=str, required=True)
    p.add_argument("--features_dir", type=str, required=True)
    p.add_argument("--model_id", type=str, default=None)
    p.add_argument("--region", type=str, required=True, help="single body region, e.g. Chest")
    p.add_argument("--ckpt", type=str, required=True, help="path to best.ckpt")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--overwrite", action="store_true", help="overwrite existing val_thresholds.json")
    return p.parse_args()


def _drop_na_label_rows(base_dataset, available_cols: list[str], split_name: str) -> None:
    if len(base_dataset.data_list) != len(base_dataset._rows_image_paths):
        raise ValueError(
            f"{split_name} 数据不一致: data_list={len(base_dataset.data_list)}, "
            f"rows_image_paths={len(base_dataset._rows_image_paths)}"
        )
    keep_mask = ~base_dataset.data_list[available_cols].isna().any(axis=1)
    keep_indices = [i for i, keep in enumerate(keep_mask.tolist()) if keep]
    base_dataset.data_list = base_dataset.data_list.iloc[keep_indices].reset_index(drop=True)
    base_dataset._rows_image_paths = [base_dataset._rows_image_paths[i] for i in keep_indices]
    if hasattr(base_dataset, "_flat_image_paths") and base_dataset._flat_image_paths is not None:
        base_dataset._flat_image_paths = []
        for paths in base_dataset._rows_image_paths:
            base_dataset._flat_image_paths.extend(paths)


def build_val_loader(
    cfg: dict, dataset_cfg: dict, features_dir: Path, model_id: str, region: str,
    batch_size: int, num_workers: int,
):
    dataset_name = cfg["dataset_name"]
    data_dir = dataset_cfg[dataset_name]["dataset_dir"]
    max_views = cfg.get("max_views", 6)
    img_size = cfg.get("img_size", 224)
    in_channels = cfg.get("in_channels", 1)
    regions = [region]

    base_val = build_private_base_dataset(
        data_dir, "validate", img_size, in_channels, regions, max_views
    )
    candidate_cols = REGION_DISEASES.get(region, DISEASE_LIST)
    available_cols = [c for c in candidate_cols if c in base_val.data_list.columns]
    if not available_cols:
        raise ValueError(f"region={region} 找不到候选疾病标签列")
    _drop_na_label_rows(base_val, available_cols, "validate")

    feature_dim = MILFeatureDataset(
        base_val, features_root=features_dir, model_id=model_id,
        max_views=max_views, disease_cols=available_cols, preload_features=False,
    ).feature_dim

    dataset = MILFeatureDataset(
        base_val, features_root=features_dir, model_id=model_id,
        max_views=max_views, disease_cols=available_cols, feature_dim=feature_dim,
        preload_features=False, split="validate",
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=mil_collate_fn,
    )
    return loader, available_cols


@torch.no_grad()
def run_val_inference(model: MILLinearProbe, loader) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    probs_chunks, labels_chunks = [], []
    for batch in tqdm(loader, desc="val inference"):
        if len(batch) == 4:
            bags, mask, labels, _ = batch
        else:
            bags, mask, labels = batch
        bags = bags.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True) if mask is not None else None
        labels = labels.to(device, non_blocking=True).float()
        logits, _ = model(bags, mask)
        probs = torch.sigmoid(logits)
        probs_chunks.append(probs.detach().cpu().numpy())
        labels_chunks.append(labels.detach().cpu().numpy())
    if not probs_chunks:
        raise RuntimeError("val loader empty")
    return np.concatenate(probs_chunks, axis=0), np.concatenate(labels_chunks, axis=0)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dataset_cfg = load_config(args.dataset_config)
    cfg["dataset_name"] = cfg.get("dataset_name", "private_a").lower()

    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    sidecar = thresholds_path_for_ckpt(ckpt_path)
    if sidecar.exists() and not args.overwrite:
        print(f"[skip] {sidecar} already exists; use --overwrite to regenerate.")
        return

    model_id = args.model_id or infer_model_id(Path(args.features_dir), [args.region])
    if model_id is None:
        raise ValueError("无法自动推断 model_id，请通过 --model_id 指定。")
    batch_size = args.batch_size or cfg.get("batch_size", 8)
    num_workers = args.num_workers if args.num_workers is not None else cfg.get("num_workers", 4)

    print(f"Region: {args.region}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Will write: {sidecar}")

    loader, available_cols = build_val_loader(
        cfg, dataset_cfg, Path(args.features_dir), model_id, args.region,
        batch_size=batch_size, num_workers=num_workers,
    )
    model = MILLinearProbe.load_from_checkpoint(str(ckpt_path))

    # cross-check that disease_cols in ckpt matches val loader columns
    hparams = model.hparams
    ckpt_cols = (
        hparams.get("disease_cols") if isinstance(hparams, dict)
        else getattr(hparams, "disease_cols", None)
    )
    if ckpt_cols is not None and list(ckpt_cols) != list(available_cols):
        # Align loader output to ckpt order if it's a permutation.
        if set(ckpt_cols) == set(available_cols):
            print("[warn] disease_cols order differs from ckpt; reindexing val preds.")
        else:
            raise ValueError(
                f"disease_cols mismatch — ckpt has {len(ckpt_cols)} cols, "
                f"val loader exposes {len(available_cols)} cols."
            )

    probs, labels = run_val_inference(model, loader)

    # Reindex to ckpt's disease_cols order if needed.
    if ckpt_cols is not None and list(ckpt_cols) != list(available_cols):
        order = [available_cols.index(c) for c in ckpt_cols]
        probs = probs[:, order]
        labels = labels[:, order]
        disease_cols_out = list(ckpt_cols)
    else:
        disease_cols_out = list(available_cols)

    thresholds = compute_per_class_youden(labels, probs)
    write_thresholds_json(
        sidecar,
        disease_cols=disease_cols_out,
        thresholds=thresholds,
        region=args.region,
        extra={"selection_rule": "youden_j", "source": "save_val_thresholds.py"},
    )
    print(f"✓ wrote {sidecar} ({len(thresholds)} classes)")


if __name__ == "__main__":
    main()
