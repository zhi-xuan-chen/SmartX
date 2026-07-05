"""Batch recompute val_thresholds.json for multiple model_id / mil_type / region combos.

Uses ``build_region_ckpt_map`` to pick the best checkpoint (highest val_custom_auc)
per region, then runs val inference + Youden's J threshold selection via
``save_val_thresholds`` helpers.

Example:
    python batch_save_val_thresholds.py \
        --model_ids topk_v19_last topk_v14_last \
        --mil_types abmil_full abmil_simple clam \
        --overwrite
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

sys.path.append(str(Path(__file__).parent.parent))
from utils import load_config

from build_region_ckpt_map import build_region_ckpt_map
from save_val_thresholds import build_val_loader, run_val_inference
from trainer_mil_lp import MILLinearProbe
from val_thresholds_utils import compute_per_class_youden, write_thresholds_json, thresholds_path_for_ckpt

DEFAULT_REGIONS = [
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch save val_thresholds.json for MIL LP ckpts")
    p.add_argument(
        "--config",
        type=str,
        default="configs/train_private_a_our_ark_large.yaml",
        help="training-style yaml (img_size/max_views should match ckpts)",
    )
    p.add_argument("--dataset_config", type=str, default="configs/dataset_dir.yaml")
    p.add_argument(
        "--features_dir",
        type=str,
        default="${SMARTX_OUTPUT_ROOT}/features/private_a",
    )
    p.add_argument(
        "--ckpt_root",
        type=str,
        default="${SMARTX_OUTPUT_ROOT}/experiments/MIL_LP",
    )
    p.add_argument("--dataset_name", type=str, default="private_a")
    p.add_argument(
        "--model_ids",
        type=str,
        nargs="+",
        default=["topk_v19_last", "topk_v14_last"],
    )
    p.add_argument(
        "--mil_types",
        type=str,
        nargs="+",
        default=["abmil_full", "abmil_simple", "clam"],
    )
    p.add_argument(
        "--regions",
        type=str,
        nargs="*",
        default=None,
        help="default: all 10 private_a regions",
    )
    p.add_argument("--batch_size", type=int, default=128, help="val inference batch size")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="only list jobs, do not run inference")
    return p.parse_args()


def _resolve_disease_cols_out(model: MILLinearProbe, available_cols: list[str]) -> list[str]:
    hparams = model.hparams
    ckpt_cols = (
        hparams.get("disease_cols")
        if isinstance(hparams, dict)
        else getattr(hparams, "disease_cols", None)
    )
    if ckpt_cols is not None and list(ckpt_cols) != list(available_cols):
        if set(ckpt_cols) != set(available_cols):
            raise ValueError(
                f"disease_cols mismatch — ckpt has {len(ckpt_cols)} cols, "
                f"val loader exposes {len(available_cols)} cols."
            )
        print("[warn] disease_cols order differs from ckpt; reindexing val preds.")
        return list(ckpt_cols)
    return list(available_cols)


def process_one(
    cfg: dict,
    dataset_cfg: dict,
    features_dir: Path,
    model_id: str,
    region: str,
    ckpt_path: Path,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
) -> bool:
    sidecar = thresholds_path_for_ckpt(ckpt_path)
    if sidecar.exists() and not overwrite:
        print(f"[skip] {sidecar} exists")
        return False

    print(f"\n{'=' * 72}")
    print(f"model_id={model_id}  region={region}")
    print(f"ckpt={ckpt_path}")
    print(f"write={sidecar}")

    loader, available_cols = build_val_loader(
        cfg,
        dataset_cfg,
        features_dir,
        model_id,
        region,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    model = MILLinearProbe.load_from_checkpoint(str(ckpt_path))
    disease_cols_out = _resolve_disease_cols_out(model, available_cols)

    probs, labels = run_val_inference(model, loader)
    if list(disease_cols_out) != list(available_cols):
        order = [available_cols.index(c) for c in disease_cols_out]
        probs = probs[:, order]
        labels = labels[:, order]

    thresholds = compute_per_class_youden(labels, probs)
    write_thresholds_json(
        sidecar,
        disease_cols=disease_cols_out,
        thresholds=thresholds,
        region=region,
        extra={
            "selection_rule": "youden_j",
            "source": "batch_save_val_thresholds.py",
            "ckpt": str(ckpt_path),
        },
    )
    print(f"✓ wrote {sidecar} ({len(thresholds)} classes)")
    return True


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dataset_cfg = load_config(args.dataset_config)
    cfg["dataset_name"] = cfg.get("dataset_name", args.dataset_name).lower()

    features_dir = Path(args.features_dir)
    ckpt_root = Path(args.ckpt_root)
    regions = args.regions or DEFAULT_REGIONS

    jobs: list[tuple[str, str, str, Path]] = []
    for model_id in args.model_ids:
        for mil_type in args.mil_types:
            region_map = build_region_ckpt_map(
                ckpt_root=ckpt_root,
                dataset_name=args.dataset_name,
                model_id=model_id,
                mil_type=mil_type,
                regions=regions,
            )
            for region in regions:
                ckpt_str = region_map.get(region)
                if not ckpt_str:
                    print(f"[missing] {model_id} / {mil_type} / {region}: no checkpoint")
                    continue
                ckpt_path = Path(ckpt_str)
                if not ckpt_path.is_file():
                    print(
                        f"[missing] {model_id} / {mil_type} / {region}: "
                        f"ckpt not found: {ckpt_path}"
                    )
                    continue
                jobs.append((model_id, mil_type, region, ckpt_path))

    print(f"Planned jobs: {len(jobs)}")
    if args.dry_run:
        for model_id, mil_type, region, ckpt_path in jobs:
            print(f"  {model_id}\t{mil_type}\t{region}\t{ckpt_path}")
        return

    ok = 0
    skipped = 0
    failed = 0
    for i, (model_id, mil_type, region, ckpt_path) in enumerate(jobs, start=1):
        print(f"\n[{i}/{len(jobs)}] mil_type={mil_type}")
        try:
            if process_one(
                cfg,
                dataset_cfg,
                features_dir,
                model_id,
                region,
                ckpt_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                overwrite=args.overwrite,
            ):
                ok += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            print(f"[ERROR] {model_id} / {mil_type} / {region}: {exc}")

    print(f"\nDone. wrote={ok}, skipped={skipped}, failed={failed}, total={len(jobs)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
