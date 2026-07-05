#!/bin/bash
# Recompute val_thresholds.json for topk_v19_last / topk_v14_last across all
# regions and abmil types (abmil_full, abmil_simple, clam where present).
#
# Uses best checkpoint per region (highest val_custom_auc), not last.ckpt.
#
# Usage:
#   bash scripts/update_val_thresholds_v14_v19.sh          # run all
#   bash scripts/update_val_thresholds_v14_v19.sh --dry-run

set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate vlp
fi

python batch_save_val_thresholds.py \
  --config configs/train_private_a_our_ark_large.yaml \
  --dataset_config configs/dataset_dir.yaml \
  --features_dir ${SMARTX_OUTPUT_ROOT}/features/private_a \
  --ckpt_root ${SMARTX_OUTPUT_ROOT}/experiments/MIL_LP \
  --model_ids topk_v19_last \
  --mil_types abmil_full abmil_simple clam \
  --batch_size 128 \
  --num_workers 4 \
  --overwrite \
  "$@"
