#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=2

ROOT=${SMARTX_REPO_ROOT}/downstream/mil_linear_probe
FEATURES_ROOT=${SMARTX_OUTPUT_ROOT}/features/private_a
EXP_ROOT=${SMARTX_OUTPUT_ROOT}/experiments/MIL_LP

cd "$ROOT"

REGIONS=(
  Chest
  Pelvis
  Wrist
  Metacarpal
  Lumbar_spine
  Cervical_spine
  Foot
  Ankle
  Abdominal
  Knee
)

for REGION in "${REGIONS[@]}"; do
  echo "=============================="
  echo "Training rad_dino on REGION=${REGION}"
  echo "=============================="

  python train_mil_lp.py \
    --config configs/train_private_a_rad_dino.yaml \
    --dataset_config configs/dataset_dir.yaml \
    --features_dir "$FEATURES_ROOT" \
    --model_id rad_dino \
    --region "$REGION" \
    --exp_root "$EXP_ROOT"
done
