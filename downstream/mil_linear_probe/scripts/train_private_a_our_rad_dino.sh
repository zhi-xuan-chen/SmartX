#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=2

ROOT=${SMARTX_REPO_ROOT}/downstream/mil_linear_probe
FEATURES_ROOT=${SMARTX_OUTPUT_ROOT}/features/private_a
EXP_ROOT=${SMARTX_OUTPUT_ROOT}/experiments/MIL_LP

cd "$ROOT"

# 提取阶段使用 pretrained_path: topk_v83_last.ckpt -> model_id=topk_v83_last
MODEL_ID=topk_v83_last

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
  echo "Training our_rad_dino (MODEL_ID=${MODEL_ID}) on REGION=${REGION}"
  echo "=============================="

  python train_mil_lp.py \
    --config configs/train_private_a_our_rad_dino.yaml \
    --dataset_config configs/dataset_dir.yaml \
    --features_dir "$FEATURES_ROOT" \
    --model_id "$MODEL_ID" \
    --region "$REGION" \
    --exp_root "$EXP_ROOT"
done
