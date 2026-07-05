#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

ROOT=${SMARTX_REPO_ROOT}/downstream/mil_linear_probe
FEATURES_ROOT=${SMARTX_OUTPUT_ROOT}/features/private_a
EXP_ROOT=${SMARTX_OUTPUT_ROOT}/experiments/MIL_LP

# 提取阶段使用 pretrained_path 的 stem 作为 model_id，需与 extract 时一致
# 例如 pretrained_path=.../topk_v100_best.ckpt -> MODEL_ID=topk_v100_best
MODEL_ID=topk_v100_best

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
  echo "Training our_siglip2 (MODEL_ID=${MODEL_ID}) on REGION=${REGION}"
  echo "=============================="

  python train_mil_lp.py \
    --config configs/train_private_a_our_siglip2.yaml \
    --dataset_config configs/dataset_dir.yaml \
    --features_dir "$FEATURES_ROOT" \
    --model_id "$MODEL_ID" \
    --region "$REGION" \
    --exp_root "$EXP_ROOT"
done
