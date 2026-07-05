#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

ROOT=${SMARTX_REPO_ROOT}/downstream/mil_linear_probe
FEATURES_ROOT=${SMARTX_OUTPUT_ROOT}/features/private_a
EXP_ROOT=${SMARTX_OUTPUT_ROOT}/experiments/MIL_LP

cd "$ROOT"

# 自动从 extract yaml 推导 MODEL_ID（保持与 extract_features.py 的 model_id 规则一致）
EXTRACT_CFG=configs/extract_private_a_our_ark_large.yaml
PRETRAINED_PATH=$(grep -E '^pretrained_path:' "$EXTRACT_CFG" | sed -E 's/^pretrained_path:[[:space:]]*"?([^"]+)"?.*/\1/')
PREPROCESS=$(grep -E '^preprocess:' "$EXTRACT_CFG" 2>/dev/null | sed -E 's/^preprocess:[[:space:]]*"?([^"]+)"?.*/\1/' || true)
MODEL_ID=$(basename "$PRETRAINED_PATH" .ckpt)
if [ "$PREPROCESS" = "lp" ]; then
  MODEL_ID="${MODEL_ID}_lp"
fi
echo "[auto] EXTRACT_CFG=${EXTRACT_CFG}"
echo "[auto] PRETRAINED_PATH=${PRETRAINED_PATH}"
echo "[auto] PREPROCESS=${PREPROCESS:-default}"
echo "[auto] MODEL_ID=${MODEL_ID}"

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
  echo "Training our_ark_large (MODEL_ID=${MODEL_ID}) on REGION=${REGION}"
  echo "=============================="

  python train_mil_lp.py \
    --config configs/train_private_a_our_ark_large.yaml \
    --dataset_config configs/dataset_dir.yaml \
    --features_dir "$FEATURES_ROOT" \
    --model_id "$MODEL_ID" \
    --region "$REGION" \
    --exp_root "$EXP_ROOT"
done
