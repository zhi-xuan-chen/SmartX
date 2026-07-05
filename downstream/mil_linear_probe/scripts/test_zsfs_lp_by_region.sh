#!/bin/bash
# 测试 our_ark_base_lp 变体（pad-to-square 预处理）在 ZSFS 上的表现
set -e
export CUDA_VISIBLE_DEVICES=0

cd ${SMARTX_REPO_ROOT}/downstream/mil_linear_probe
source scripts/test_by_region_common.sh

FEATURES_ROOT_BASE="${SMARTX_OUTPUT_ROOT}/features"
CKPT_ROOT="${SMARTX_OUTPUT_ROOT}/experiments/MIL_LP"
OUTPUT_ROOT="${SMARTX_REPO_ROOT}/downstream/mil_linear_probe/results"
MIL_TYPE="abmil_simple"
CSV_NAME="${CSV_NAME:-zsfs_prospective_with_labels_en_dedup_updated.csv}"
RESULT_DATASET_NAME="${RESULT_DATASET_NAME:-}"

declare -A CKPT_MODEL_ID_MAP=(
  [our_ark_base_lp]="topk_v95_last_lp"
  [our_ark_large_lp]="topk_v101_best_lp"
)
declare -A FEATURE_MODEL_ID_MAP=(
  [our_ark_base_lp]="topk_v97_best_lp"
  [our_ark_large_lp]="topk_v101_best_lp"
)

MODELS=(
  our_ark_base_lp
  our_ark_large_lp
)

for MODEL_ID in "${MODELS[@]}"; do
  CKPT_MODEL_ID="${CKPT_MODEL_ID_MAP[$MODEL_ID]}"
  FEATURE_MODEL_ID="${FEATURE_MODEL_ID_MAP[$MODEL_ID]}"
  RESULT_MODEL_ID="${MODEL_ID}"
  run_region_test "zsfs" "${MODEL_ID}" "${MIL_TYPE}" "${FEATURES_ROOT_BASE}" "${CKPT_ROOT}" "${OUTPUT_ROOT}" "${CKPT_MODEL_ID}" "${CSV_NAME}" "${RESULT_DATASET_NAME}" "${RESULT_MODEL_ID}" "${FEATURE_MODEL_ID}"
done
