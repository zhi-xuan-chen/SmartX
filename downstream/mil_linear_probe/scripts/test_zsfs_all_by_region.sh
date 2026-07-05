#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0

cd ${SMARTX_REPO_ROOT}/downstream/mil_linear_probe
source scripts/test_by_region_common.sh

FEATURES_ROOT_BASE="${SMARTX_OUTPUT_ROOT}/features"
CKPT_ROOT="${SMARTX_OUTPUT_ROOT}/experiments/MIL_LP"
OUTPUT_ROOT="${SMARTX_REPO_ROOT}/downstream/mil_linear_probe/results"
MIL_TYPE="abmil_simple"
# 可选：指定评估 CSV 文件名
CSV_NAME="${CSV_NAME:-zsfs_prospective_with_labels_en_dedup_updated.csv}"
# 可选：指定结果目录中的数据集名（不指定则按 CSV 自动映射）
RESULT_DATASET_NAME="${RESULT_DATASET_NAME:-}"

# 可选：为 our_* 模型指定 checkpoint 目录名映射（默认等于 MODEL_ID）
# declare -A CKPT_MODEL_ID_MAP=(
#   [our_ark_base]="topk_v97_last"
#   [our_rad_dino]="topk_v83_last"
#   [our_siglip2]="topk_v100_best"
# )

MODELS=(
  ark_base
  ark_large
  chess
  chexzero
  gloria
  biomedclip
  maco
  topk_v97_last
  topk_v97_best
  topk_v95_last
  topk_v95_best
  topk_v83_last
  topk_v100_best
  rad_dino
  siglip2
)

for MODEL_ID in "${MODELS[@]}"; do
  CKPT_MODEL_ID="${MODEL_ID}"
  RESULT_MODEL_ID="${MODEL_ID}"
  FEATURE_MODEL_ID="${MODEL_ID}"
  run_region_test "zsfs" "${MODEL_ID}" "${MIL_TYPE}" "${FEATURES_ROOT_BASE}" "${CKPT_ROOT}" "${OUTPUT_ROOT}" "${CKPT_MODEL_ID}" "${CSV_NAME}" "${RESULT_DATASET_NAME}" "${RESULT_MODEL_ID}" "${FEATURE_MODEL_ID}"
done

