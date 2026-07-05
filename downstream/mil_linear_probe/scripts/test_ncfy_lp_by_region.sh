#!/bin/bash
# 测试 our_ark_base_lp 变体（pad-to-square 预处理）在 NCFY 上的表现
set -e
export CUDA_VISIBLE_DEVICES=0

cd ${SMARTX_REPO_ROOT}/downstream/mil_linear_probe
source scripts/test_by_region_common.sh

FEATURES_ROOT_BASE="${SMARTX_OUTPUT_ROOT}/features"
CKPT_ROOT="${SMARTX_OUTPUT_ROOT}/experiments/MIL_LP"
OUTPUT_ROOT="${SMARTX_REPO_ROOT}/downstream/mil_linear_probe/results"
MIL_TYPE="abmil_simple"
CSV_NAME="${CSV_NAME:-ncfy_prospective_en_with_labels.csv}"
RESULT_DATASET_NAME="${RESULT_DATASET_NAME:-}"

# LP 变体：
#   CKPT_MODEL_ID    = 训练 MIL 时使用的 private_a 特征 model_id（即 train 脚本里 MODEL_ID）
#   FEATURE_MODEL_ID = 在 NCFY 上提取特征时使用的 model_id（取自 extract_ncfy_our_ark_base_lp.yaml 的 pretrained_path）
#   RESULT_MODEL_ID  = 结果目录中的展示名，方便和 default 预处理区分
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
  run_region_test "ncfy" "${MODEL_ID}" "${MIL_TYPE}" "${FEATURES_ROOT_BASE}" "${CKPT_ROOT}" "${OUTPUT_ROOT}" "${CKPT_MODEL_ID}" "${CSV_NAME}" "${RESULT_DATASET_NAME}" "${RESULT_MODEL_ID}" "${FEATURE_MODEL_ID}"
done
