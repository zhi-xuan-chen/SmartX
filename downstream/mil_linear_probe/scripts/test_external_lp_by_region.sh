#!/bin/bash
# 一键 by-region 测试 [LP 变体方法 our_ark_base_lp / our_ark_large_lp] × [4 个外部数据集]。
# 对于 NCFY 和 ZSFS,既有 prospective 又有 retrospective 的 CSV,两份都跑。
# 结果通过 infer_result_dataset_name 自动分到 ncfy_pro / ncfy_ret / nyfs / zsfs_pro / zsfs_ret / zsfy 目录。
#
# 用法:
#   bash scripts/test_external_lp_by_region.sh                   # 默认 GPU 0
#   CUDA_VISIBLE_DEVICES=3 bash scripts/test_external_lp_by_region.sh
#
# 注意:此脚本不影响已有的 test_<ds>_lp_by_region.sh,只是新增"全数据集 × LP 方法"入口。
set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/test_by_region_common.sh

FEATURES_ROOT_BASE="${SMARTX_OUTPUT_ROOT}/features"
CKPT_ROOT="${SMARTX_OUTPUT_ROOT}/experiments/MIL_LP"
OUTPUT_ROOT="${SMARTX_REPO_ROOT}/downstream/mil_linear_probe/results"
MIL_TYPE="abmil_simple"
RESULT_DATASET_NAME=""

# (dataset, csv_name) 对,与 test_external_all_by_region.sh 一致。
declare -a DS_CSV_PAIRS=(
  "ncfy|ncfy_prospective_en_with_labels.csv"
  "ncfy|ncfy_retrospective_en_with_labels.csv"
  "nyfs|nyfs_prospective_with_labels_en_updated.csv"
  "zsfs|zsfs_prospective_with_labels_en_dedup_updated.csv"
  "zsfs|zsfs_retrospective_with_labels_en_dedup_updated.csv"
  "zsfy|zhognshanfuyi_with_labels_qwen3-30b_single_region_updated.csv"
)

# LP 变体:
#   CKPT_MODEL_ID    = 训练 MIL 时使用的 private_a 特征 model_id
#   FEATURE_MODEL_ID = 在外部数据集上提取特征时使用的 model_id
#   RESULT_MODEL_ID  = 结果目录中的展示名,跟 default 预处理区分
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
  for PAIR in "${DS_CSV_PAIRS[@]}"; do
    DATASET="${PAIR%%|*}"
    CSV_NAME="${PAIR##*|}"
    run_region_test \
      "${DATASET}" "${MODEL_ID}" "${MIL_TYPE}" \
      "${FEATURES_ROOT_BASE}" "${CKPT_ROOT}" "${OUTPUT_ROOT}" \
      "${CKPT_MODEL_ID}" "${CSV_NAME}" "${RESULT_DATASET_NAME}" \
      "${RESULT_MODEL_ID}" "${FEATURE_MODEL_ID}"
  done
done

echo
echo "✓ External 4 Datasets × 2 LP Methods by-region testing complete (GPU=$CUDA_VISIBLE_DEVICES)"
