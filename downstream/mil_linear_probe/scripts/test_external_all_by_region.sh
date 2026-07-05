#!/bin/bash
# 一键 by-region 测试 [全部 14 个 baseline 方法] × [4 个外部数据集 NCFY/NYFS/ZSFS/ZSFY]。
# 对于 NCFY 和 ZSFS,既有 prospective 又有 retrospective 的 CSV,两份都跑。
# 结果通过 infer_result_dataset_name 自动分到 results/<mil_type>/ncfy_pro / ncfy_ret / nyfs / zsfs_pro / zsfs_ret / zsfy 目录。
#
# 用法:
#   bash scripts/test_external_all_by_region.sh                   # 默认 GPU 0
#   CUDA_VISIBLE_DEVICES=3 bash scripts/test_external_all_by_region.sh
#
# 注意:此脚本不影响已有的 test_<ds>_all_by_region.sh,只是新增"全数据集 × 全方法"入口。
set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/test_by_region_common.sh

# NOTE: temporarily pointed at ${SMARTX_STORAGE_ROOT} to re-test biomedclip, whose external
# features were extracted there (default root). Revert to ${SMARTX_STORAGE_ROOT} before
# re-running any OTHER baseline — their external features live under ${SMARTX_STORAGE_ROOT}.
FEATURES_ROOT_BASE="${SMARTX_OUTPUT_ROOT}/features"
CKPT_ROOT="${SMARTX_OUTPUT_ROOT}/experiments/MIL_LP"
OUTPUT_ROOT="${SMARTX_REPO_ROOT}/downstream/mil_linear_probe/results"
MIL_TYPE="clam"
# RESULT_DATASET_NAME 留空,让 common.sh 按 CSV 自动映射(pros→<ds>_pro / retro→<ds>_ret)
RESULT_DATASET_NAME=""

# (dataset, csv_name) 对:NCFY/ZSFS 各两份,NYFS/ZSFY 各一份。共 6 个组合。
declare -a DS_CSV_PAIRS=(
  "ncfy|ncfy_prospective_en_with_labels.csv"
  "ncfy|ncfy_retrospective_en_with_labels.csv"
  "nyfs|nyfs_prospective_with_labels_en_updated.csv"
  "zsfs|zsfs_prospective_with_labels_en_dedup_updated.csv"
  "zsfs|zsfs_retrospective_with_labels_en_dedup_updated.csv"
  "zsfy|zhognshanfuyi_with_labels_qwen3-30b_single_region_updated.csv"
)

# 与各 test_<ds>_all_by_region.sh 完全一致的 14 个 baseline 方法。
MODELS=(
  # ark_base
  # ark_large
  # chess
  # chexzero
  # gloria
  biomedclip
  # maco
  # rad_dino
  # siglip2
  # topk_v97_last
  # topk_v97_best
  # topk_v95_last
  # topk_v95_best
  # topk_v83_last
  # topk_v100_best
  # topk_v14_last
  # topk_v19_last
  # topk_v21_last
)

for MODEL_ID in "${MODELS[@]}"; do
  CKPT_MODEL_ID="${MODEL_ID}"
  RESULT_MODEL_ID="${MODEL_ID}"
  FEATURE_MODEL_ID="${MODEL_ID}"
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
echo "✓ External 4 Datasets × 14 Methods by-region testing complete (GPU=$CUDA_VISIBLE_DEVICES)"
