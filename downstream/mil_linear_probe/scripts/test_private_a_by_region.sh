#!/bin/bash
# Private A in-domain test：按 region 用 best ckpt + 最新 val_thresholds.json 重跑 test split，
# 覆盖各实验 version_* 目录下的 test_results.csv（供 bootstrap / delong 使用）。
#
# 用法：
#   bash scripts/test_private_a_by_region.sh topk_v19_last abmil_full
#   bash scripts/test_private_a_by_region.sh topk_v14_last abmil_simple
#
# 环境变量（可选）：
#   CUDA_VISIBLE_DEVICES   默认 0
#   CKPT_MODEL_ID          查 MIL_LP 实验目录用的 model 子目录名，默认同第一个参数
#   FEATURE_MODEL_ID       特征目录下的 model_id，默认同第一个参数
#   PRELOAD_FEATURES       true/false，默认 true（读 preload_features/*.pt，远快于逐文件读 NAS）

set -e

MODEL_ID="${1:?用法: $0 <model_id> <mil_type>，例如 topk_v19_last abmil_full}"
MIL_TYPE="${2:?用法: $0 <model_id> <mil_type>，例如 topk_v19_last abmil_full}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FEATURES_ROOT="${SMARTX_OUTPUT_ROOT}/features/private_a"
CKPT_ROOT="${SMARTX_OUTPUT_ROOT}/experiments/MIL_LP"
CONFIG="configs/train_private_a_our_ark_large.yaml"

CKPT_MODEL_ID="${CKPT_MODEL_ID:-$MODEL_ID}"
FEATURE_MODEL_ID="${FEATURE_MODEL_ID:-$MODEL_ID}"
PRELOAD_FEATURES="${PRELOAD_FEATURES:-true}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate vlp
fi

REGION_CKPT_JSON="$(mktemp "/tmp/private_a_${CKPT_MODEL_ID}_${MIL_TYPE}_ckpt_map.XXXXXX.json")"
cleanup() { rm -f "${REGION_CKPT_JSON}"; }
trap cleanup EXIT

echo "============================================================"
echo "Private A in-domain retest"
echo "  MODEL_ID=${MODEL_ID}"
echo "  CKPT_MODEL_ID=${CKPT_MODEL_ID}"
echo "  FEATURE_MODEL_ID=${FEATURE_MODEL_ID}"
echo "  MIL_TYPE=${MIL_TYPE}"
echo "  PRELOAD_FEATURES=${PRELOAD_FEATURES}"
echo "============================================================"

python build_region_ckpt_map.py \
  --ckpt_root "${CKPT_ROOT}" \
  --dataset_name private_a \
  --model_id "${CKPT_MODEL_ID}" \
  --mil_type "${MIL_TYPE}" \
  --output "${REGION_CKPT_JSON}"

python test_mil_lp_private_a_by_region.py \
  --config "${CONFIG}" \
  --dataset_config configs/dataset_dir.yaml \
  --features_dir "${FEATURES_ROOT}" \
  --model_id "${FEATURE_MODEL_ID}" \
  --region_ckpt_map "${REGION_CKPT_JSON}" \
  --split test \
  --preload_features "${PRELOAD_FEATURES}" \
  --results_dir experiment
