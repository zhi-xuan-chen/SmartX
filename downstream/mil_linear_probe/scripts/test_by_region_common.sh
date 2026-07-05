#!/bin/bash
# 通用函数：按 Region 测试指定 dataset + model

set -e

infer_result_dataset_name() {
  local DATASET="$1"
  local CSV_NAME="$2"
  local LOWER
  LOWER="$(echo "${CSV_NAME}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${LOWER}" == *"prospective"* ]]; then
    echo "${DATASET}_pro"
  elif [[ "${LOWER}" == *"retrospective"* ]]; then
    echo "${DATASET}_ret"
  else
    echo "${DATASET}"
  fi
}

run_region_test() {
  local DATASET="$1"          # nyfs / zsfs / zsfy / ncfy
  local MODEL_ID="$2"         # 逻辑模型名（配置项）
  local MIL_TYPE="$3"         # abmil_simple / ...
  local FEATURES_ROOT_BASE="$4"
  local CKPT_ROOT="$5"
  local OUTPUT_ROOT="$6"
  local CKPT_MODEL_ID="${7:-$MODEL_ID}"   # 用于 experiments/MIL_LP 下查找 checkpoint 的 model_id（可映射）
  local CSV_NAME="${8:-}"                 # 可选：评估所用 CSV 文件名
  local RESULT_DATASET_NAME="${9:-}"      # 可选：结果目录中的数据集名，默认按 CSV 自动映射
  local RESULT_MODEL_ID="${10:-}"         # 可选：结果目录中的模型名（默认等于 MODEL_ID）
  local FEATURE_MODEL_ID="${11:-}"        # 可选：特征目录中的模型名（默认等于 MODEL_ID）
  local PRELOAD_FEATURES="false"          # 调试方便，按需从磁盘读取特征

  local TEST_PY="test_mil_lp_${DATASET}_by_region.py"
  local EVAL_CFG="configs/eval_${DATASET}_mil.yaml"
  local FEATURE_ROOT="${FEATURES_ROOT_BASE}/${DATASET}"
  if [ -z "${RESULT_DATASET_NAME}" ]; then
    if [ -n "${CSV_NAME}" ]; then
      RESULT_DATASET_NAME="$(infer_result_dataset_name "${DATASET}" "${CSV_NAME}")"
    else
      RESULT_DATASET_NAME="${DATASET}"
    fi
  fi
  if [ -z "${RESULT_MODEL_ID}" ]; then
    RESULT_MODEL_ID="${MODEL_ID}"
  fi
  if [ -z "${FEATURE_MODEL_ID}" ]; then
    FEATURE_MODEL_ID="${MODEL_ID}"
  fi
  local OUTPUT_DIR="${OUTPUT_ROOT}/${MIL_TYPE}/${RESULT_DATASET_NAME}"

  if [ ! -f "${TEST_PY}" ]; then
    echo "[跳过] ${DATASET}/${MODEL_ID}: 未找到 ${TEST_PY}"
    return 0
  fi
  if [ ! -d "${FEATURE_ROOT}" ]; then
    echo "[跳过] ${DATASET}/${MODEL_ID}: 特征目录不存在 ${FEATURE_ROOT}"
    return 0
  fi

  local REGION_CKPT_JSON
  REGION_CKPT_JSON="$(mktemp "/tmp/${DATASET}_${MODEL_ID}_region_ckpt_map.XXXXXX.json")"
  cleanup_one() { rm -f "${REGION_CKPT_JSON}"; }
  trap cleanup_one RETURN

  echo "============================================================"
  echo "Dataset: ${DATASET} | ResultDataset: ${RESULT_DATASET_NAME} | MIL: ${MIL_TYPE} | Model: ${MODEL_ID} | CkptModel: ${CKPT_MODEL_ID} | FeatureModel: ${FEATURE_MODEL_ID} | ResultModel: ${RESULT_MODEL_ID}"
  if [ -n "${CSV_NAME}" ]; then
    echo "CSV: ${CSV_NAME}"
  fi
  echo "Preload features: ${PRELOAD_FEATURES}"
  echo "============================================================"

  python build_region_ckpt_map.py \
    --ckpt_root "${CKPT_ROOT}" \
    --dataset_name private_a \
    --model_id "${CKPT_MODEL_ID}" \
    --mil_type "${MIL_TYPE}" \
    --output "${REGION_CKPT_JSON}"

  local CSV_ARG=()
  if [ -n "${CSV_NAME}" ]; then
    CSV_ARG=(--csv_name "${CSV_NAME}")
  fi
  local RESULT_MODEL_ARG=()
  if [ -n "${RESULT_MODEL_ID}" ]; then
    RESULT_MODEL_ARG=(--result_model_id "${RESULT_MODEL_ID}")
  fi

  python "${TEST_PY}" \
    --config "${EVAL_CFG}" \
    --dataset_config configs/dataset_dir.yaml \
    --features_dir "${FEATURE_ROOT}" \
    --model_id "${FEATURE_MODEL_ID}" \
    --preload_features "${PRELOAD_FEATURES}" \
    --region_ckpt_map "${REGION_CKPT_JSON}" \
    --output_dir "${OUTPUT_DIR}" \
    "${CSV_ARG[@]}" \
    "${RESULT_MODEL_ARG[@]}"
}

