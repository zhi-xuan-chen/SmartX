#!/bin/bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL="our_ark_large"
PRETRAINED_DEFAULT="${SMARTX_DATA_ROOT}/pretrained_weights/topk_v19_last.ckpt"
STAGE1_BATCH_SIZE_DEFAULT="16"
STAGE2_BATCH_SIZE_DEFAULT="8"
EVAL_BATCH_SIZE_DEFAULT="1"

mode="${1:-all}"
llavarad_merged="${2:-${SMARTX_OUTPUT_ROOT}/llavarad_merged}"
pretrained_path="${3:-${PRETRAINED_DEFAULT}}"
checkpoint_root="${CHECKPOINT_ROOT:-${SMARTX_OUTPUT_ROOT}/XRay-VLP/experiments/RG/iu_llavarad}"

# Derive checkpoint suffix from pretrained weight name (e.g. our_ark_large or topk_xxx_last) + _ar
suffix=$(basename "${pretrained_path%.*}")_ar

if [[ "${mode}" == "all" ]]; then
    bash "$0" stage1 "${llavarad_merged}" "${3:-${PRETRAINED_DEFAULT}}" "${4:-${STAGE1_BATCH_SIZE_DEFAULT}}"
    bash "$0" stage2 "${llavarad_merged}" "${3:-${PRETRAINED_DEFAULT}}" "${5:-${STAGE2_BATCH_SIZE_DEFAULT}}"
    bash "$0" eval "${llavarad_merged}" "${6:-${checkpoint_root}/llavarad_${suffix}_stage2}" "${7:-llavarad_${suffix}_iu_eval}" "${8:-${PROJECT_ROOT}/results/llavarad_${suffix}_iu_eval}" "${9:-${EVAL_BATCH_SIZE_DEFAULT}}"
    exit 0
fi

case "${mode}" in
    stage1)
        pretrained_path="${3:-${PRETRAINED_DEFAULT}}"
        stage1_batch_size="${4:-${STAGE1_BATCH_SIZE_DEFAULT}}"
        output_dir="${checkpoint_root}/llavarad_${suffix}_stage1"
        bash "${SCRIPT_DIR}/train_stage1_custom_iu.sh" "${llavarad_merged}" "${MODEL}" "${pretrained_path}" "${output_dir}" "${stage1_batch_size}"
        ;;
    stage2)
        pretrained_path="${3:-${PRETRAINED_DEFAULT}}"
        stage2_batch_size="${4:-${STAGE2_BATCH_SIZE_DEFAULT}}"
        output_dir="${checkpoint_root}/llavarad_${suffix}_stage2"
        stage1_output_dir="${checkpoint_root}/llavarad_${suffix}_stage1"
        bash "${SCRIPT_DIR}/train_stage2_custom_iu.sh" "${llavarad_merged}" "${MODEL}" "${pretrained_path}" "${stage1_output_dir}" "${output_dir}" "${stage2_batch_size}"
        ;;
    eval)
        checkpoint_root_dir="${3:-${checkpoint_root}/llavarad_${suffix}_stage2}"
        run_name="${4:-llavarad_${suffix}_iu_eval}"
        prediction_base="${5:-${PROJECT_ROOT}/results/${run_name}}"
        eval_batch_size="${6:-${EVAL_BATCH_SIZE_DEFAULT}}"
        bash "${SCRIPT_DIR}/eval_all_checkpoints.sh" iu \
            "${llavarad_merged}" "${checkpoint_root_dir}" "${run_name}" "${prediction_base}" "${eval_batch_size}"
        ;;
    *)
        echo "Unknown mode: ${mode}"
        echo "Supported modes: all | stage1 | stage2 | eval"
        exit 1
        ;;
esac
