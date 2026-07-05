#!/bin/bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL="chess"
PRETRAINED_DEFAULT=""
STAGE1_BATCH_SIZE_DEFAULT="32"
STAGE2_BATCH_SIZE_DEFAULT="16"
EVAL_BATCH_SIZE_DEFAULT="2"

mode="${1:-all}"
llavarad_merged="${2:-${SMARTX_OUTPUT_ROOT}/llavarad_merged}"
checkpoint_root="${CHECKPOINT_ROOT:-${SMARTX_OUTPUT_ROOT}/XRay-VLP/experiments/RG/chexpert_llavarad}"

if [[ "${mode}" == "all" ]]; then
    bash "$0" stage1 "${llavarad_merged}" "${3:-${PRETRAINED_DEFAULT}}" "${4:-${STAGE1_BATCH_SIZE_DEFAULT}}"
    bash "$0" stage2 "${llavarad_merged}" "${3:-${PRETRAINED_DEFAULT}}" "${5:-${STAGE2_BATCH_SIZE_DEFAULT}}"
    bash "$0" eval "${llavarad_merged}" "${6:-${checkpoint_root}/llavarad_${MODEL}_stage2}" "${7:-llavarad_${MODEL}_chexpert_eval}" "${8:-${PROJECT_ROOT}/results/llavarad_${MODEL}_chexpert_eval}" "${9:-${EVAL_BATCH_SIZE_DEFAULT}}"
    exit 0
fi

case "${mode}" in
    stage1)
        pretrained_path="${3:-${PRETRAINED_DEFAULT}}"
        stage1_batch_size="${4:-${STAGE1_BATCH_SIZE_DEFAULT}}"
        output_dir="${checkpoint_root}/llavarad_${MODEL}_stage1"
        bash "${SCRIPT_DIR}/train_stage1_custom_chexpert.sh" "${llavarad_merged}" "${MODEL}" "${pretrained_path}" "${output_dir}" "${stage1_batch_size}"
        ;;
    stage2)
        pretrained_path="${3:-${PRETRAINED_DEFAULT}}"
        stage2_batch_size="${4:-${STAGE2_BATCH_SIZE_DEFAULT}}"
        output_dir="${checkpoint_root}/llavarad_${MODEL}_stage2"
        stage1_output_dir="${checkpoint_root}/llavarad_${MODEL}_stage1"
        bash "${SCRIPT_DIR}/train_stage2_custom_chexpert.sh" "${llavarad_merged}" "${MODEL}" "${pretrained_path}" "${stage1_output_dir}" "${output_dir}" "${stage2_batch_size}"
        ;;
    eval)
        checkpoint_dir="${3:-${checkpoint_root}/llavarad_${MODEL}_stage2}"
        run_name="${4:-llavarad_${MODEL}_chexpert_eval}"
        prediction_dir="${5:-${PROJECT_ROOT}/results/${run_name}}"
        eval_batch_size="${6:-${EVAL_BATCH_SIZE_DEFAULT}}"
        bash "${SCRIPT_DIR}/eval_custom_chexpert.sh" "${llavarad_merged}" "${checkpoint_dir}" "${run_name}" "${prediction_dir}" "${eval_batch_size}"
        ;;
    *)
        echo "Unknown mode: ${mode}"
        echo "Supported modes: all | stage1 | stage2 | eval"
        exit 1
        ;;
esac
