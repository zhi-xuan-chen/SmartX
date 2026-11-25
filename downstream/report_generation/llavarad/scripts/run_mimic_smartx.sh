#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

MODEL="smartx_large"
STAGE1_BATCH_SIZE_DEFAULT="16"
STAGE2_BATCH_SIZE_DEFAULT="8"
EVAL_BATCH_SIZE_DEFAULT="1"

REPORT_OUTPUT_ROOT="${REPORT_OUTPUT_ROOT:-${REPO_ROOT}/outputs/report_generation}"
export REPORT_OUTPUT_ROOT

mode="${1:-all}"
llavarad_merged="${2:-${REPORT_OUTPUT_ROOT}/llavarad_merged}"
checkpoint_root="${CHECKPOINT_ROOT:-${REPORT_OUTPUT_ROOT}/mimic}"

resolve_checkpoint() {
    if [[ -n "${1:-}" ]]; then
        printf '%s\n' "$1"
    else
        python -c 'from downstream.modules import resolve_smartx_checkpoint; print(resolve_smartx_checkpoint())'
    fi
}

if [[ "${mode}" == "all" ]]; then
    pretrained_path="$(resolve_checkpoint "${3:-${SMARTX_CHECKPOINT:-}}")"
    bash "$0" stage1 "${llavarad_merged}" "${pretrained_path}" "${4:-${STAGE1_BATCH_SIZE_DEFAULT}}"
    bash "$0" stage2 "${llavarad_merged}" "${pretrained_path}" "${5:-${STAGE2_BATCH_SIZE_DEFAULT}}"
    bash "$0" eval "${llavarad_merged}" "${6:-${checkpoint_root}/llavarad_smartx_large_stage2}" "${7:-smartx_large_mimic_eval}" "${8:-${PROJECT_ROOT}/results/smartx_large_mimic_eval}" "${9:-${EVAL_BATCH_SIZE_DEFAULT}}"
    exit 0
fi

case "${mode}" in
    stage1)
        pretrained_path="$(resolve_checkpoint "${3:-${SMARTX_CHECKPOINT:-}}")"
        stage1_batch_size="${4:-${STAGE1_BATCH_SIZE_DEFAULT}}"
        output_dir="${checkpoint_root}/llavarad_smartx_large_stage1"
        bash "${SCRIPT_DIR}/train_stage1_custom_mimic.sh" "${llavarad_merged}" "${MODEL}" "${pretrained_path}" "${output_dir}" "${stage1_batch_size}"
        ;;
    stage2)
        pretrained_path="$(resolve_checkpoint "${3:-${SMARTX_CHECKPOINT:-}}")"
        stage2_batch_size="${4:-${STAGE2_BATCH_SIZE_DEFAULT}}"
        output_dir="${checkpoint_root}/llavarad_smartx_large_stage2"
        stage1_output_dir="${checkpoint_root}/llavarad_smartx_large_stage1"
        bash "${SCRIPT_DIR}/train_stage2_custom_mimic.sh" "${llavarad_merged}" "${MODEL}" "${pretrained_path}" "${stage1_output_dir}" "${output_dir}" "${stage2_batch_size}"
        ;;
    eval)
        checkpoint_dir="${3:-${checkpoint_root}/llavarad_smartx_large_stage2}"
        run_name="${4:-smartx_large_mimic_eval}"
        prediction_dir="${5:-${PROJECT_ROOT}/results/${run_name}}"
        eval_batch_size="${6:-${EVAL_BATCH_SIZE_DEFAULT}}"
        bash "${SCRIPT_DIR}/eval_custom_mimic.sh" \
            "${llavarad_merged}" "${checkpoint_dir}" "${run_name}" "${prediction_dir}" "${eval_batch_size}"
        ;;
    *)
        echo "Unknown mode: ${mode}"
        echo "Supported modes: all | stage1 | stage2 | eval"
        exit 1
        ;;
esac
