#!/bin/bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="ark_base"
EVAL_BATCH_SIZE_DEFAULT="2"

llavarad_merged="${1:-${SMARTX_OUTPUT_ROOT}/llavarad_merged}"
checkpoint_root="${CHECKPOINT_ROOT:-/scratch/medimgfmod/zchenhi/checkpoints/XRay-VLP/experiments/RG/mimic_llavarad}"
checkpoint_dir="${2:-${checkpoint_root}/llavarad_${MODEL}_stage2}"
run_name="${3:-llavarad_${MODEL}_mimic_eval}"
prediction_dir="${4:-results/${run_name}}"
eval_batch_size="${5:-${EVAL_BATCH_SIZE_DEFAULT}}"

bash "${SCRIPT_DIR}/eval_custom_mimic.sh" \
    "${llavarad_merged}" \
    "${checkpoint_dir}" \
    "${run_name}" \
    "${prediction_dir}" \
    "${eval_batch_size}"
