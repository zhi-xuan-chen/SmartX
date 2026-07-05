#!/bin/bash
# Evaluate a Stage 1 (projector-only) LLaVA-Rad checkpoint on MIMIC-CXR test set.
# Usage: bash scripts/eval_stage1_custom_mimic.sh [model_name] [checkpoint_dir] [run_name] [prediction_dir] [eval_batch_size]
#   model_name     : e.g. ark_base, ark_large, maco, etc. (used to find default stage1 checkpoint)
#   checkpoint_dir : override stage1 checkpoint path

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

model_name="${1:-ark_base}"
checkpoint_root="${CHECKPOINT_ROOT:-${SMARTX_OUTPUT_ROOT}/XRay-VLP/experiments/RG/mimic_llavarad}"
checkpoint_dir="${2:-${checkpoint_root}/llavarad_${model_name}_stage1}"
run_name="${3:-llavarad_${model_name}_stage1_mimic_eval}"
prediction_dir="${4:-results/${run_name}}"
eval_batch_size="${5:-2}"

# Resolve to absolute path before pushd later
case "${prediction_dir}" in
    /*) ;;
    *) prediction_dir="${PROJECT_ROOT}/${prediction_dir}" ;;
esac

# If the checkpoint root has no config.json, fallback to latest checkpoint-* subdir
if [ ! -f "${checkpoint_dir}/config.json" ]; then
    latest_ckpt=$(ls -d "${checkpoint_dir}"/checkpoint-* 2>/dev/null | sort -t'-' -k2 -rn | head -1)
    if [ -n "${latest_ckpt}" ] && [ -f "${latest_ckpt}/config.json" ]; then
        echo "  (no config.json in root, using latest checkpoint subdir)"
        checkpoint_dir="${latest_ckpt}"
    fi
fi

echo "Evaluating Stage 1 checkpoint:"
echo "  checkpoint_dir : ${checkpoint_dir}"
echo "  run_name       : ${run_name}"
echo "  prediction_dir : ${prediction_dir}"
echo "  batch_size     : ${eval_batch_size}"

bash "${SCRIPT_DIR}/eval_custom_mimic.sh" \
    "${SMARTX_OUTPUT_ROOT}/llavarad_merged" \
    "${checkpoint_dir}" \
    "${run_name}" \
    "${prediction_dir}" \
    "${eval_batch_size}"
