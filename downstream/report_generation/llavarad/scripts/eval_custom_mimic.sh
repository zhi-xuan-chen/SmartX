#!/bin/bash
# Evaluate a custom-encoder LLaVA-Rad checkpoint on MIMIC-CXR test set.
# Evaluates an explicit Stage 2 (LoRA) checkpoint.
#
# Usage: bash scripts/eval_custom_mimic.sh <llavarad_merged> <checkpoint_dir> [run_name] [prediction_dir] [eval_batch_size]

set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

# Ensure local llava package is imported instead of any other workspace package.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

llavarad_merged="${1:-}"
checkpoint_dir="${2:-}"
run_name="${3:-llavarad_custom}"
prediction_dir="${4:-results/${run_name}}"
eval_batch_size="${5:-2}"

if [ -z "${llavarad_merged}" ] || [ -z "${checkpoint_dir}" ]; then
    echo "Usage: bash scripts/eval_custom_mimic.sh <llavarad_merged> <checkpoint_dir> [run_name] [prediction_dir] [eval_batch_size]"
    echo "  llavarad_merged : path to merged LLaVA-Rad base model"
    echo "  checkpoint_dir  : Stage 2 output directory"
    exit 1
fi

# Resolve to absolute path before pushd later
case "${prediction_dir}" in
    /*) ;;
    *) prediction_dir="${PROJECT_ROOT}/${prediction_dir}" ;;
esac

prediction_file=${prediction_dir}/test
if [[ -z "${MIMIC_ROOT:-}" ]]; then
    : "${SMARTX_DATA_ROOT:?Set SMARTX_DATA_ROOT or a dataset-specific root}"
    MIMIC_ROOT="${SMARTX_DATA_ROOT}/MIMIC-CXR"
fi
query_file="${MIMIC_ROOT}/labels/mimic_cxr_test_llava_rad_frontal_2461.csv"
image_folder="${MIMIC_ROOT}/imgs-1024"

[[ -d "${llavarad_merged}" ]] || { echo "Missing merged LLaVA-Rad model: ${llavarad_merged}" >&2; exit 1; }
[[ -d "${checkpoint_dir}" ]] || { echo "Missing Stage 2 checkpoint: ${checkpoint_dir}" >&2; exit 1; }
[[ -f "${query_file}" ]] || { echo "Missing evaluation manifest: ${query_file}" >&2; exit 1; }
[[ -d "${image_folder}" ]] || { echo "Missing image directory: ${image_folder}" >&2; exit 1; }

IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
CHUNKS=${#GPU_IDS[@]}

mkdir -p "${prediction_dir}"

pids=()
for (( idx=0; idx<$CHUNKS; idx++ )); do
    CUDA_VISIBLE_DEVICES=${GPU_IDS[$idx]} python -m llava.eval.model_mimic_cxr_custom \
        --model_path "${checkpoint_dir}" \
        --model_base "${llavarad_merged}" \
        --query_file "${query_file}" \
        --loader mimic_test_findings \
        --image_folder "${image_folder}" \
        --conv_mode v1 \
        --prediction_file "${prediction_file}_${idx}.jsonl" \
        --temperature 0 \
        --num_beams 1 \
        --chunk_idx ${idx} \
        --num_chunks ${CHUNKS} \
        --batch_size "${eval_batch_size}" \
        --group_by_length &
    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || { echo "ERROR: inference subprocess $pid failed"; fail=1; }
done
[ $fail -eq 0 ] || exit 1

merged_preds="${prediction_dir}/predictions.jsonl"
: > "${merged_preds}"
for (( idx=0; idx<CHUNKS; idx++ )); do
    cat "${prediction_file}_${idx}.jsonl" >> "${merged_preds}"
done

export WANDB_MODE="${WANDB_MODE:-disabled}"
pushd "${PROJECT_ROOT}/llava/eval/rrg_eval"
WANDB_PROJECT="llava" \
WANDB_RUN_ID="llava-eval-$(date +%Y%m%d%H%M%S)" \
WANDB_RUN_GROUP=evaluate \
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" \
    python run.py "${merged_preds}" \
    --run_name "${run_name}" \
    --output_dir "${prediction_dir}/eval"
popd
