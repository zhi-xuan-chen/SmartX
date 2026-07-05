#!/bin/bash
# Evaluate a custom-encoder LLaVA-Rad checkpoint on the CheXpert-Plus test set
# (official `validate` split of labels/chexpert_processed_grouped.csv).
# Supports both Stage 1 (projector-only) and Stage 2 (LoRA) checkpoints.
# The checkpoint dir name must contain "llava" (e.g. llavarad_ark_base_stage1 / llavarad_ark_base_stage2).
#
# Usage: bash scripts/eval_custom_chexpert.sh <llavarad_merged> <checkpoint_dir> [run_name] [prediction_dir] [eval_batch_size]

set -e
set -o pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

# Ensure local llava package is imported instead of any other workspace package.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

llavarad_merged="${1}"
checkpoint_dir="${2}"
run_name="${3:-llavarad_custom_chexpert}"
prediction_dir="${4:-results/${run_name}}"
eval_batch_size="${5:-2}"

if [ -z "${llavarad_merged}" ] || [ -z "${checkpoint_dir}" ]; then
    echo "Usage: bash scripts/eval_custom_chexpert.sh <llavarad_merged> <checkpoint_dir> [run_name] [prediction_dir] [eval_batch_size]"
    echo "  llavarad_merged : path to merged LLaVA-Rad base model"
    echo "  checkpoint_dir  : Stage 1/Stage 2 output dir (must contain 'llava' in its basename)"
    exit 1
fi

# Resolve to absolute path before pushd later
case "${prediction_dir}" in
    /*) ;;
    *) prediction_dir="${PROJECT_ROOT}/${prediction_dir}" ;;
esac

prediction_file=${prediction_dir}/test
CHEXPERT_ROOT="${CHEXPERT_ROOT:-${SMARTX_DATA_ROOT}/CheXpert-Plus}"
query_file="${CHEXPERT_ROOT}/labels/chexpert_processed_grouped.csv"
image_folder="${CHEXPERT_ROOT}/imgs-1024"

IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
CHUNKS=${#GPU_IDS[@]}

mkdir -p ${prediction_dir}

pids=()
for (( idx=0; idx<$CHUNKS; idx++ )); do
    CUDA_VISIBLE_DEVICES=${GPU_IDS[$idx]} python -m llava.eval.model_mimic_cxr_custom \
        --model_path ${checkpoint_dir} \
        --model_base ${llavarad_merged} \
        --query_file ${query_file} \
        --loader chexpert_test_findings \
        --image_folder ${image_folder} \
        --conv_mode v1 \
        --prediction_file ${prediction_file}_${idx}.jsonl \
        --temperature 0 \
        --chunk_idx ${idx} \
        --num_chunks ${CHUNKS} \
        --batch_size ${eval_batch_size} \
        --group_by_length &
    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || { echo "ERROR: inference subprocess $pid failed"; fail=1; }
done
[ $fail -eq 0 ] || exit 1

merged_preds="${PROJECT_ROOT}/chexpert_cxr_preds.jsonl"
cat ${prediction_file}_*.jsonl > "${merged_preds}"

pushd "${PROJECT_ROOT}/llava/eval/rrg_eval"
WANDB_PROJECT="llava" \
WANDB_RUN_ID="llava-eval-$(date +%Y%m%d%H%M%S)" \
WANDB_RUN_GROUP=evaluate \
CUDA_VISIBLE_DEVICES=0 \
    python run.py "${merged_preds}" \
    --run_name ${run_name} \
    --output_dir "${prediction_dir}/eval"
popd

rm "${merged_preds}"
