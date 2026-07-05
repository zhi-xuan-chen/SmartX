#!/bin/bash
# Evaluate the original LLaVA-Rad model (microsoft/llava-rad) on the SAME CheXpert-Plus test set
# (official `validate` split of labels/chexpert_processed_grouped.csv) used by the custom-encoder
# evals, so NLG/CE metrics are directly comparable.
#
# Usage: bash scripts/eval_llavarad_chexpert.sh [model_base] [model_path] [run_name] [prediction_dir] [eval_batch_size]

set -e
set -o pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

model_base="${1:-lmsys/vicuna-7b-v1.5}"
model_path="${2:-microsoft/llava-rad}"
run_name="${3:-llavarad_chexpert_eval}"
prediction_dir="${4:-results/${run_name}}"
eval_batch_size="${5:-2}"

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
    CUDA_VISIBLE_DEVICES=${GPU_IDS[$idx]} python -m llava.eval.model_mimic_cxr \
        --model_path ${model_path} \
        --model_base ${model_base} \
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

merged_preds="${PROJECT_ROOT}/chexpert_cxr_preds_llavarad.jsonl"
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
