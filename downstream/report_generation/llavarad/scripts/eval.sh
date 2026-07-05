#!/bin/bash

set -e
set -o pipefail

# Default to 3 GPUs, can be overridden externally.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
export CUDA_VISIBLE_DEVICES

model_base=lmsys/vicuna-7b-v1.5
model_path=microsoft/llava-rad

model_base="${1:-$model_base}"
model_path="${2:-$model_path}"
prediction_dir="${3:-results/llavarad}"
prediction_file=$prediction_dir/test

run_name="${4:-llavarad}"


query_file=${SMARTX_DATA_ROOT}/MIMIC-CXR/labels/chat_test_MIMIC_CXR_all_gpt4extract_rulebased_v1.json

image_folder=${SMARTX_DATA_ROOT}/MIMIC-CXR/files

loader="mimic_test_findings"
conv_mode="v1"

IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
CHUNKS=${#GPU_IDS[@]}

for (( idx=0; idx<$CHUNKS; idx++ ))
do
    CUDA_VISIBLE_DEVICES=${GPU_IDS[$idx]} python -m llava.eval.model_mimic_cxr \
        --query_file ${query_file} \
        --loader ${loader} \
        --image_folder ${image_folder} \
        --conv_mode ${conv_mode} \
        --prediction_file ${prediction_file}_${idx}.jsonl \
        --temperature 0 \
        --model_path ${model_path} \
        --model_base ${model_base} \
        --chunk_idx ${idx} \
        --num_chunks ${CHUNKS} \
        --batch_size 2 \
        --group_by_length &
done

wait

cat ${prediction_file}_*.jsonl > mimic_cxr_preds.jsonl

pushd llava/eval/rrg_eval
WANDB_PROJECT="llava" WANDB_RUN_ID="llava-eval-$(date +%Y%m%d%H%M%S)" WANDB_RUN_GROUP=evaluate CUDA_VISIBLE_DEVICES=0 \
    python run.py ../../../mimic_cxr_preds.jsonl --run_name ${run_name} --output_dir ../../../${prediction_dir}/eval
popd

rm mimic_cxr_preds.jsonl