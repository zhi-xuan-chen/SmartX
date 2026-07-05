#!/bin/bash
# Stage 1: train mm_projector only (LLM + vision encoder frozen) on CheXpert-Plus.
# Usage: bash scripts/train_stage1_custom_chexpert.sh [llavarad_merged] [encoder_name] [pretrained_path(only for our*)] [output_dir] [train_batch_size]
# Supported custom vision encoder names:
#   ark_base | ark_large | our_ark_base | our_ark_large | rad_dino | gloria | chexzero | chess | biomedclip | conceptclip | maco
# Notes:
#   - Only our_* encoders require --custom_vision_pretrained_path.
#   - Other encoders should use their default config (do not pass pretrained path).
#   - Data source mirrors pre-training: labels/chexpert_processed_grouped.csv, filtered by the `split` column.
#   - Supported loaders: chexpert_train_findings (split=train) | chexpert_test_findings (split=validate)

set -e
set -o pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

llavarad_merged="${1:-${SMARTX_OUTPUT_ROOT}/llavarad_merged}"
custom_vision_encoder="${2:-gloria}"
custom_vision_pretrained_path="${3:-${SMARTX_DATA_ROOT}/pretrained_weights/our_ark_base.pth}"
output_dir="${4:-${SMARTX_OUTPUT_ROOT}/XRay-VLP/experiments/RG/chexpert_llavarad/llavarad_${custom_vision_encoder}_stage1}"
train_batch_size="${5:-48}"

CHEXPERT_ROOT="${CHEXPERT_ROOT:-${SMARTX_DATA_ROOT}/CheXpert-Plus}"
data_path="${CHEXPERT_ROOT}/labels/chexpert_processed_grouped.csv"
image_folder="${CHEXPERT_ROOT}/imgs-1024"

IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=${#GPU_IDS[@]}
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-65000 -n 1)}"

# Ensure local llava package is imported instead of any site-packages llava.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

mkdir -p ${output_dir}

ARGS=(
    --model_name_or_path "${llavarad_merged}"
    --version v1
    --custom_vision_encoder "${custom_vision_encoder}"
    --mm_projector_type mlp2x_gelu
    --data_path "${data_path}"
    --loader chexpert_train_findings
    --image_folder "${image_folder}"
    --is_multimodal True
    --image_aspect_ratio square
    --stage stage1
    --bf16 True
    --output_dir "${output_dir}"
    --num_train_epochs 1
    --per_device_train_batch_size "${train_batch_size}"
    --per_device_eval_batch_size 4
    --gradient_accumulation_steps 1
    --evaluation_strategy no
    --save_strategy steps
    --save_steps 500
    --save_total_limit 2
    --learning_rate 1e-3
    --weight_decay 0.
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --logging_steps 1
    --model_max_length 2048
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --report_to wandb
    --run_name "llavarad_${custom_vision_encoder}_chexpert_stage1"
)

if [[ "${custom_vision_encoder}" == our_* ]]; then
    ARGS+=(--custom_vision_pretrained_path "${custom_vision_pretrained_path}")
fi

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --module llava.train.train_llavarad_custom "${ARGS[@]}"
