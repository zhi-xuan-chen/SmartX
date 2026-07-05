#!/bin/bash
# Stage 2: LoRA fine-tune LLM + continue training mm_projector on IU X-Ray
# Usage: bash scripts/train_stage2_custom_iu.sh [llavarad_merged] [encoder_name] [pretrained_path(only for our*)] [stage1_output_dir(optional)] [output_dir] [train_batch_size]
# Supported custom vision encoder names:
#   ark_base | ark_large | our_ark_base | our_ark_large | rad_dino | gloria | chexzero | chess | biomedclip | conceptclip | maco

set -e
set -o pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

llavarad_merged="${1:-${SMARTX_OUTPUT_ROOT}/llavarad_merged}"
custom_vision_encoder="${2:-ark_base}"
custom_vision_pretrained_path="${3:-${SMARTX_DATA_ROOT}/pretrained_weights/our_ark_base.pth}"
stage1_output_dir="${4:-${SMARTX_OUTPUT_ROOT}/XRay-VLP/experiments/RG/iu_llavarad/llavarad_${custom_vision_encoder}_stage1}"
output_dir="${5:-${SMARTX_OUTPUT_ROOT}/XRay-VLP/experiments/RG/iu_llavarad/llavarad_${custom_vision_encoder}_stage2}"
train_batch_size="${6:-24}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-2}"

data_path=${SMARTX_DATA_ROOT}/IUXray/labels/indiana_meta_frontal_findings_w_chexbert_labels_train.csv
image_folder=${SMARTX_DATA_ROOT}/IUXray/images/images_normalized

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
    --loader iu_train_findings
    --image_folder "${image_folder}"
    --is_multimodal True
    --image_aspect_ratio square
    --stage stage2
    --lora_enable True
    --lora_r 64
    --lora_alpha 128
    --lora_dropout 0.05
    --bf16 True
    --output_dir "${output_dir}"
    --num_train_epochs 3
    --per_device_train_batch_size "${train_batch_size}"
    --per_device_eval_batch_size 4
    --gradient_accumulation_steps "${gradient_accumulation_steps}"
    --evaluation_strategy no
    --save_strategy steps
    --save_steps 20
    --learning_rate 2e-4
    --weight_decay 0.
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --logging_steps 1
    --model_max_length 2048
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --report_to wandb
    --run_name "llavarad_${custom_vision_encoder}_stage2_iu"
)

if [[ "${custom_vision_encoder}" == our_* ]]; then
    ARGS+=(--custom_vision_pretrained_path "${custom_vision_pretrained_path}")
else
    ARGS+=(--save_total_limit 2)
fi

if [[ -n "${stage1_output_dir}" && "${stage1_output_dir,,}" != "none" && "${stage1_output_dir,,}" != "null" ]]; then
    ARGS+=(--stage1_output_dir "${stage1_output_dir}")
fi

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --module llava.train.train_llavarad_custom "${ARGS[@]}"
