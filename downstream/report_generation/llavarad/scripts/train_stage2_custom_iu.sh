#!/bin/bash
# Stage 2: LoRA fine-tune LLM + continue training mm_projector on IU X-Ray
# Usage: bash scripts/train_stage2_custom_iu.sh [llavarad_merged] [encoder_name] [pretrained_path] [stage1_output_dir(optional)] [output_dir] [train_batch_size]
# Supported custom vision encoder names:
#   smartx_large (other benchmark encoders are supported separately)

set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

: "${SMARTX_DATA_ROOT:?Set SMARTX_DATA_ROOT to the report-generation data root}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
REPORT_OUTPUT_ROOT="${REPORT_OUTPUT_ROOT:-${REPO_ROOT}/outputs/report_generation}"

llavarad_merged="${1:-${REPORT_OUTPUT_ROOT}/llavarad_merged}"
custom_vision_encoder="${2:-smartx_large}"
custom_vision_pretrained_path="${3:-}"
stage1_output_dir="${4:-${REPORT_OUTPUT_ROOT}/iu/llavarad_${custom_vision_encoder}_stage1}"
output_dir="${5:-${REPORT_OUTPUT_ROOT}/iu/llavarad_${custom_vision_encoder}_stage2}"
train_batch_size="${6:-24}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-2}"

data_path=${SMARTX_DATA_ROOT}/IUXray/labels/indiana_meta_frontal_findings_w_chexbert_labels_train.csv
image_folder=${SMARTX_DATA_ROOT}/IUXray/images/images_normalized

[[ -d "${llavarad_merged}" ]] || { echo "Missing merged LLaVA-Rad model: ${llavarad_merged}" >&2; exit 1; }
[[ -f "${data_path}" ]] || { echo "Missing training manifest: ${data_path}" >&2; exit 1; }
[[ -d "${image_folder}" ]] || { echo "Missing image directory: ${image_folder}" >&2; exit 1; }
if [[ -n "${custom_vision_pretrained_path}" && ! -f "${custom_vision_pretrained_path}" ]]; then
    echo "Missing vision checkpoint: ${custom_vision_pretrained_path}" >&2
    exit 1
fi

IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=${#GPU_IDS[@]}
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-65000 -n 1)}"

# Ensure local llava package is imported instead of any site-packages llava.
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${output_dir}"

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
    --eval_strategy no
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
    --report_to "${REPORT_TO:-none}"
    --run_name "llavarad_${custom_vision_encoder}_stage2_iu"
)

if [[ -n "${custom_vision_pretrained_path}" ]]; then
    ARGS+=(--custom_vision_pretrained_path "${custom_vision_pretrained_path}")
fi
ARGS+=(--save_total_limit 2)

if [[ -n "${stage1_output_dir}" && "${stage1_output_dir,,}" != "none" && "${stage1_output_dir,,}" != "null" ]]; then
    ARGS+=(--stage1_output_dir "${stage1_output_dir}")
fi

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --module llava.train.train_llavarad_custom "${ARGS[@]}"
