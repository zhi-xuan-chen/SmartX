#!/bin/bash
# Stage 1: train mm_projector only (LLM + vision encoder frozen)
# Usage: bash scripts/train_stage1_custom_mimic.sh [llavarad_merged] [encoder_name] [pretrained_path] [output_dir] [train_batch_size]
# Supported custom vision encoder names:
#   ark_base | ark_large | our_ark_base | our_ark_large | rad_dino | gloria | chexzero | chess | biomedclip | conceptclip | maco
# Notes:
#   - Pass pretrained_path for encoders whose weights are stored locally.
#   - Supported loaders: default | mimic_train_findings | mimic_test_findings | mimic_cxr_all_frontal_findings | mimic_cxr_all_views_findings

set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

: "${SMARTX_DATA_ROOT:?Set SMARTX_DATA_ROOT to the report-generation data root}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
SMARTX_OUTPUT_ROOT="${SMARTX_OUTPUT_ROOT:-${REPO_ROOT}/outputs/report_generation}"

llavarad_merged="${1:-${SMARTX_OUTPUT_ROOT}/llavarad_merged}"
custom_vision_encoder="${2:-smartx_large}"
custom_vision_pretrained_path="${3:-}"
output_dir="${4:-${SMARTX_OUTPUT_ROOT}/mimic/llavarad_${custom_vision_encoder}_stage1}"
train_batch_size="${5:-48}"

data_path=${SMARTX_DATA_ROOT}/MIMIC-CXR/labels/mimic_cxr_train_llava_rad_frontal_rule_and_gpt4_use_reason_301681.csv
image_folder=${SMARTX_DATA_ROOT}/MIMIC-CXR/imgs-1024

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
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${output_dir}"

ARGS=(
    --model_name_or_path "${llavarad_merged}"
    --version v1
    --custom_vision_encoder "${custom_vision_encoder}"
    --mm_projector_type mlp2x_gelu
    --data_path "${data_path}"
    --loader mimic_train_findings
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
    --report_to "${REPORT_TO:-none}"
    --run_name "llavarad_${custom_vision_encoder}_stage1"
)

if [[ -n "${custom_vision_pretrained_path}" ]]; then
    ARGS+=(--custom_vision_pretrained_path "${custom_vision_pretrained_path}")
fi

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --module llava.train.train_llavarad_custom "${ARGS[@]}"
