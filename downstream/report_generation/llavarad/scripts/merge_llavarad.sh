#!/bin/bash
# Merge Vicuna base + LLaVA-Rad LoRA into a single checkpoint.
# Usage: bash scripts/merge_llavarad.sh [model_base] [model_path] [save_path]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
SMARTX_OUTPUT_ROOT="${SMARTX_OUTPUT_ROOT:-${REPO_ROOT}/outputs/report_generation}"

model_base="${1:-lmsys/vicuna-7b-v1.5}"
model_path="${2:-microsoft/llava-rad}"
save_model_path="${3:-${SMARTX_OUTPUT_ROOT}/llavarad_merged}"

echo "Merging LLaVA-Rad weights..."
echo "  base  : ${model_base}"
echo "  lora  : ${model_path}"
echo "  output: ${save_model_path}"

mkdir -p "${save_model_path}"

python "${SCRIPT_DIR}/merge_lora_weights.py" \
    --model-base "${model_base}" \
    --model-path "${model_path}" \
    --save-model-path "${save_model_path}"

echo "Done. Merged model saved to ${save_model_path}"
