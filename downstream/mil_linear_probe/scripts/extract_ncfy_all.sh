#!/bin/bash
# 按顺序执行 NCFY 数据集的所有特征提取脚本（同一 GPU 顺序跑）
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "========== NCFY 特征提取 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) =========="

bash scripts/extract_ncfy_ark_base.sh
bash scripts/extract_ncfy_ark_large.sh
bash scripts/extract_ncfy_chexzero.sh
bash scripts/extract_ncfy_chess.sh
bash scripts/extract_ncfy_gloria.sh
bash scripts/extract_ncfy_biomedclip.sh
bash scripts/extract_ncfy_maco.sh
bash scripts/extract_ncfy_our_ark_base.sh
bash scripts/extract_ncfy_our_ark_large.sh
bash scripts/extract_ncfy_our_rad_dino.sh
bash scripts/extract_ncfy_our_siglip2.sh
bash scripts/extract_ncfy_rad_dino.sh
bash scripts/extract_ncfy_siglip2.sh

echo "========== NCFY 全部完成 =========="
