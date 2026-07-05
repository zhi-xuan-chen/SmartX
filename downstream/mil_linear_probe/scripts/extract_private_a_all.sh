#!/bin/bash
# 按顺序执行 Private A 数据集的所有特征提取脚本（同一 GPU 顺序跑）
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "========== Private A 特征提取 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) =========="

bash scripts/extract_private_a_ark_base.sh
bash scripts/extract_private_a_ark_large.sh
bash scripts/extract_private_a_chexzero.sh
bash scripts/extract_private_a_chess.sh
bash scripts/extract_private_a_gloria.sh
bash scripts/extract_private_a_biomedclip.sh
bash scripts/extract_private_a_maco.sh
bash scripts/extract_private_a_our_ark_base.sh
bash scripts/extract_private_a_our_ark_large.sh
bash scripts/extract_private_a_our_rad_dino.sh
bash scripts/extract_private_a_our_siglip2.sh
bash scripts/extract_private_a_rad_dino.sh
bash scripts/extract_private_a_siglip2.sh

echo "========== Private A 全部完成 =========="
