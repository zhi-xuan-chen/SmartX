#!/bin/bash
# 按顺序执行 ZSFS 数据集的所有特征提取脚本（同一 GPU 顺序跑）
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "========== ZSFS 特征提取 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) =========="

bash scripts/extract_zsfs_ark_base.sh
bash scripts/extract_zsfs_ark_large.sh
bash scripts/extract_zsfs_chexzero.sh
bash scripts/extract_zsfs_chess.sh
bash scripts/extract_zsfs_gloria.sh
bash scripts/extract_zsfs_biomedclip.sh
bash scripts/extract_zsfs_maco.sh
bash scripts/extract_zsfs_our_ark_base.sh
bash scripts/extract_zsfs_our_ark_large.sh
bash scripts/extract_zsfs_our_siglip2.sh
bash scripts/extract_zsfs_rad_dino.sh
bash scripts/extract_zsfs_siglip2.sh

echo "========== ZSFS 全部完成 =========="
