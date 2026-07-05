#!/bin/bash
# 按顺序执行 NYFS 数据集的所有特征提取（同一张卡顺序跑，由本脚本统一指定 GPU）
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "========== NYFS 特征提取 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES，单卡顺序) =========="

run_one() {
  python extract_features.py \
    --config "$1" \
    --data_config configs/dataset_dir.yaml \
    --force
}

run_one configs/extract_nyfs_ark_base.yaml
run_one configs/extract_nyfs_ark_large.yaml
run_one configs/extract_nyfs_chexzero.yaml
run_one configs/extract_nyfs_chess.yaml
run_one configs/extract_nyfs_gloria.yaml
run_one configs/extract_nyfs_maco.yaml
run_one configs/extract_nyfs_our_ark_base.yaml
run_one configs/extract_nyfs_our_ark_large.yaml
run_one configs/extract_nyfs_our_rad_dino.yaml
run_one configs/extract_nyfs_our_siglip2.yaml
run_one configs/extract_nyfs_rad_dino.yaml
run_one configs/extract_nyfs_siglip2.yaml

echo "========== NYFS 全部完成 =========="
