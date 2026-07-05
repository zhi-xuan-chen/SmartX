#!/bin/bash
# 一键抽取 [全部 14 个方法] × [4 个外部数据集 NCFY/NYFS/ZSFS/ZSFY] 的特征,
# 同一 GPU 上顺序跑(method 内部已经把 4 个数据集都跑完)。
#
# 用法:
#   bash scripts/extract_external_all.sh                     # 默认 GPU 0
#   CUDA_VISIBLE_DEVICES=3 bash scripts/extract_external_all.sh
#
# 等价于按顺序执行:
#   bash scripts/extract_external_<method>.sh    # 共 14 个 method
#
# 跑完所有特征会落到:
#   ${SMARTX_OUTPUT_ROOT}/features/{ncfy|nyfs|zsfs|zsfy}/
#                       <region>/<model_id>/<相对路径>.npy
#
# 注意:此脚本是"外部数据 × 所有方法"的对偶版本,跟现有 extract_<ds>_all.sh
#       (单数据集 × 所有方法)对应,不影响任何现有脚本。
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "========== External 4 Datasets × All Methods (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) =========="

# Baselines
# bash scripts/extract_external_ark_base.sh
bash scripts/extract_external_ark_large.sh
bash scripts/extract_external_chess.sh
bash scripts/extract_external_chexzero.sh
bash scripts/extract_external_gloria.sh
bash scripts/extract_external_biomedclip.sh
bash scripts/extract_external_maco.sh
bash scripts/extract_external_rad_dino.sh
# bash scripts/extract_external_siglip2.sh

# # Our models (default preprocess)
# bash scripts/extract_external_our_ark_base.sh
# bash scripts/extract_external_our_ark_large.sh
# bash scripts/extract_external_our_rad_dino.sh
# bash scripts/extract_external_our_siglip2.sh

# # Our models (pad-to-square / "lp" preprocess)
# bash scripts/extract_external_our_ark_base_lp.sh
# bash scripts/extract_external_our_ark_large_lp.sh

echo "========== External 4 Datasets × All Methods 全部完成 =========="
