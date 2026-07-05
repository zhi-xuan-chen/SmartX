#!/bin/bash
# 一键抽取全部 4 个外部测试集(NCFY / NYFS / ZSFS / ZSFY)的特征 —— our_rad_dino 模型。
# 对于 NCFY 和 ZSFS:既有 prospective 又有 retrospective CSV,两份都跑(同一个特征目录,
# 互不冲突,因为路径含 patient/case 维度;同时使用 --resume 避免重复抽已有的图片特征)。
#
# 用法:
#   bash scripts/extract_external_our_rad_dino.sh           # 默认 GPU 0
#   CUDA_VISIBLE_DEVICES=3 bash scripts/extract_external_our_rad_dino.sh
#
# 输出:
#   ${SMARTX_OUTPUT_ROOT}/features/{ncfy|nyfs|zsfs|zsfy}/
#                       <region>/<model_id>/<相对路径>.npy
#
# 注意:此脚本不影响已有的 extract_<ds>_our_rad_dino.sh,只是新增一种"一键全跑"入口。
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

EXTRACT="python extract_features.py"
DATA_CFG="--data_config configs/dataset_dir.yaml"
# 首份 CSV 用 --force 确保是干净抽特征;后续同数据集的第二份 CSV 用 --resume,
# 已经抽过的图直接跳过(NCFY/ZSFS 的 pros/retro 病例多数不重叠,但稳妥起见)。

echo "######## NCFY (prospective, 467 cases) ########"
$EXTRACT --config configs/extract_ncfy_our_rad_dino.yaml $DATA_CFG \
    --csv_name ncfy_prospective_en_with_labels.csv --force

echo "######## NCFY (retrospective, 488 cases) ########"
$EXTRACT --config configs/extract_ncfy_our_rad_dino.yaml $DATA_CFG \
    --csv_name ncfy_retrospective_en_with_labels.csv --resume

echo "######## NYFS (prospective_updated, 4802 cases) ########"
$EXTRACT --config configs/extract_nyfs_our_rad_dino.yaml $DATA_CFG --force

echo "######## ZSFS (retrospective_updated, 5106 cases) ########"
$EXTRACT --config configs/extract_zsfs_our_rad_dino.yaml $DATA_CFG \
    --csv_name zsfs_retrospective_with_labels_en_dedup_updated.csv --force

echo "######## ZSFS (prospective_updated, 1156 cases) ########"
$EXTRACT --config configs/extract_zsfs_our_rad_dino.yaml $DATA_CFG \
    --csv_name zsfs_prospective_with_labels_en_dedup_updated.csv --resume

echo "######## ZSFY (qwen3-30b_single_region_updated, 2839 cases) ########"
$EXTRACT --config configs/extract_zsfy_our_rad_dino.yaml $DATA_CFG --force

echo
echo "✓ 全部 4 个外部数据集特征提取完成 (model: our_rad_dino, GPU=$CUDA_VISIBLE_DEVICES)"
