#!/bin/bash
# Extract features for NCFY dataset using Chess model

export CUDA_VISIBLE_DEVICES=2

cd ${SMARTX_REPO_ROOT}/downstream/mil_linear_probe

python extract_features.py \
    --config configs/extract_ncfy_chess.yaml \
    --data_config configs/dataset_dir.yaml \
    --force

