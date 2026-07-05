#!/bin/bash
# Extract features for ZSFY dataset using CheXzero model

export CUDA_VISIBLE_DEVICES=2

cd ${SMARTX_REPO_ROOT}/downstream/mil_linear_probe

python extract_features.py \
    --config configs/extract_zsfy_chexzero.yaml \
    --data_config configs/dataset_dir.yaml \
    --force
