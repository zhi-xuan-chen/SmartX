#!/bin/bash
# Extract features for ZSFY dataset using Ark base model

export CUDA_VISIBLE_DEVICES=1

cd ${SMARTX_REPO_ROOT}/downstream/mil_linear_probe

python extract_features.py \
    --config configs/extract_zsfy_ark_base.yaml \
    --data_config configs/dataset_dir.yaml \
    --force

