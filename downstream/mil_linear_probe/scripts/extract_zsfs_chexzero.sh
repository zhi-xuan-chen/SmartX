#!/bin/bash
# Extract features for ZSFS dataset using CheXzero model

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

cd ${SMARTX_REPO_ROOT}/downstream/mil_linear_probe

python extract_features.py \
    --config configs/extract_zsfs_chexzero.yaml \
    --data_config configs/dataset_dir.yaml \
    --force

