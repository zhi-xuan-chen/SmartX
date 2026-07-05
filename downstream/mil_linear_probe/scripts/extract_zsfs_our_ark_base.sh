#!/bin/bash
# Extract features for ZSFS dataset using our Ark+ base model

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}

cd ${SMARTX_REPO_ROOT}/downstream/mil_linear_probe

python extract_features.py \
    --config configs/extract_zsfs_our_ark_base.yaml \
    --data_config configs/dataset_dir.yaml \
    --force

