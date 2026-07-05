#!/bin/bash
# Extract features for ZSFS dataset using our Rad-DINO model

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd ${SMARTX_REPO_ROOT}/downstream/mil_linear_probe

python extract_features.py \
    --config configs/extract_zsfs_our_rad_dino.yaml \
    --data_config configs/dataset_dir.yaml \
    --force

