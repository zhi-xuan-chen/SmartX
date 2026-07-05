#!/bin/bash
# Extract features for private_a dataset using RAD-DINO model

export CUDA_VISIBLE_DEVICES=1

cd ${SMARTX_REPO_ROOT}/downstream/mil_linear_probe

python extract_features.py \
    --config configs/extract_private_a_rad_dino.yaml \
    --data_config configs/dataset_dir.yaml \
    --force

