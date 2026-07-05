#!/bin/bash
# Extract features for ZSFY dataset using our Ark+ large model

export CUDA_VISIBLE_DEVICES=3

cd ${SMARTX_REPO_ROOT}/downstream/mil_linear_probe

python extract_features.py \
    --config configs/extract_zsfy_our_ark_large_lp.yaml \
    --data_config configs/dataset_dir.yaml \
    --force
