#!/bin/bash
# Extract features for NCFY dataset using our Ark+ base model

export CUDA_VISIBLE_DEVICES=0

cd ${SMARTX_REPO_ROOT}/downstream/mil_linear_probe

python extract_features.py \
    --config configs/extract_ncfy_our_ark_base_lp.yaml \
    --data_config configs/dataset_dir.yaml \
    --force

