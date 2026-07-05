#!/bin/bash
# Extract features for NYFS dataset using SigLIP2 vision encoder

export CUDA_VISIBLE_DEVICES=2

cd ${SMARTX_REPO_ROOT}/downstream/mil_linear_probe

python extract_features.py \
    --config configs/extract_nyfs_siglip2.yaml \
    --data_config configs/dataset_dir.yaml \
    --force

