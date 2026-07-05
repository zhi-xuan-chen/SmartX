#!/bin/bash
# Extract features for NCFY dataset using SigLIP2 vision encoder

export CUDA_VISIBLE_DEVICES=1

cd ${SMARTX_REPO_ROOT}/downstream/mil_linear_probe

python extract_features.py \
    --config configs/extract_ncfy_siglip2.yaml \
    --data_config configs/dataset_dir.yaml \
    --force

