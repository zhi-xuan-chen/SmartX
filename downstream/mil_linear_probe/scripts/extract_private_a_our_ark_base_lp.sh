#!/bin/bash
# Extract features for private_a dataset using our Ark+ base model

export CUDA_VISIBLE_DEVICES=2

cd ${SMARTX_REPO_ROOT}/downstream/mil_linear_probe

FEATURE_ROOT=${SMARTX_OUTPUT_ROOT}/features/private_a
mkdir -p "$FEATURE_ROOT"

python extract_features.py \
    --config configs/extract_private_a_our_ark_base_lp.yaml \
    --data_config configs/dataset_dir.yaml \
    --feature_root "$FEATURE_ROOT" \
    --force

