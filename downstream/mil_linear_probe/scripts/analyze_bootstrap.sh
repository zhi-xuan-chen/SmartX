#!/bin/bash

python ${SMARTX_REPO_ROOT}/downstream/mil_linear_probe/statistical_analysis/bootstrap.py \
        --base_dir ${SMARTX_OUTPUT_ROOT}/experiments/MIL_LP \
        --output_dir ${SMARTX_REPO_ROOT}/downstream/mil_linear_probe/analysis_results/v97 \
        --n_bootstrap 1000 \
        --mil_type abmil_simple \
        --n_workers 8