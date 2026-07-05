#!/bin/bash

python ${SMARTX_REPO_ROOT}/downstream/mil_linear_probe/statistical_analysis/analyze_mil_lp_delong.py \
        --base_dir ${SMARTX_OUTPUT_ROOT}/experiments/MIL_LP \
        --output_dir ${SMARTX_REPO_ROOT}/downstream/mil_linear_probe/analysis_results/v97 \
        --reference_model topk_v97_last \
        --mil_type abmil_simple \
        --n_workers 8
