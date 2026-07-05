#!/bin/bash

MIL_TYPE="${MIL_TYPE:-abmil_simple}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python ${SMARTX_REPO_ROOT}/downstream/mil_linear_probe/statistical_analysis/bootstrap_external.py \
  --results_dir "${SMARTX_REPO_ROOT}/downstream/mil_linear_probe/results/${MIL_TYPE}" \
  --output_dir ${SMARTX_REPO_ROOT}/downstream/mil_linear_probe/analysis_results/v97/external \
  --n_bootstrap 1000 \
  --n_workers 8
