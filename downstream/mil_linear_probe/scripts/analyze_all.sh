#!/bin/bash
# 一次跑完 6 个统计分析：
#   1. bootstrap.py               (in-domain bootstrap CI -> results_mean.csv, 含 AUC/F1/AP/MCC/b-acc)
#   2. bootstrap_external.py      (external bootstrap CI  -> external/results_mean.csv, 同上)
#   3. analyze_mil_lp_delong.py   (in-domain AUC: DeLong + paired significance -> delong_results.csv)
#   4. analyze_delong_external.py (external AUC: DeLong + paired significance -> external/delong_results.csv)
#   5. analyze_threshold_metrics.py          (in-domain F1/MCC/b-acc: paired significance -> threshold_results.csv)
#   6. analyze_threshold_metrics_external.py (external F1/MCC/b-acc: paired significance -> external/threshold_results.csv)
#
# 分工：AUC per-class 走 DeLong（闭式方差）；mean AUC 和阈值类指标 F1/MCC/b-acc
#      可选 paired_permutation / bootstrap_wilcoxon / bootstrap_pvalue，默认 bootstrap_wilcoxon。
#      bootstrap_pvalue: 双侧 bootstrap ASL p=2*min(P(diff<=0),P(diff>=0))。
# 两类都按 (dataset[,metric], is_mean) 分族做 BH-FDR；raw p_value 与 p_value_fdr 各占一列。
#
# 用法：
#   bash scripts/analyze_all.sh
#
# 注意：6 个脚本会顺序执行（任意一个失败立即终止，便于排查）。
# 如果想并行，可以分别在多个终端各跑一个 —— 它们读写互不冲突。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 改这里即可切换主比较模型和 MIL 聚合类型。
REFERENCE_MODEL="topk_v19_last"
MIL_TYPE="clam"

# 模型过滤（二选一或组合使用；都留空则分析目录下全部模型）
# MODELS: 白名单，仅分析/比较列表中的模型（需包含 REFERENCE_MODEL）
# EXCLUDE_MODELS: 黑名单，跳过旧模型
# 例: EXCLUDE_MODELS="topk_v14_last,topk_v83_last,topk_v95_best,topk_v95_last,topk_v97_best,topk_v97_last,topk_v100_best"
MODELS=""
EXCLUDE_MODELS="topk_v83_last,topk_v95_best,topk_v95_last,topk_v97_best,topk_v97_last,topk_v100_best"

MODEL_FILTER_ARGS=()
if [[ -n "${MODELS}" ]]; then
    MODEL_FILTER_ARGS+=(--models "${MODELS}")
fi
if [[ -n "${EXCLUDE_MODELS}" ]]; then
    MODEL_FILTER_ARGS+=(--exclude_models "${EXCLUDE_MODELS}")
fi

BASE_DIR="${SMARTX_OUTPUT_ROOT}/experiments/MIL_LP"
RESULTS_DIR="${PROJECT_DIR}/results/${MIL_TYPE}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"
N_PERMUTATION="${N_PERMUTATION:-10000}"
SIGNIFICANCE_METHOD="${SIGNIFICANCE_METHOD:-bootstrap_wilcoxon}"
N_WORKERS="${N_WORKERS:-8}"
ALPHA="${ALPHA:-0.05}"

if [[ "${REFERENCE_MODEL}" =~ v([0-9]+) ]]; then
    RESULT_VERSION="v${BASH_REMATCH[1]}"
else
    RESULT_VERSION="${REFERENCE_MODEL//[^A-Za-z0-9_]/_}"
fi

if [[ "${REFERENCE_MODEL}" =~ (^|_)lp($|_) ]]; then
    RESULT_VERSION="${RESULT_VERSION}_lp"
fi

if [[ -n "${MIL_TYPE}" ]]; then
    RESULT_VERSION="${RESULT_VERSION}_${MIL_TYPE}"
fi

OUTPUT_DIR="${PROJECT_DIR}/analysis_results/${RESULT_VERSION}"
EXTERNAL_OUTPUT_DIR="${OUTPUT_DIR}/external"

run_step() {
    local label="$1"
    shift
    echo "=================================================================="
    echo "[$(date '+%H:%M:%S')] >>> ${label}"
    echo "    command: $*"
    echo "=================================================================="
    "$@"
    echo "[$(date '+%H:%M:%S')] <<< ${label} done"
    echo
}

echo "REFERENCE_MODEL=${REFERENCE_MODEL}"
echo "MIL_TYPE=${MIL_TYPE}"
echo "MODELS=${MODELS:-<all>}"
echo "EXCLUDE_MODELS=${EXCLUDE_MODELS:-<none>}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "EXTERNAL_OUTPUT_DIR=${EXTERNAL_OUTPUT_DIR}"
echo "N_BOOTSTRAP=${N_BOOTSTRAP}"
echo "N_PERMUTATION=${N_PERMUTATION}"
echo "SIGNIFICANCE_METHOD=${SIGNIFICANCE_METHOD}"
echo

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_step "1/4 in-domain bootstrap" \
    python "${PROJECT_DIR}/statistical_analysis/bootstrap.py" \
        --base_dir "${BASE_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --n_bootstrap "${N_BOOTSTRAP}" \
        --mil_type "${MIL_TYPE}" \
        --n_workers "${N_WORKERS}" \
        "${MODEL_FILTER_ARGS[@]}"

run_step "2/4 external bootstrap" \
    python "${PROJECT_DIR}/statistical_analysis/bootstrap_external.py" \
        --results_dir "${RESULTS_DIR}" \
        --output_dir "${EXTERNAL_OUTPUT_DIR}" \
        --n_bootstrap "${N_BOOTSTRAP}" \
        --n_workers "${N_WORKERS}" \
        "${MODEL_FILTER_ARGS[@]}"

run_step "3/6 in-domain DeLong + paired significance" \
    python "${PROJECT_DIR}/statistical_analysis/analyze_mil_lp_delong.py" \
        --base_dir "${BASE_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --reference_model "${REFERENCE_MODEL}" \
        --mil_type "${MIL_TYPE}" \
        --n_permutation "${N_PERMUTATION}" \
        --n_bootstrap "${N_BOOTSTRAP}" \
        --significance_method "${SIGNIFICANCE_METHOD}" \
        --n_workers "${N_WORKERS}" \
        "${MODEL_FILTER_ARGS[@]}"

run_step "4/6 external DeLong + paired significance" \
    python "${PROJECT_DIR}/statistical_analysis/analyze_delong_external.py" \
        --results_dir "${RESULTS_DIR}" \
        --output_dir "${EXTERNAL_OUTPUT_DIR}" \
        --our_model "${REFERENCE_MODEL}" \
        --alpha "${ALPHA}" \
        --n_permutation "${N_PERMUTATION}" \
        --n_bootstrap "${N_BOOTSTRAP}" \
        --significance_method "${SIGNIFICANCE_METHOD}" \
        --n_workers "${N_WORKERS}" \
        "${MODEL_FILTER_ARGS[@]}"

run_step "5/6 in-domain threshold metrics (F1/MCC/b-acc) paired significance" \
    python "${PROJECT_DIR}/statistical_analysis/analyze_threshold_metrics.py" \
        --base_dir "${BASE_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --reference_model "${REFERENCE_MODEL}" \
        --mil_type "${MIL_TYPE}" \
        --alpha "${ALPHA}" \
        --n_permutation "${N_PERMUTATION}" \
        --n_bootstrap "${N_BOOTSTRAP}" \
        --significance_method "${SIGNIFICANCE_METHOD}" \
        --n_workers "${N_WORKERS}" \
        "${MODEL_FILTER_ARGS[@]}"

run_step "6/6 external threshold metrics (F1/MCC/b-acc) paired significance" \
    python "${PROJECT_DIR}/statistical_analysis/analyze_threshold_metrics_external.py" \
        --results_dir "${RESULTS_DIR}" \
        --output_dir "${EXTERNAL_OUTPUT_DIR}" \
        --our_model "${REFERENCE_MODEL}" \
        --alpha "${ALPHA}" \
        --n_permutation "${N_PERMUTATION}" \
        --n_bootstrap "${N_BOOTSTRAP}" \
        --significance_method "${SIGNIFICANCE_METHOD}" \
        --n_workers "${N_WORKERS}" \
        "${MODEL_FILTER_ARGS[@]}"

echo "=================================================================="
echo "[$(date '+%H:%M:%S')] All 6 analyses completed."
echo "=================================================================="
