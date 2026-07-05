#!/bin/bash
# 一次跑完所有方法 × 所有数据集的统计显著性检验：
#   0. _build_leaderboard.py        重新生成 results/leaderboard_best_ckpt.csv
#   1. compute_clinical_labels.py   per dataset (CheXBERT + RadGraph, 需 GPU)
#   2. analyze_significance.py      per dataset，输出两张表（均追加，含 dataset 列）：
#                                     - pairwise_significance_all.csv  (p 值)
#                                     - metrics_ci_all.csv  (每模型点估计 + 95% CI，
#                                       从 jsonl + 缓存标签重算，绝不读 main.csv)
#   3. _enrich_with_ci.py           把 metrics_ci_all.csv 的点估计/CI join 进 pairwise
#
# 说明：各文件夹里的 eval/main.csv 完全不受影响，仅作参考；论文用的最终指标在
#       analysis_results/<REF_TAG>/{metrics_ci_all.csv, pairwise_significance_all.csv}。
#
# 用法：
#   bash scripts/analyze_all.sh
#
# 可通过环境变量覆盖：
#   CUDA_VISIBLE_DEVICES - Step 1 用哪张卡 (默认 "0")
#   REFERENCE_MODEL    - leaderboard `method` 列里的展示名 (默认 "topk_v19 (AR)")
#   TOPK_COMPARE_VERSIONS - 显著性检验保留的 topk 版本 (默认 "v19 v14 v21 v23")；
#                           其他 topk_* 不参与 pairwise；非 topk baseline 仍保留
#   DATASETS           - 空格分隔 (默认 "mimic iu")
#   METRICS            - analyze_significance.py 的 --metrics
#   SIGNIFICANCE_METHOD - paired_permutation / bootstrap_wilcoxon / bootstrap_pvalue
#                         (默认 bootstrap_wilcoxon)
#                         bootstrap_pvalue: 双侧 bootstrap ASL p=2*min(P(diff<=0),P(diff>=0))
#   N_PERMUTATION      - 默认 10000
#   N_BOOTSTRAP        - 默认 1000
#   BUILD_LEADERBOARD  - 1=重生成 leaderboard (默认 1)
#   FORCE_CLINICAL     - 1=强制重算 chexbert/radgraph (默认 0，文件已存在则跳过)
#   SKIP_CHEXBERT      - 1=强制跳过 CheXBERT (默认 0)
#   SKIP_RADGRAPH      - 1=强制跳过 RadGraph (默认 0)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Step 1 (compute_clinical_labels.py) 需要 GPU；改这里换卡。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

REFERENCE_MODEL="${REFERENCE_MODEL:-topk_v19 (AR)}"
TOPK_COMPARE_VERSIONS="${TOPK_COMPARE_VERSIONS:-v19 v21}"
DATASETS_STR="${DATASETS:-mimic iu}"
read -r -a DATASETS <<< "${DATASETS_STR}"
METRICS="${METRICS:-BLEU-4,BLEU-1,ROUGE-L,ROUGE-2,Micro-F1-14+,Macro-F1-14+,Micro-F1-5+,Macro-F1-5+,F1-RadGraph}"
SIGNIFICANCE_METHOD="${SIGNIFICANCE_METHOD:-bootstrap_wilcoxon}"
N_PERMUTATION="${N_PERMUTATION:-10000}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"
BUILD_LEADERBOARD="${BUILD_LEADERBOARD:-1}"
FORCE_CLINICAL="${FORCE_CLINICAL:-0}"
SKIP_CHEXBERT="${SKIP_CHEXBERT:-0}"
SKIP_RADGRAPH="${SKIP_RADGRAPH:-0}"

LEADERBOARD="${PROJECT_DIR}/results/leaderboard_best_ckpt.csv"
RESULTS_DIR="${PROJECT_DIR}/results"

REF_TAG="$(echo "${REFERENCE_MODEL}" | tr -c '[:alnum:]' '_' | tr -s '_' | sed 's/^_//;s/_$//')"
OUTPUT_ROOT="${PROJECT_DIR}/analysis_results/${REF_TAG}"
COMBINED_CSV="${OUTPUT_ROOT}/pairwise_significance_all.csv"
METRICS_CI_CSV="${OUTPUT_ROOT}/metrics_ci_all.csv"

mkdir -p "${OUTPUT_ROOT}"

run_step() {
    local label="$1"; shift
    echo "=================================================================="
    echo "[$(date '+%H:%M:%S')] >>> ${label}"
    echo "    command: $*"
    echo "=================================================================="
    "$@"
    echo "[$(date '+%H:%M:%S')] <<< ${label} done"
    echo
}

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "REFERENCE_MODEL=${REFERENCE_MODEL}"
echo "TOPK_COMPARE_VERSIONS=${TOPK_COMPARE_VERSIONS}"
echo "DATASETS=${DATASETS[*]}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "COMBINED_CSV=${COMBINED_CSV}"
echo "METRICS_CI_CSV=${METRICS_CI_CSV}"
echo "SIGNIFICANCE_METHOD=${SIGNIFICANCE_METHOD}"
echo "FORCE_CLINICAL=${FORCE_CLINICAL}  SKIP_CHEXBERT=${SKIP_CHEXBERT}  SKIP_RADGRAPH=${SKIP_RADGRAPH}"
echo

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---- Step 0: regenerate leaderboard ----
if [[ "${BUILD_LEADERBOARD}" == "1" ]]; then
    run_step "build leaderboard" \
        python "${SCRIPT_DIR}/_build_leaderboard.py" \
            --results_dir "${RESULTS_DIR}" \
            --out "${LEADERBOARD}"
else
    echo "BUILD_LEADERBOARD=0 -> using existing ${LEADERBOARD}"
fi

# 每次全量重跑时清空合并结果，避免旧 dataset 残留
rm -f "${COMBINED_CSV}" "${METRICS_CI_CSV}"

for dataset in "${DATASETS[@]}"; do
    # ---- Build --jsonls arg + reference model key ----
    mapfile -t LINES < <(python "${SCRIPT_DIR}/_build_jsonls_arg.py" \
        --leaderboard "${LEADERBOARD}" \
        --results_dir "${RESULTS_DIR}" \
        --dataset "${dataset}" \
        --reference_model "${REFERENCE_MODEL}" \
        --topk_versions "${TOPK_COMPARE_VERSIONS}")
    JSONLS_ARG="${LINES[0]}"
    REF_KEY="${LINES[1]}"

    if [[ -z "${JSONLS_ARG}" || -z "${REF_KEY}" ]]; then
        echo "[${dataset}] FATAL: failed to build jsonls / ref_key" >&2
        exit 1
    fi
    echo "[${dataset}] reference key = ${REF_KEY}"
    echo "[${dataset}] candidates    = $(echo "${JSONLS_ARG}" | tr ',' '\n' | wc -l) models"

    # ---- Step 1: clinical labels (skip if file exists, unless FORCE_CLINICAL=1) ----
    CHEXBERT_PATH="${OUTPUT_ROOT}/chexbert_labels_${dataset}.json"
    RADGRAPH_PATH="${OUTPUT_ROOT}/radgraph_rewards_${dataset}.json"

    do_chexbert=1
    do_radgraph=1
    [[ "${SKIP_CHEXBERT}" == "1" ]] && do_chexbert=0
    [[ "${SKIP_RADGRAPH}" == "1" ]] && do_radgraph=0
    if [[ "${FORCE_CLINICAL}" != "1" ]]; then
        [[ -f "${CHEXBERT_PATH}" ]] && do_chexbert=0 && echo "[${dataset}] CheXBERT exists -> skip"
        [[ -f "${RADGRAPH_PATH}" ]] && do_radgraph=0 && echo "[${dataset}] RadGraph exists -> skip"
    fi

    if [[ "${do_chexbert}" == "1" || "${do_radgraph}" == "1" ]]; then
        CLINICAL_FLAGS=()
        [[ "${do_chexbert}" == "0" ]] && CLINICAL_FLAGS+=(--skip_chexbert)
        [[ "${do_radgraph}" == "0" ]] && CLINICAL_FLAGS+=(--skip_radgraph)
        run_step "[${dataset}] clinical labels" \
            python "${PROJECT_DIR}/compute_clinical_labels.py" \
                --jsonls "${JSONLS_ARG}" \
                --output_dir "${OUTPUT_ROOT}" \
                --dataset "${dataset}" \
                "${CLINICAL_FLAGS[@]}"
    else
        echo "[${dataset}] both CheXBERT and RadGraph available; skipping step 1"
        echo
    fi

    # ---- Step 2: paired significance tests (append to combined CSV) ----
    SIG_FLAGS=()
    [[ -f "${CHEXBERT_PATH}" ]] && SIG_FLAGS+=(--chexbert_labels "${CHEXBERT_PATH}")
    [[ -f "${RADGRAPH_PATH}" ]] && SIG_FLAGS+=(--radgraph_rewards "${RADGRAPH_PATH}")

    # Step 2 also recomputes the per-model point estimate + 95% bootstrap CI
    # (--metrics_ci_csv) from the same jsonls + cached labels — never main.csv.
    run_step "[${dataset}] paired significance + bootstrap CI" \
        python "${PROJECT_DIR}/analyze_significance.py" \
            --jsonls "${JSONLS_ARG}" \
            --reference_model "${REF_KEY}" \
            --metrics "${METRICS}" \
            --significance_method "${SIGNIFICANCE_METHOD}" \
            --n_permutation "${N_PERMUTATION}" \
            --n_bootstrap "${N_BOOTSTRAP}" \
            --output_dir "${OUTPUT_ROOT}" \
            --dataset "${dataset}" \
            --combined_csv "${COMBINED_CSV}" \
            --metrics_ci_csv "${METRICS_CI_CSV}" \
            "${SIG_FLAGS[@]}"
done

# ---- Step 3: join per-model point estimate + 95% CI onto the pairwise table ----
run_step "enrich pairwise with CI" \
    python "${SCRIPT_DIR}/_enrich_with_ci.py" \
        --output_root "${OUTPUT_ROOT}"

echo "=================================================================="
echo "[$(date '+%H:%M:%S')] All analyses completed."
echo "Pairwise (p-value + CI): ${COMBINED_CSV}"
echo "All models metrics + CI: ${METRICS_CI_CSV}"
echo "=================================================================="
