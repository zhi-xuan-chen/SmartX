#!/bin/bash
# Evaluate all Stage 2 checkpoints and select the best by F1-RadGraph.
# Saves per-checkpoint predictions and a comparison summary.
#
# Usage: bash scripts/eval_all_checkpoints.sh <dataset> <llavarad_merged> <checkpoint_root_dir> <run_name> <prediction_base> <eval_batch_size>
#   dataset: "iu", "mimic" or "chexpert"

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

dataset="${1}"
llavarad_merged="${2}"
checkpoint_root_dir="${3}"
run_name="${4}"
prediction_base="${5:-${PROJECT_ROOT}/results/${run_name}}"
eval_batch_size="${6:-2}"

if [ -z "${dataset}" ] || [ -z "${llavarad_merged}" ] || [ -z "${checkpoint_root_dir}" ] || [ -z "${run_name}" ]; then
    echo "Usage: $0 <iu|mimic|chexpert> <llavarad_merged> <checkpoint_root_dir> <run_name> [prediction_base] [eval_batch_size]"
    exit 1
fi

case "${dataset}" in
    iu)       eval_script="${SCRIPT_DIR}/eval_custom_iu.sh" ;;
    mimic)    eval_script="${SCRIPT_DIR}/eval_custom_mimic.sh" ;;
    chexpert) eval_script="${SCRIPT_DIR}/eval_custom_chexpert.sh" ;;
    *)
        echo "Unknown dataset: ${dataset} (expected iu, mimic or chexpert)"
        exit 1
        ;;
esac

mkdir -p "${prediction_base}"

# Collect all checkpoint subdirectories
shopt -s nullglob
ckpt_dirs=($(ls -d "${checkpoint_root_dir}"/checkpoint-* 2>/dev/null | sort -t'-' -k2 -n))
shopt -u nullglob

if [ ${#ckpt_dirs[@]} -eq 0 ]; then
    echo "No checkpoint-* found in ${checkpoint_root_dir}, falling back to root directory itself"
    ckpt_dirs=("${checkpoint_root_dir}")
fi

echo "Found ${#ckpt_dirs[@]} checkpoint(s) to evaluate"
echo ""

summary_csv="${prediction_base}/checkpoint_comparison.csv"
echo "checkpoint,BLEU-4,ROUGE-L,Micro-F1-14+,Macro-F1-14+,F1-RadGraph" > "${summary_csv}"

best_f1=0
best_ckpt=""

for ckpt_dir in "${ckpt_dirs[@]}"; do
    ckpt_name=$(basename "${ckpt_dir}")
    echo "========== [${ckpt_name}] Evaluating =========="
    ckpt_pred_dir="${prediction_base}/${ckpt_name}"

    bash "${eval_script}" \
        "${llavarad_merged}" "${ckpt_dir}" \
        "${run_name}_${ckpt_name}" "${ckpt_pred_dir}" "${eval_batch_size}"

    main_csv="${ckpt_pred_dir}/eval/main.csv"
    if [ -f "${main_csv}" ]; then
        median_line=$(sed -n '2p' "${main_csv}")
        bleu4=$(echo "${median_line}" | cut -d',' -f3)
        rougel=$(echo "${median_line}" | cut -d',' -f4)
        mif1=$(echo "${median_line}" | cut -d',' -f9)
        maf1=$(echo "${median_line}" | cut -d',' -f10)
        f1_rg=$(echo "${median_line}" | cut -d',' -f13)

        echo "${ckpt_name},${bleu4},${rougel},${mif1},${maf1},${f1_rg}" >> "${summary_csv}"
        echo "  BLEU-4: ${bleu4}  ROUGE-L: ${rougel}  MiF1-14+: ${mif1}  MaF1-14+: ${maf1}  F1-RadGraph: ${f1_rg}"

        if (( $(echo "${f1_rg} > ${best_f1}" | bc -l) )); then
            best_f1="${f1_rg}"
            best_ckpt="${ckpt_name}"
        fi
    else
        echo "  WARNING: eval failed, no main.csv found"
    fi
    echo ""
done

echo "=============================================="
echo "  Checkpoint Comparison"
echo "=============================================="
column -t -s',' "${summary_csv}"
echo ""
echo "Best: ${best_ckpt}  (F1-RadGraph: ${best_f1})"
echo "All predictions: ${prediction_base}"
echo "Summary: ${summary_csv}"
