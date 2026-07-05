#!/usr/bin/env bash
#
# One-click reproduction of the reader-study analysis layer.
#
# Reproduces (from pre-existing test_results.csv + checkpoints + NAS label CSVs):
#   1) results/reader_study_balanced/   per-dataset balanced eval + plots
#   2) results/reader_study_merge/       multi-center equal-quota merge eval + plots
#   3) results/reader_study_merge/case_manifest.csv   reader-study case manifest
#   4) results/reader_study_merge/images/             reader-facing image folder
#
# NOTE: these scripts only READ the model outputs (test_results.csv). They do
# NOT re-run inference; regenerating test_results.csv is a separate step
# (test_mil_lp_*.py). Report mode is best-seed (max), i.e. exploratory upper
# bound -- switch REPORT_MODE=mean for paper-grade mean+/-std numbers.
#
# The case manifest (step 3) uses PATIENT-INDEPENDENT sampling: region-scoped
# one-patient-one-study uniqueness, scarce-first conflict resolution, 1:1 balance
# toward POS_TARGETS, + EXTRA unscored cases per disease. Per-disease seed choice
# uses objective=min_sens_spec_gapmatch: rank seeds by min(sens,spec), keep the
# best one whose sens-vs-spec relationship matches the full test set (gapmatch,
# GAP_FRAC) and whose per-metric inflation vs the full test set stays within
# MAX_INFLATION points -- so the read subset represents true model behaviour.
# Step 4 copies the images; pass a retrieval-augmented manifest to also copy
# retrieved reference images (see RETRIEVAL_MANIFEST below).
#
# Usage:
#   bash run_reader_study.sh             # run everything
#   REPORT_MODE=mean bash run_reader_study.sh   # mean+/-std instead of best-seed

set -euo pipefail
# script lives in scripts/, run everything from the mil_linear_probe root
cd "$(dirname "$0")/.."

# --- conda env (vlp) ---
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate vlp
fi

# --- shared parameters ---
REPORT_MODE="${REPORT_MODE:-max}"        # max (best-seed, exploratory) | mean
QUOTA="${QUOTA:-20}"
MIL="${MIL:-clam}"
MODEL_ID="${MODEL_ID:-topk_v19_last}"
FOCUS="肾结石 尿路结石 隐裂 峡部裂 腰椎骶化 气胸 肠梗阻 终板炎 骨质疏松 纤维灶"
SEL="气胸 肠梗阻 肾结石 隐裂 峡部裂"
EXTERNAL="ncfy_pro ncfy_ret nyfs_pro zsfs_pro zsfs_ret zsfy"
MERGE_SOURCES="private_a nyfs_pro ncfy_pro ncfy_ret"
MANIFEST_DISEASES="气胸 肠梗阻 肾结石 隐裂 峡部裂"
# manifest sampling: sources in PRIORITY order (scarce external first), per-
# disease positive target, and N extra unscored cases per disease.
MANIFEST_SOURCES="ncfy_ret private_a"
POS_TARGETS="气胸=40 肠梗阻=40 肾结石=40 隐裂=40 峡部裂=40"
EXTRA_PER_DISEASE="${EXTRA_PER_DISEASE:-4}"
EXTRA_MAX_POS="${EXTRA_MAX_POS:-2}"
# best-seed objective: rank by min(sens,spec), then keep the best seed whose
# sens-vs-spec relationship matches the full test set (gapmatch), and whose
# per-metric inflation vs the full test set stays <= MAX_INFLATION points.
OBJECTIVE="${OBJECTIVE:-min_sens_spec_gapmatch}"
GAP_FRAC="${GAP_FRAC:-0.5}"                # gapmatch: |gap| >= GAP_FRAC*|full_gap|
MAX_INFLATION="${MAX_INFLATION:-10}"      # cap: no metric inflated > 10pt vs full test
UNIQUENESS="${UNIQUENESS:-region}"        # patient-uniqueness scope (per region)
# 气胸 is sampled from SIIM-ACR (too few in-house pneumothorax positives):
DISEASE_CSV="气胸=results/reader_study_merge/siim_acr_csv_manifiest.csv"
# optional retrieval-augmented manifest (must carry image_dir + the retrieval
# JSON column); if present, step 4 also copies retrieved reference images.
RETRIEVAL_MANIFEST="${RETRIEVAL_MANIFEST:-results/reader_study_merge/case_manifest_add_retieval.csv}"
RETRIEVAL="${RETRIEVAL:-on}"          # on = copy retrieved refs too | off = views only

echo "==== report_mode=${REPORT_MODE}  quota=${QUOTA}  mil=${MIL}  model=${MODEL_ID} ===="

# === 1) per-dataset balanced eval (in-domain + 6 external) ===
echo ">>> [1/3] balanced eval (in-domain private_a)"
python reader_study_balanced_eval.py \
  --model_id "$MODEL_ID" --diseases $FOCUS --selected $SEL \
  --report_mode "$REPORT_MODE" --plot

echo ">>> [1/3] balanced eval (external datasets)"
python reader_study_balanced_eval.py \
  --model_id "$MODEL_ID" --diseases $FOCUS --selected $SEL \
  --report_mode "$REPORT_MODE" --plot \
  --datasets $EXTERNAL

# === 2) multi-center equal-quota merge eval ===
echo ">>> [2/3] merge eval (per-source equal quota)"
python reader_study_merge_eval.py \
  --model_id "$MODEL_ID" --sources $MERGE_SOURCES \
  --quota "$QUOTA" --report_mode "$REPORT_MODE" --mil_types "$MIL" \
  --diseases $FOCUS --selected $SEL --plot

# === 3) reader-study case manifest (patient-independent sampling) ===
echo ">>> [3/4] case manifest (patient-independent + extras)"
python reader_study_case_manifest.py \
  --model_id "$MODEL_ID" --sources $MANIFEST_SOURCES \
  --diseases $MANIFEST_DISEASES --pos_targets $POS_TARGETS \
  --disease_csv "$DISEASE_CSV" \
  --objective "$OBJECTIVE" --gap_frac "$GAP_FRAC" \
  --max_inflation "$MAX_INFLATION" --uniqueness "$UNIQUENESS" \
  --extra_per_disease "$EXTRA_PER_DISEASE" --extra_max_pos "$EXTRA_MAX_POS" \
  --mil_type "$MIL" \
  --out results/reader_study_merge/case_manifest.csv

# === 4) copy the reader-facing image folder ===
# If a retrieval-augmented manifest exists (image_dir + retrieval JSON column),
# copy views + retrieved reference images from it; otherwise copy views only
# from the plain manifest. Re-run your retrieval step on the NEW manifest first
# to produce RETRIEVAL_MANIFEST with an image_dir column.
echo ">>> [4/4] copy images"
if [ "$RETRIEVAL" = "on" ] && [ -f "$RETRIEVAL_MANIFEST" ] && \
   python -c "import sys,pandas as pd; sys.exit(0 if 'image_dir' in pd.read_csv('$RETRIEVAL_MANIFEST',nrows=1).columns else 1)" 2>/dev/null; then
  echo "    using retrieval-augmented manifest: $RETRIEVAL_MANIFEST"
  python copy_reader_study_images.py --manifest "$RETRIEVAL_MANIFEST" --clean --retrieval on
else
  echo "    copying views only (RETRIEVAL=$RETRIEVAL)"
  python copy_reader_study_images.py \
    --manifest results/reader_study_merge/case_manifest.csv --clean --retrieval off
fi

echo "==== DONE. Outputs under results/reader_study_balanced/ and results/reader_study_merge/ ===="
