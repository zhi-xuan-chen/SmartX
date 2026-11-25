# Report generation

This folder contains the SmartX report-generation workflow used for the
MIMIC-CXR and IU X-Ray experiments. It adapts LLaVA-Rad with a frozen SmartX
vision encoder and a Vicuna 7B language backbone.

## Installation

Install the base project first, followed by the task-specific dependencies:

```bash
pip install -r requirements.txt
pip install -r downstream/report_generation/requirements-optional.txt
```

Training requires CUDA GPUs with enough memory for Vicuna 7B. Dataset access
and use remain subject to the original MIMIC-CXR and IU X-Ray licenses.

## Data

Set the data root before running any command:

```bash
export SMARTX_DATA_ROOT=/path/to/report-generation-data
```

The scripts expect this layout:

```text
${SMARTX_DATA_ROOT}/
├── MIMIC-CXR/
│   ├── imgs-1024/
│   └── labels/
│       ├── mimic_cxr_train_llava_rad_frontal_rule_and_gpt4_use_reason_301681.csv
│       └── mimic_cxr_test_llava_rad_frontal_2461.csv
└── IUXray/
    ├── images/images_normalized/
    ├── indiana_projections.csv
    └── labels/
        ├── indiana_meta_frontal_findings_w_chexbert_labels_train.csv
        └── indiana_meta_frontal_findings_w_chexbert_labels_test.csv
```

MIMIC CSV files require `id`, `image` (or `image_path`) and `reference` (or
`findings`). Optional columns are `query`, `view`, `reason` and
`generate_method`. IU CSV files require `uid`, `findings` and either `filename`
or a matching `uid` in `indiana_projections.csv`; `projection` and `indication`
are optional. Image paths are relative to the corresponding image directory.

The filenames above match the script defaults. Prepare your local CSV files
with the required columns; dataset files are not included.

## Models

Create the merged LLaVA-Rad base once:

```bash
bash downstream/report_generation/llavarad/scripts/merge_llavarad.sh
```

This merges `lmsys/vicuna-7b-v1.5` with `microsoft/llava-rad` into
`outputs/report_generation/llavarad_merged` by default. Their licenses and
access requirements apply.

The SmartX checkpoint is resolved from `Trusure/SmartX` automatically. To use
a local copy instead:

```bash
export SMARTX_CHECKPOINT=/path/to/smartx_large.ckpt
```

## Run SmartX

Each entry point accepts `stage1`, `stage2`, `eval` or `all`:

```bash
bash downstream/report_generation/llavarad/scripts/run_mimic_smartx.sh all
bash downstream/report_generation/llavarad/scripts/run_iu_smartx.sh all
```

To evaluate an explicit Stage 2 checkpoint without training:

```bash
bash downstream/report_generation/llavarad/scripts/run_mimic_smartx.sh eval \
  outputs/report_generation/llavarad_merged \
  outputs/report_generation/mimic/llavarad_smartx_large_stage2
```

Training and decoding settings are defined in the corresponding
[scripts](llavarad/scripts/).

W&B is disabled by default. Set `REPORT_TO=wandb` and `WANDB_MODE=online` to
enable it. Override `CUDA_VISIBLE_DEVICES`, `REPORT_OUTPUT_ROOT` or
`CHECKPOINT_ROOT` as needed.

Predictions default to `llavarad/results/<run_name>/`. Follow
[Complete report evaluation](#complete-report-evaluation) below to compute all
paper metrics and combine them in `metrics.csv`. Evaluate the checkpoint
selected on validation data.

## Paper baselines

MIMIC-CXR and IU X-Ray scripts are provided for Ark+, RAD-DINO,
CheSS, BiomedCLIP, MaCo, CheXzero and GLoRIA. Their weights, preprocessing and
licenses come from the respective upstream projects. For models that do not
download weights automatically, pass the local vision checkpoint as the third
argument to the corresponding `run_<dataset>_<model>.sh` script.

## Acknowledgements

This workflow builds on [LLaVA](https://github.com/haotian-liu/LLaVA) and
[LLaVA-Rad](https://github.com/microsoft/LLaVA-Rad). We thank their authors
for making their research and resources available. Use of upstream code,
models and weights remains subject to their respective licences.

## Complete report evaluation

The paper evaluation includes BLEU-4, ROUGE-L, RadGraph F1, CheXbert
micro/macro-F1 over 14 positive findings, BERTScore F1 and GREEN.
Compute them in two environments using the same `predictions.jsonl` and
output directory. Both scoring steps automatically update `metrics.csv`.

Run the following from the repository root. Set dataset paths as described
above and use your trained Stage 2 checkpoint.

### 1. Generate predictions and compute standard metrics

In the report-generation environment:

```bash
conda activate smartx
export SMARTX_DATA_ROOT=/path/to/report-generation-data
export REPORT_EVAL_DIR="$PWD/outputs/report_generation/mimic_eval"
export WANDB_MODE=disabled
bash downstream/report_generation/llavarad/scripts/run_mimic_smartx.sh eval \
  "$PWD/outputs/report_generation/llavarad_merged" \
  "$PWD/outputs/report_generation/mimic/llavarad_smartx_large_stage2" \
  smartx_mimic_eval "$REPORT_EVAL_DIR"
```

This creates `$REPORT_EVAL_DIR/predictions.jsonl` and writes standard metrics
to `$REPORT_EVAL_DIR/eval/`. At this point `metrics.csv` marks BERTScore and
GREEN as `missing`: the complete paper evaluation is not finished yet.
For IU X-Ray use `run_iu_smartx.sh`, the corresponding IU Stage 2 checkpoint,
and a separate output directory.

To score an existing prediction file without regenerating reports:

```bash
python downstream/report_generation/llavarad/llava/eval/rrg_eval/run.py \
  "$REPORT_EVAL_DIR/predictions.jsonl" \
  --output_dir "$REPORT_EVAL_DIR/eval"
```

### 2. Compute BERTScore and GREEN

GREEN requires a separate environment because its dependencies conflict with
the training environment. Create it once:

```bash
conda create -n smartx-metrics python=3.10 -y
conda activate smartx-metrics
python -m pip install -r downstream/report_generation/requirements-model-metrics.txt
```

In the same shell, keep `REPORT_EVAL_DIR` from step 1 and run:

```bash
CUDA_VISIBLE_DEVICES=0 python downstream/report_generation/llavarad/llava/eval/rrg_eval/rrg_eval/model_metrics.py \
  "$REPORT_EVAL_DIR/predictions.jsonl" \
  --metrics BERTScore GREEN \
  --output-dir "$REPORT_EVAL_DIR/eval"
```

The first run downloads scoring weights. Use
`--green-model /path/to/GREEN-radllama2-7b` for a local copy matching the
configured revision. Expose one GPU for GREEN; `--device cpu` is supported
but slow. Scoring models and settings are defined in `model_metrics.py`.

### Results and completion status

All results are under `$REPORT_EVAL_DIR/eval/`:

| File | Contents |
| --- | --- |
| `metrics.csv` | Combined paper metrics, scores, scales and completion status |
| `metrics_manifest.json` | Prediction-file SHA-256, row count and metric status |
| `main.csv` | Standard evaluation scores from step 1 |
| `model_metrics.csv` | BERTScore/GREEN scores from step 2 |
| `breakdown_p.csv`, `breakdown_n.csv` | Per-condition CheXbert results |
| `BERTScore_summary.json`, `GREEN_summary.json` | Valid/invalid counts and scoring configuration |
| `BERTScore_details.json`, `GREEN_details.json` | Per-row scores; GREEN also retains judge responses |
| `cache/` | Resumable model-metric scoring cache |

`metrics.csv` includes `metric`, `score`, `scale`, `status`, `n_total`,
`n_valid`, `n_invalid`, `prediction_sha256` and `error`. Scores retain their
native scales: BLEU is 0–100; other conventional metrics and GREEN are 0–1;
baseline-rescaled BERTScore F1 can be negative. CheXbert paper rows are
`Micro-F1-14+` and `Macro-F1-14+`; RadGraph is `F1-RadGraph`.

- `ok`: metric completed successfully.
- `missing`: metric has not been run for this prediction file.
- `running`: scoring started; if the process stopped, rerun that scoring step.
- `failed`: scoring failed; the score is blank and `error` describes the failure.
- `partial`: a score was computed from only valid rows; inspect the invalid counts.

Both commands verify the exact prediction-file bytes with SHA-256 before
updating results. A different file fingerprint is rejected without merging;
use a new output directory for different predictions. Run scoring steps
sequentially, with one process per output directory. Existing CSV files without
a matching manifest are not imported as verified results; rerun both steps.

A failed metric does not prevent attempts at the other requested metrics, but
the command exits with an error after saving results. Failed scores are never
replaced by zero. GREEN averages valid rows and marks incomplete coverage as
`partial`; no valid scores means `failed`. Review all seven paper metric rows
before treating evaluation as complete. Rerun step 2 to reuse cached scores
and retry invalid responses.
