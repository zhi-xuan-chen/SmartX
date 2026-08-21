# Report generation

This folder contains the public SmartX report-generation workflow used for the
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

The public scripts expect this layout:

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

The paper evaluates 301,681 MIMIC-CXR training pairs and 2,461 test pairs, and
2,314 IU X-Ray training pairs and 993 test pairs. Dataset files are not
included in this repository.

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

Each public entry point accepts `stage1`, `stage2`, `eval` or `all`:

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

Stage 1 freezes the language and vision models and trains the two-layer GELU
projector for one epoch with learning rate `1e-3`, warm-up ratio `0.03` and
cosine decay. Stage 2 trains the projector and language-model LoRA adapters for
three epochs using rank 64, alpha 128, dropout 0.05, learning rate `2e-4`,
gradient accumulation 2 and maximum sequence length 2,048. Evaluation uses
temperature 0, no sampling, one beam and at most 256 new tokens.

W&B is disabled by default. Set `REPORT_TO=wandb` and `WANDB_MODE=online` to
enable it. Override `CUDA_VISIBLE_DEVICES`, `SMARTX_OUTPUT_ROOT` or
`CHECKPOINT_ROOT` as needed.

Predictions are written to `llavarad/results/`, and evaluation produces
BLEU-4, ROUGE-L, RadGraph F1, CheXbert micro/macro F1 and 95% bootstrap
confidence intervals. Evaluation runs the checkpoint explicitly supplied by
the user; it never selects a checkpoint using test-set scores.

## Paper baselines

Matching MIMIC-CXR and IU X-Ray wrappers are retained for Ark+, RAD-DINO,
CheSS, BiomedCLIP, MaCo, CheXzero and GLoRIA. Their weights, preprocessing and
licenses come from the respective upstream projects. For models that do not
download weights automatically, pass the local vision checkpoint as the third
argument to the corresponding `run_<dataset>_<model>.sh` script.

Do not commit datasets, merged language models, checkpoints, predictions,
metric caches, W&B runs or machine-specific paths.
