# Downstream Task Guide

This guide defines the public release conventions for the downstream tasks:

- `downstream/zeroshot/`
- `downstream/retrieval/`
- `downstream/linear_probe/`
- `downstream/detection/`
- `downstream/mil_linear_probe/`

Use the shared SmartX loader instead of writing a new checkpoint loader:

```python
from downstream.modules import create_clip_style_model_global

vision_model, text_model, tokenizer, model_info = create_clip_style_model_global({
    "model_name": "smartx_large",
    "checkpoint": "checkpoints/smartx_large.ckpt",  # optional; downloads from Trusure/SmartX if omitted
})
```

## Required Shape

Each task should keep only the files needed to run and understand that task:

```text
downstream/<task>/
├── README.md
├── configs/
│   └── <task>.example.yaml
├── run_<task>.py
└── metrics.py        # only if the task has its own metrics
```

If the task needs dataset code, add `datasets.py`. If it needs feature caching, add `extract_features.py`. Framework folders should contain only code used by the release entry points.

## Required Behavior

Each task must support:

- `--config configs/<task>.example.yaml`
- local paths through config values or environment variables
- CPU smoke mode on tiny demo data when possible
- output under a user-specified `output_dir`
- no writes into the repository except explicit demo outputs ignored by git

Do not commit:

- private images, reports, labels, splits, or reader-study data
- generated predictions, metrics tables, plots, cached features, W&B runs, or checkpoints
- private or machine-local absolute paths

## Per-Task Minimums

### Zero-Shot

Add:

- `run_zeroshot.py`: load SmartX, encode prompts, encode images, compute class scores.
- `configs/zeroshot.example.yaml`: demo image folder, label file, prompts, checkpoint/output settings.
- `metrics.py`: AUROC/AP/accuracy helpers only if not already covered by shared utilities.

Smoke check: run on `demo/assets/images` and `demo/assets/zero_shot_labels.json`.

### Retrieval

Add:

- `run_retrieval.py`: encode images and texts, compute similarity matrix, write rankings.
- `configs/retrieval.example.yaml`: image/text pair file, top-k values, checkpoint/output settings.
- `metrics.py`: Recall@K, median rank, mean rank.

Smoke check: run on `demo/assets/retrieval_pairs.jsonl`.

### Linear Probe

Add:

- `extract_features.py`: export SmartX image embeddings.
- `train_linear_probe.py`: train a simple linear classifier on cached features.
- `configs/linear_probe.example.yaml`: feature paths, labels, split files, output settings.
- `metrics.py`: classification metrics used by the manuscript task.

Smoke check: train for one epoch on tiny synthetic/demo features committed as small text/JSON/NPY only if license-safe and non-private.

### Case-level MIL Linear Probe

Add:

- `extract_features.py`: export padded case-level SmartX embeddings and view masks.
- `train_mil_linear_probe.py`: train the manuscript CLAM-style attention head.
- `test_mil_linear_probe.py`: evaluate with validation-derived thresholds.
- `configs/mil_linear_probe.example.yaml`: generic case manifests and output settings.

Smoke check: train, save, reload and evaluate on tiny synthetic feature bags.
Keep private cohort schemas, region-specific wrappers and analysis outputs out
of the release repository.

### Detection

Add:

- `run_detection.py` or `train_detection.py`: use SmartX features/backbone with the chosen detection head.
- `configs/detection.example.yaml`: dataset annotation schema, image root, checkpoint/output settings.
- `metrics.py`: mAP/IoU helpers only if not supplied by the detection framework.

Smoke check: parse a tiny public/demo annotation file and run one forward/eval pass. Do not commit full detection datasets.

## Tests

Add one lightweight test per task under `tests/` after implementation. Prefer import/config/metric smoke checks. Do not add GPU or full-dataset tests to the release test suite.
