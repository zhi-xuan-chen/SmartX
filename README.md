# SmartX

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/smartx_icon.png">
    <img src="docs/assets/smartx_icon.png" alt="SmartX icon" width="160" style="background: #ffffff; padding: 16px; border-radius: 12px;">
  </picture>
</p>

SmartX is a clinically evaluated vision-language foundation model for generalizable multi-region X-ray interpretation. The model was pretrained on case-level radiograph-report pairs spanning ten anatomical regions and is designed to learn from the native format of radiology archives: paired studies and reports, without requiring curated image-level annotations. This repository provides the public code interface for loading SmartX, running lightweight examples, reproducing the TopK pretraining workflow, and adding downstream evaluations.

## Installation

Use Python 3.10 or newer. A conda environment named `smartx` is recommended:

```bash
git clone <SMARTX_REPOSITORY_URL>
cd SmartX
conda create -n smartx python=3.10 -y
conda activate smartx
pip install -r requirements.txt
```

## Model Weights

Download the public checkpoint from Hugging Face or let the loader download it automatically:

```text
https://huggingface.co/Trusure/SmartX/blob/main/smartx_large.ckpt
```

Optional local layout:

```text
checkpoints/
└── smartx_large.ckpt
```

See `docs/model_zoo.md` for model details.

## Loading SmartX

Use the shared loader in `downstream/modules/`:

```python
from downstream.modules import create_clip_style_model_global

vision_model, text_model, tokenizer, model_info = create_clip_style_model_global({
    "model_name": "smartx_large",
    "checkpoint": "checkpoints/smartx_large.ckpt",  # optional; downloads from HF if omitted
})
```

For the full VLP module:

```python
from downstream.modules import create_smartx_vlp_model

model, tokenizer, load_report = create_smartx_vlp_model("checkpoints/smartx_large.ckpt")
```

## Demo

The `demo/` directory is reserved for small reviewer-facing examples that do not require full benchmark datasets.

Current planned entry points:

```bash
python demo/extract_features.py --config demo/configs/extract_features.yaml --checkpoint checkpoints/smartx_large.ckpt
python demo/zero_shot.py --config demo/configs/zero_shot.yaml --checkpoint checkpoints/smartx_large.ckpt
python demo/retrieval.py --config demo/configs/retrieval.yaml --checkpoint checkpoints/smartx_large.ckpt
```

The demo scripts are still listed in `docs/release_checklist.md` because they need final runnable implementations and public demo images before release.

## Pretraining

TopK pretraining code is under `pretrain/`.

```bash
python pretrain/run.py --config pretrain/configs/topk_ark_base.example.yaml
```

The example config uses placeholder paths. Replace them with local public or approved data paths. Do not commit private data paths.

## Downstream Tasks

Downstream code is organized under `downstream/`:

- `downstream/modules/`: shared SmartX model-loading helpers.
- `downstream/mil_linear_probe/`: included MIL linear-probe code.
- `downstream/report_generation/`: included LLaVA-Rad report-generation code.
- `downstream/zeroshot/`: prompt-based zero-shot classification.
- `downstream/retrieval/`: image-text retrieval with public manifest inputs.
- `downstream/linear_probe/`: cached-feature extraction, training, and evaluation.
- `downstream/detection/`: SmartX-backed VitDet training/evaluation for COCO-format annotations.

For downstream release conventions, follow `docs/downstream_task_guide.md`.

## Tests

Run the lightweight release checks with:

```bash
python -m pytest -q
```

These tests check repository structure, config templates, release checklist visibility, and lightweight utility behavior. They do not run full training or full benchmark evaluation.

## Release Status

Known remaining work is tracked in `docs/release_checklist.md`.

Before public release, verify that:

- reviewer-facing demos run on public demo assets
- downstream tasks use public configs and no private data paths
- `CITATION.cff` and `docs/code_availability.md` metadata are filled
- no private data, private paths, checkpoints, cached features, W&B runs, or generated results are committed

## Documentation

- `docs/model_zoo.md`: model files and loading notes
- `docs/model_card.md`: intended use and limitations
- `docs/data_availability.md`: data-sharing statement
- `docs/code_availability.md`: code availability statement
- `docs/reproducibility.md`: reproduction command structure
- `docs/downstream_task_guide.md`: downstream task release conventions
- `docs/release_checklist.md`: remaining work before release
