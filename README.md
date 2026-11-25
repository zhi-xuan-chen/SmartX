# SmartX

<p align="center">
  <img src="docs/assets/smartx_icon_white.png" alt="SmartX" width="180">
</p>

<h1 align="center">An open vision–language foundation model for<br>real-world multi-anatomy X-ray interpretation</h1>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python 3.10" src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white"></a>
  <a href="https://github.com/zhi-xuan-chen/SmartX/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="https://huggingface.co/Trusure/SmartX"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97-Model_Weights-yellow"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> · <a href="#quick-start">Quick Start</a> · <a href="#demo">Demo</a> · <a href="#downstream-tasks">Downstream Tasks</a> · <a href="#pretraining">Pretraining</a> · <a href="#documentation">Documentation</a>
</p>

---

## Overview

**SmartX** is an open vision–language foundation model for real-world multi-anatomy X-ray interpretation, pretrained on **2.7 million radiographs** with paired clinical reports across **10 anatomical regions**. Using progressive global-to-local alignment, it learns transferable vision–language representations directly from native clinical records without requiring extra curated image-level annotations.

This repository provides model loading utilities, pretraining code, a runnable demo notebook, and implementations of **six downstream tasks** for inference, transfer training, and evaluation.

---

## Quick Start

### 1. Installation

SmartX is validated with **Python 3.10**:

```bash
git clone https://github.com/zhi-xuan-chen/SmartX.git
cd SmartX
conda create -n smartx python=3.10 -y
conda activate smartx
python -m pip install -r requirements.txt
```

### 2. Model Weights

| Model | Vision Backbone | Config Name | Checkpoint |
| :--- | :--- | :--- | :--- |
| **SmartX** | Swin-L | `smartx_large` | [`smartx_large.ckpt`](https://huggingface.co/Trusure/SmartX/resolve/main/smartx_large.ckpt) |

When `checkpoint` is omitted, the shared loader automatically downloads `smartx_large.ckpt` from [`Trusure/SmartX`](https://huggingface.co/Trusure/SmartX). To use a local copy instead, place it at `checkpoints/smartx_large.ckpt`.

### 3. Loading SmartX

**CLIP-style image & text encoders** (for zero-shot classification, retrieval, and feature extraction):

```python
from downstream.modules import create_clip_style_model_global

vision_model, text_model, tokenizer, model_info = create_clip_style_model_global({
    "model_name": "smartx_large",
    # "checkpoint": "checkpoints/smartx_large.ckpt",  # optional local path
})
```

**Full vision–language module:**

```python
from downstream.modules import create_smartx_vlp_model

model, tokenizer, load_report = create_smartx_vlp_model(
    "checkpoints/smartx_large.ckpt"
)
```

---

## Demo

Run the end-to-end [SmartX demo notebook](demo/smartx_demo.ipynb) for **zero-shot classification** and **image–text retrieval** on public-domain X-rays using pretrained weights only (supports both CPU and GPU):

```bash
python -m pip install notebook
jupyter notebook demo/smartx_demo.ipynb
```

> The notebook uses `checkpoints/smartx_large.ckpt` if present, or downloads it automatically from Hugging Face [`Trusure/SmartX`](https://huggingface.co/Trusure/SmartX).

---

## Downstream Tasks

All six downstream workflows use the shared [SmartX loaders](downstream/modules/README.md). See the [Downstream Task Guide](docs/downstream_task_guide.md) for full input schemas and workflows.

| Task | Description | Code & Guide |
| :--- | :--- | :--- |
| **Zero-shot classification** | Prompt-based abnormality recognition without task training | [`downstream/zero_shot_classification/`](downstream/zero_shot_classification/README.md) |
| **Image–text retrieval** | Bidirectional radiograph–report embedding retrieval | [`downstream/retrieval/`](downstream/retrieval/README.md) |
| **Linear-probe classification** | Frozen-feature extraction, linear training, and evaluation | [`downstream/linear_probe_classification/`](downstream/linear_probe_classification/README.md) |
| **Examination-level classification** | Multi-view attention-based multiple-instance learning (MIL) | [`downstream/examination_level_classification/`](downstream/examination_level_classification/README.md) |
| **Detection** | SmartX-backed ViTDet for COCO-format abnormality localization | [`downstream/detection/`](downstream/detection/README.md) |
| **Report generation** | LLaVA-Rad two-stage training and clinical report evaluation | [`downstream/report_generation/`](downstream/report_generation/README.md) |

---

## Pretraining

To continue pretraining from the released SmartX weights, configure your data paths in [`pretrain/configs/smartx_large.example.yaml`](pretrain/configs/smartx_large.example.yaml) and run:

```bash
python pretrain/run.py --config pretrain/configs/smartx_large.example.yaml
```

See the [Pretraining Guide](pretrain/README.md) for dataset formatting, local/automatic weight loading, and multi-GPU distributed launch.

---

## Documentation

| Document | Description |
| :--- | :--- |
| [Model Zoo](docs/model_zoo.md) | Checkpoint details and local directory layout |
| [Model Card](docs/model_card.md) | Intended use, scope, and limitations |
| [Downstream Task Guide](docs/downstream_task_guide.md) | Input formats and workflows for all six downstream tasks |
| [Running SmartX](docs/reproducibility.md) | End-to-end execution guide and lightweight verification (`python -m pytest -q`) |
| [Data Availability](docs/data_availability.md) | Public datasets, demo assets, and clinical data governance |
| [Code Availability](docs/code_availability.md) | Repository contents and release notes |

---

## Citation

If you find SmartX useful in your research, please cite the associated manuscript. Complete author metadata is provided in [`CITATION.cff`](CITATION.cff).

## License

This repository is released under the [MIT License](LICENSE). Third-party models and datasets remain subject to their respective licenses and terms of use.
