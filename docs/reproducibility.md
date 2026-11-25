# Running SmartX

This guide summarizes the end-to-end setup and execution paths across the repository.

## 1. Environment and Model Weights

Follow the [Quick Start](../README.md#quick-start) to create the Python 3.10 environment and install `requirements.txt`. The released model configuration is `smartx_large` (`smartx_large.ckpt`; see [Model Zoo](model_zoo.md)).

## 2. Runnable Demo

Verify model loading and inference on bundled public-domain X-rays:

```bash
jupyter notebook demo/smartx_demo.ipynb
```

See the [Demo Guide](../demo/README.md) and [Demo Asset Provenance](../demo/assets/README.md) for details.

## 3. Downstream Tasks

Follow the [Downstream Task Guide](downstream_task_guide.md) for all six evaluation workflows, input schemas, and optional dependencies (Detectron2 for detection; LLaVA-Rad and a separate scoring environment for report generation).

## 4. Pretraining

Continue pretraining from the released SmartX weights using [`pretrain/configs/smartx_large.example.yaml`](../pretrain/configs/smartx_large.example.yaml). See the [Pretraining Guide](../pretrain/README.md) for dataset setup and multi-GPU `torchrun` commands.

## 5. Lightweight Tests

From the repository root:

```bash
python -m pytest -q
```

Tests cover repository structure, configuration templates, loaders, metrics, and runtime regressions (optional Detectron2 tests skip automatically when Detectron2 is not installed).
