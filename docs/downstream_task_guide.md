# Downstream Task Guide

Run all commands from the repository root after installing `requirements.txt`. Prepare your local datasets in the format described by each task guide, then update the corresponding example YAML configuration or script environment variables.

## Task Overview

| Task | Workflow | Entry Point | Directory & Guide |
| :--- | :--- | :--- | :--- |
| **Zero-shot classification** | Encode images and text prompts, then evaluate | `--config` YAML | [`downstream/zero_shot_classification/`](../downstream/zero_shot_classification/README.md) |
| **Image–text retrieval** | Encode radiograph–report pairs, rank, and evaluate | `--config` YAML | [`downstream/retrieval/`](../downstream/retrieval/README.md) |
| **Linear-probe classification** | Extract frozen features, train linear head, evaluate | `--config` YAML | [`downstream/linear_probe_classification/`](../downstream/linear_probe_classification/README.md) |
| **Examination-level classification** | Extract multi-view features, train MIL head, evaluate | `--config` YAML | [`downstream/examination_level_classification/`](../downstream/examination_level_classification/README.md) |
| **Detection** | Train SmartX-backed ViTDet on COCO annotations, evaluate | `--config` YAML | [`downstream/detection/`](../downstream/detection/README.md) |
| **Report generation** | Two-stage LLaVA-Rad training, inference, and clinical scoring | Shell scripts | [`downstream/report_generation/`](../downstream/report_generation/README.md) |

> **Note:** Detection and report generation require additional task-specific dependencies documented in their respective READMEs. For complete report-generation scoring (including BERTScore and GREEN), see [Complete Report Evaluation](../downstream/report_generation/README.md#complete-report-evaluation).

## Shared Model Loading

All tasks load SmartX through the [shared loaders](../downstream/modules/README.md) (including `create_clip_style_model_global`). Supply a local checkpoint path via `SMARTX_CHECKPOINT` or YAML config, or leave it unset to download automatically from [`Trusure/SmartX`](https://huggingface.co/Trusure/SmartX).

## Lightweight Verification

```bash
python -m pytest -q
```

These tests validate configuration templates, model loaders, metric utilities, and lightweight runtime paths. Full benchmark reproduction requires the corresponding datasets, weights, and optional task dependencies.
