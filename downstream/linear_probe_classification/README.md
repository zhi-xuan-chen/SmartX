# Linear-Probe Classification

Frozen-feature linear-probe classification using [`downstream.modules.create_clip_style_model_global()`](../modules/README.md).

## Workflow

```bash
# 1. Extract and cache frozen image features
python downstream/linear_probe_classification/extract_features.py   --config downstream/linear_probe_classification/configs/linear_probe_classification.example.yaml

# 2. Train linear classifier on cached features
python downstream/linear_probe_classification/train_linear_probe_classification.py   --config downstream/linear_probe_classification/configs/linear_probe_classification.example.yaml

# 3. Evaluate saved checkpoint
python downstream/linear_probe_classification/test_linear_probe_classification.py   --config downstream/linear_probe_classification/configs/linear_probe_classification.example.yaml   --checkpoint outputs/linear_probe_classification/run/linear_probe_classification_best.pt
```

## Feature Cache & Thresholding

- **Cached files**: `{split}_features.pt`, `{split}_labels.pt`, `{split}_samples.json`, and `metadata.json`.
- **Thresholding**: Binary and multi-label F1/MCC use per-class Youden thresholds fitted on the validation split and saved inside the trained checkpoint for test evaluation.

See [`docs/downstream_task_guide.md`](../../docs/downstream_task_guide.md) for full input formats and task workflows.
