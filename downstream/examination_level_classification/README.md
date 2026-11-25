# Examination-Level Classification

Multi-view case-level abnormality classification combining a frozen SmartX vision encoder with a lightweight class-specific gated attention MIL head. Models are loaded via [`downstream.modules.create_clip_style_model_global()`](../modules/README.md).

## Data Format

Provide separate `train`, `valid`, and `test` manifests with one case per row (supports standard images, DICOM, MHA/MHD, and NIfTI):

```csv
case_uid,image_paths,labels
case_001,"front.png;lateral.png","[1,0,1]"
case_002,"single.png","[0,1,0]"
```

Alternatively, provide one numeric column per abnormality and configure `dataset.label_columns`.

## Workflow

```bash
# 1. Extract multi-view case features
python downstream/examination_level_classification/extract_features.py   --config downstream/examination_level_classification/configs/examination_level_classification.example.yaml

# 2. Train class-specific gated attention MIL head
python downstream/examination_level_classification/train_examination_level_classification.py   --config downstream/examination_level_classification/configs/examination_level_classification.example.yaml

# 3. Evaluate best checkpoint on test split
python downstream/examination_level_classification/test_examination_level_classification.py   --config downstream/examination_level_classification/configs/examination_level_classification.example.yaml   --checkpoint outputs/examination_level_classification/run/examination_level_classification_best.pt
```

## Outputs & Thresholding

- Training selects the best checkpoint by validation macro AUROC and stores validation-derived per-class thresholds in the checkpoint for testing.
- Outputs include cached padded case tensors and masks, `train_history.json`, `summary.json`, `metrics.json`, per-case predictions, and optional attention weights.

See [`configs/examination_level_classification.example.yaml`](configs/examination_level_classification.example.yaml) and [`docs/downstream_task_guide.md`](../../docs/downstream_task_guide.md) for full details.
