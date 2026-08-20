# Case-level MIL linear probe

This folder provides the public SmartX multi-instance learning workflow for
case-level abnormality classification. The SmartX vision encoder is frozen;
a lightweight class-specific gated attention head pools every radiographic
view of a case into one prediction per finding.

Checkpoint loading uses the shared
`downstream.modules.create_clip_style_model_global()` public loader.

## Data format

Provide separate train, validation and test manifests. Each row is one case:

```csv
case_uid,image_paths,labels
case_001,"front.png;lateral.png","[1,0,1]"
case_002,"single.png","[0,1,0]"
```

Alternatively, store one numeric label per column and configure
`dataset.label_columns`. The runner supports binary and multi-label tasks,
ordinary images, DICOM, MHA/MHD and NIfTI.

## Run

```bash
python downstream/mil_linear_probe/extract_features.py --config downstream/mil_linear_probe/configs/mil_linear_probe.example.yaml
python downstream/mil_linear_probe/train_mil_linear_probe.py --config downstream/mil_linear_probe/configs/mil_linear_probe.example.yaml
python downstream/mil_linear_probe/test_mil_linear_probe.py --config downstream/mil_linear_probe/configs/mil_linear_probe.example.yaml --checkpoint outputs/mil_linear_probe/run/mil_linear_probe_best.pt
```

Feature extraction writes padded case tensors and validity masks for each
split. Training selects the best checkpoint by validation macro AUROC and
stores validation-derived per-class thresholds in that checkpoint. Test F1
and MCC use those fixed validation thresholds rather than test-set thresholds.

Outputs include cached features, `train_history.json`, `summary.json`,
`metrics.json`, per-case predictions and optional attention weights.

See [`docs/downstream_task_guide.md`](../../docs/downstream_task_guide.md) for
release conventions. Do not commit manifests, images, cached features,
checkpoints, predictions or machine-specific paths.
