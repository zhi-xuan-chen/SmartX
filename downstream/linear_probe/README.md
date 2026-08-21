# Linear Probe

This folder provides the SmartX linear-probe workflow adapted from the evaluation repository. It uses [`downstream.modules.create_clip_style_model_global()`](../modules/README.md) for checkpoint loading and keeps dataset paths in user config files.

## Entry Points

```bash
python downstream/linear_probe/extract_features.py --config downstream/linear_probe/configs/linear_probe.example.yaml
python downstream/linear_probe/train_linear_probe.py --config downstream/linear_probe/configs/linear_probe.example.yaml
python downstream/linear_probe/test_linear_probe.py --config downstream/linear_probe/configs/linear_probe.example.yaml --checkpoint outputs/linear_probe/run/linear_probe_best.pt
```

Feature extraction writes the release cache format:

- `{split}_features.pt`
- `{split}_labels.pt`
- `{split}_samples.json`
- `metadata.json`

See [`docs/downstream_task_guide.md`](../../docs/downstream_task_guide.md) for the release conventions. Do not commit manifests, cached features, checkpoints, generated outputs, or machine-specific paths.
