# Zero-Shot Classification

Status: pending implementation. Add this task following [`docs/downstream_task_guide.md`](../../docs/downstream_task_guide.md).

Minimum files expected here:

- `run_zeroshot.py`
- `configs/zeroshot.example.yaml`
- `metrics.py` only if task-specific metrics are needed
- `datasets.py` only if task-specific dataset parsing is needed

Use `downstream.modules.create_clip_style_model_global()` for SmartX checkpoint loading. Do not add private data, checkpoints, generated outputs, W&B runs, or machine-specific paths.
