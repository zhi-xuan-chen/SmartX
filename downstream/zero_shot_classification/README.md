# Zero-Shot Classification

Prompt-based abnormality recognition using frozen SmartX image and text encoders via [`downstream.modules.create_clip_style_model_global()`](../modules/README.md).

## Quick Run

```bash
python downstream/zero_shot_classification/run_zero_shot_classification.py   --config downstream/zero_shot_classification/configs/zero_shot_classification.example.yaml
```

## Configuration & Outputs

| Item | Details |
| :--- | :--- |
| **Input Manifest** | CSV/JSONL with an image column plus either a multi-label column or one column per class |
| **Configuration** | Set paths, class names, positive/negative prompt templates, batch size, and output directory in [`configs/zero_shot_classification.example.yaml`](configs/zero_shot_classification.example.yaml) |
| **Decision Rule** | Binary and multi-label predictions use a fixed zero threshold on positive-minus-negative prompt similarity (no test-set threshold tuning) |
| **Outputs** | Summary metrics and per-sample prediction scores in the configured output directory |

See [`docs/downstream_task_guide.md`](../../docs/downstream_task_guide.md) for full input formats and task workflows.
