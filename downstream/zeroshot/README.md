# Zero-Shot Classification

This folder provides SmartX zero-shot classification adapted from the evaluation repository. It uses positive and negative text prompts per class, encodes images and prompts with [`downstream.modules.create_clip_style_model_global()`](../modules/README.md), and writes metrics plus per-sample scores to a user-selected output directory.

```bash
python downstream/zeroshot/run_zeroshot.py --config downstream/zeroshot/configs/zeroshot.example.yaml
```

The manifest should contain an image column and either a label column or one label column per class. Paths, class names, prompts, split, batch size, checkpoint, and output directory are all configured in YAML.

See [`docs/downstream_task_guide.md`](../../docs/downstream_task_guide.md) for the release conventions. Do not commit dataset manifests, labels, generated predictions, cached features, checkpoints, or machine-specific paths.
