# Image-Text Retrieval

This folder provides public SmartX image-text retrieval adapted from the evaluation repository. It keeps the upstream square-similarity ranking metrics and replaces private dataset defaults with a generic case manifest.

```bash
python downstream/retrieval/run_retrieval.py --config downstream/retrieval/configs/retrieval.example.yaml
```

The manifest must contain one report/caption text per case and one or more image paths. The runner writes a generated manifest copy, `similarity.npy`, `summary.json`, `summary.csv`, and optional per-case ranks under the configured output directory.

Metrics include image-to-text and text-to-image `recall@K`, `mrr`, `mean_rank`, and `median_rank`.

See [`docs/downstream_task_guide.md`](../../docs/downstream_task_guide.md) for the release conventions. The model is loaded through [`downstream.modules.create_clip_style_model_global()`](../modules/README.md). Do not commit private reports, manifests, generated results, cached embeddings, checkpoints, or machine-specific paths.
