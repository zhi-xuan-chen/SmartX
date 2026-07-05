# Demo

This directory is for small reviewer-facing demos that run without full benchmark datasets.

Planned entry points:

- `extract_features.py`: load a checkpoint and export image embeddings for demo images.
- `zero_shot.py`: run prompt-based zero-shot classification on demo X-rays.
- `retrieval.py`: run image-text retrieval on demo image/report pairs.

The zero-shot and retrieval implementations depend on downstream code that will be added from the evaluation repository.
