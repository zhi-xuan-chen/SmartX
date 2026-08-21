# Demo

This directory contains small runnable demos that do not require full benchmark datasets.

## Notebook

- `smartx_demo.ipynb`: end-to-end **zero-shot classification** and **image–text retrieval**
  on a handful of public-domain chest X-rays (`assets/images/`), using the pretrained
  weights only — no training. Requires a local checkpoint at
  `checkpoints/smartx_large.ckpt` (set `CHECKPOINT = None` in the notebook to auto-download
  from Hugging Face `Trusure/SmartX`). Runs on CPU or GPU.

For full-dataset evaluation, see `downstream/zeroshot/` and `downstream/retrieval/`.
