# Code Availability

This repository contains the SmartX release code for model usage, TopK pretraining, and downstream evaluation interfaces.

## Repository

- Public repository: https://github.com/zhi-xuan-chen/SmartX
- Release tag: Not yet available
- Archived DOI: Not yet available

## Included Code

- TopK vision-language pretraining implementation in `pretrain/`.
- Shared model wrappers in `downstream/modules/`.
- Entry points for linear probe, zero-shot classification, image--text retrieval, and detection in `downstream/`.
- Included MIL linear-probe and LLaVA-Rad report-generation code in `downstream/`.
- A runnable zero-shot and image--text retrieval notebook in `demo/`.

## Not Included

- Private clinical data, private splits, patient images, reports, or reader-study material.
- Model checkpoints; see `docs/model_zoo.md`.
- Generated results, W&B runs, checkpoints, intermediate features, and manuscript working files.
