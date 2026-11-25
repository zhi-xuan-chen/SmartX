# Code Availability

This repository provides the public SmartX implementation for model loading, pretraining, runnable demos, and downstream evaluation.

## Repository Metadata

| Field | Value |
| :--- | :--- |
| **Public Repository** | [`https://github.com/zhi-xuan-chen/SmartX`](https://github.com/zhi-xuan-chen/SmartX) |
| **Model Weights** | [`https://huggingface.co/Trusure/SmartX`](https://huggingface.co/Trusure/SmartX) |
| **License** | MIT License |
| **Release Tag / DOI** | Pending archival release |

## Included Components

| Directory | Contents |
| :--- | :--- |
| [`pretrain/`](../pretrain/README.md) | Vision–language pretraining pipeline and configuration template |
| [`downstream/modules/`](../downstream/modules/README.md) | Shared SmartX model loaders and image preprocessing utilities |
| [`downstream/`](../downstream/README.md) | All six downstream task workflows ([Downstream Task Guide](downstream_task_guide.md)) |
| [`demo/`](../demo/README.md) | Runnable zero-shot classification and image–text retrieval notebook |
| `tests/` | Lightweight verification suite (`python -m pytest -q`) |

## Excluded Artifacts

- Private clinical radiographs, reports, labels, and institutional evaluation splits
- Model checkpoint binaries (hosted separately on Hugging Face; see [Model Zoo](model_zoo.md))
- Training logs, cached features, and generated predictions
