# Shared Downstream Loaders

Unified model-loading and radiograph-preprocessing utilities shared across demos and all six downstream workflows.

## Entry Points

| Function | Purpose |
| :--- | :--- |
| `create_smartx_vlp_model()` | Load the full SmartX vision–language model from a local checkpoint or [`Trusure/SmartX`](https://huggingface.co/Trusure/SmartX) |
| `create_clip_style_model_global()` | Return projected image/text encoders plus tokenizer for zero-shot classification, retrieval, and feature extraction |

## Checkpoint Resolution

Model weights are not tracked in git. Leave `checkpoint` unset to download `smartx_large.ckpt` automatically from [`Trusure/SmartX`](https://huggingface.co/Trusure/SmartX), or place a local copy at `checkpoints/smartx_large.ckpt`.
