# Model Zoo

Model checkpoints are hosted on Hugging Face and are not tracked in git.

## Released Weights

| Model | Vision Backbone | Config Name | Filename | Repository | Use Cases |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SmartX** | Swin-L | `smartx_large` | `smartx_large.ckpt` | [`Trusure/SmartX`](https://huggingface.co/Trusure/SmartX) | Pretraining initialization, downstream tasks, and demos |

## Local Checkpoint Layout

To use a local copy instead of automatic Hugging Face download, place the checkpoint under `checkpoints/`:

```text
checkpoints/
└── smartx_large.ckpt
```

## Loading

Use the shared loaders in [`downstream/modules/`](../downstream/modules/README.md):

- `downstream.modules.create_clip_style_model_global()` — projected image and text encoders plus tokenizer
- `downstream.modules.create_smartx_vlp_model()` — full vision–language pretraining module
