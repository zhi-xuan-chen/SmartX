# Model Zoo

Model files are not tracked in git. Download them from the release links below and place them under `checkpoints/`.

| Model | Backbone | File | Link | Used for |
| --- | --- | --- | --- | --- |
| SmartX-Large | ARK ViT-L | `smartx_large.ckpt` | https://huggingface.co/Trusure/SmartX | Main experiments and demos |

## Local Layout

```text
checkpoints/
└── smartx_large.ckpt
```

## Loading

Use `downstream.modules.create_clip_style_model_global()` or `downstream.modules.create_smartx_vlp_model()` to load the public checkpoint from Hugging Face or a local `checkpoints/smartx_large.ckpt` file.
