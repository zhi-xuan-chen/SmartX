# Shared Downstream Modules

This directory contains the SmartX loading helpers used by demos and downstream evaluations.

Main entry points:

- `create_smartx_vlp_model()`: load the full SmartX VLP model from a local checkpoint or `Trusure/SmartX` on Hugging Face.
- `create_clip_style_model_global()`: return projected image/text wrappers plus tokenizer for zero-shot, retrieval, and feature extraction code.

Default official checkpoint:

```text
https://huggingface.co/Trusure/SmartX/blob/main/smartx_large.ckpt
```

Model weights are not tracked in git. Use Hugging Face download or place `smartx_large.ckpt` under `checkpoints/`.
