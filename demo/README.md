# Demo

Runnable examples using pretrained SmartX weights without full benchmark datasets.

## Notebook

- **[`smartx_demo.ipynb`](smartx_demo.ipynb)** — end-to-end **zero-shot classification** and **image–text retrieval** on public-domain chest X-rays (`assets/images/`). Supports both CPU and GPU execution.

## Quick Run

```bash
python -m pip install notebook
jupyter notebook demo/smartx_demo.ipynb
```

## Checkpoint Resolution

| Source | Behavior |
| :--- | :--- |
| `SMARTX_CHECKPOINT` env var | Uses the specified local checkpoint path |
| `checkpoints/smartx_large.ckpt` | Automatically used when present locally |
| Default fallback | Automatically downloads `smartx_large.ckpt` from Hugging Face [`Trusure/SmartX`](https://huggingface.co/Trusure/SmartX) |

> On the first run, the tokenizer and text encoder configuration are fetched automatically (no duplicate model weights are downloaded). For full-dataset evaluation, see [`downstream/zero_shot_classification/`](../downstream/zero_shot_classification/README.md) and [`downstream/retrieval/`](../downstream/retrieval/README.md).
