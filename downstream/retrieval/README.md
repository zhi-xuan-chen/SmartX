# Image–Text Retrieval

Bidirectional radiograph–report retrieval using [`downstream.modules.create_clip_style_model_global()`](../modules/README.md).

## Quick Run

```bash
python downstream/retrieval/run_retrieval.py   --config downstream/retrieval/configs/retrieval.example.yaml
```

## Configuration & Outputs

| Item | Details |
| :--- | :--- |
| **Input Manifest** | Case-level JSONL/CSV containing one report/caption and one or more radiograph paths per case |
| **Metrics** | Bidirectional image-to-text and text-to-image `recall@K`, `mrr`, `mean_rank`, and `median_rank` |
| **Outputs** | `similarity.npy`, `summary.json`, `summary.csv`, generated manifest copy, and optional per-case ranks |

See [`docs/downstream_task_guide.md`](../../docs/downstream_task_guide.md) for full input formats and task workflows.
