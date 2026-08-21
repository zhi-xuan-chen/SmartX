# Reproducibility

This file records the command structure for reproducing manuscript experiments after datasets and checkpoints are prepared locally.

## Release Tests

From the repository root in a Python 3.10 environment:

```bash
python -m pytest -q
```

## TopK Pretraining

```bash
cd pretrain
python run.py --config configs/topk_ark_base.example.yaml --smoke_test
```

## Demos

```bash
jupyter notebook demo/smartx_demo.ipynb
```

Runs zero-shot classification and image–text retrieval on public-domain demo X-rays using
the pretrained weights only (no training). Place the checkpoint at
`checkpoints/smartx_large.ckpt`, or set `CHECKPOINT = None` in the notebook to auto-download
from Hugging Face `Trusure/SmartX`.

Release smoke validation with the official checkpoint completed all eight code
cells, loaded without missing or shape-skipped parameters, and achieved
image-to-text Recall@1 of 4/4 on the bundled examples.

## Downstream Evaluations

Runnable entry points are provided for:

- zero-shot classification: `downstream/zeroshot/`
- image--text retrieval: `downstream/retrieval/`
- image-level linear probing: `downstream/linear_probe/`
- COCO-format detection: `downstream/detection/`
- case-level MIL linear probing: `downstream/mil_linear_probe/`
- MIMIC-CXR and IU X-Ray report generation: `downstream/report_generation/`

Each directory contains its data contract, example configuration, and exact
commands. The MIL workflow runs feature extraction, training, then testing:

```bash
python downstream/mil_linear_probe/extract_features.py --config downstream/mil_linear_probe/configs/mil_linear_probe.example.yaml
python downstream/mil_linear_probe/train_mil_linear_probe.py --config downstream/mil_linear_probe/configs/mil_linear_probe.example.yaml
python downstream/mil_linear_probe/test_mil_linear_probe.py --config downstream/mil_linear_probe/configs/mil_linear_probe.example.yaml --checkpoint outputs/mil_linear_probe/run/mil_linear_probe_best.pt
```

Report generation exposes the complete two-stage training and evaluation
workflow for both datasets:

```bash
bash downstream/report_generation/llavarad/scripts/run_mimic_smartx.sh all
bash downstream/report_generation/llavarad/scripts/run_iu_smartx.sh all
```

Detection and report generation require their task-specific optional
dependencies. Dataset access, pretrained language models, and compute are not
included; see the corresponding task README before running.
