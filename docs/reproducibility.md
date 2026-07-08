# Reproducibility

This file records the command structure for reproducing manuscript experiments after datasets and checkpoints are prepared locally.

## Release Test Environment

On `jhcpu7`, run repository checks in the existing `vlp` conda environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlp
cd ~/Workspace/SmartX
python -m pytest -q
```

The current checked environment is `~/miniconda3/envs/vlp/bin/python` with Python 3.10.18 and pytest 7.2.0.

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

## Downstream Evaluations

Downstream task directories provide the expected structure. Zero-shot, retrieval, linear-probe, and detection code will be added from the evaluation repository.
