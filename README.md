# SmartX

SmartX is the release repository for the X-ray vision-language foundation model described in the manuscript. This repository is organized for review and reuse: model usage demos, TopK pretraining code, downstream evaluation entry points, and Nature-style code/data availability documentation.

## Quick Start

On `jhcpu7`, use the existing `vlp` conda environment for release checks:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlp
python -m pytest -q
```

For a fresh external environment:

```bash
pip install -r requirements.txt
```

Download model checkpoints from `docs/model_zoo.md` and place them under `checkpoints/`.

```bash
python demo/extract_features.py --help
python demo/zero_shot.py --help
python demo/retrieval.py --help
```

The zero-shot, retrieval, linear-probe, and detection implementations are scaffolded here and will be filled from the evaluation repository. TopK pretraining, MIL linear probe, and report-generation code are included.

## Repository Layout

- `docs/`: code availability, data availability, model zoo, model card, reproducibility notes, and remote workflow notes.
- `checkpoints/`: local checkpoint placement only; model weights are not tracked by git.
- `demo/`: minimal reviewer-facing model usage demos and demo assets.
- `pretrain/`: TopK vision-language pretraining code.
- `downstream/`: task entry points and implementation slots for manuscript evaluations.
- `tests/`: lightweight release sanity checks.
