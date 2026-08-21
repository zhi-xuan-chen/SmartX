# SmartX

<p align="center">
  <img src="docs/assets/smartx_icon_white.png" alt="SmartX" width="190">
</p>

<p align="center">
  <strong>A clinically evaluated vision-language foundation model for generalizable multi-region X-ray interpretation</strong>
</p>

<p align="center">
  <a href="https://github.com/zhi-xuan-chen/SmartX/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/zhi-xuan-chen/SmartX/actions/workflows/tests.yml/badge.svg"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <a href="https://github.com/zhi-xuan-chen/SmartX/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="https://huggingface.co/Trusure/SmartX"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97-Model_Weights-yellow"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#demo">Demo</a> ·
  <a href="#downstream-tasks">Downstream tasks</a> ·
  <a href="#documentation">Documentation</a>
</p>

SmartX learns from case-level radiograph-report pairs across ten anatomical
regions, matching the native structure of radiology archives without requiring
curated image-level annotations. This repository provides model loading,
pretraining, reviewer-facing demos, and public implementations of six downstream
evaluation tasks.

## Quick start

SmartX requires Python 3.10 or newer.

```bash
git clone https://github.com/zhi-xuan-chen/SmartX.git
cd SmartX
conda create -n smartx python=3.10 -y
conda activate smartx
python -m pip install -r requirements.txt
```

### Model weights

| Model | Vision backbone | Checkpoint |
| --- | --- | --- |
| SmartX-Large | ARK ViT-L | [`smartx_large.ckpt`](https://huggingface.co/Trusure/SmartX/resolve/main/smartx_large.ckpt) |

The shared loader downloads the checkpoint from
[`Trusure/SmartX`](https://huggingface.co/Trusure/SmartX) automatically when
`checkpoint` is omitted. To use a local copy, place it at
`checkpoints/smartx_large.ckpt`.

```python
from downstream.modules import create_clip_style_model_global

vision_model, text_model, tokenizer, model_info = create_clip_style_model_global({
    "model_name": "smartx_large",
    # "checkpoint": "checkpoints/smartx_large.ckpt",  # optional
})
```

To load the full VLP module:

```python
from downstream.modules import create_smartx_vlp_model

model, tokenizer, load_report = create_smartx_vlp_model(
    "checkpoints/smartx_large.ckpt"
)
```

See the [model zoo](docs/model_zoo.md) and [model card](docs/model_card.md) for
model details, intended use, and limitations.

## Demo

The [SmartX demo notebook](demo/smartx_demo.ipynb) runs zero-shot classification
and image-text retrieval on public-domain X-rays using pretrained weights only.
It supports CPU and GPU execution.

```bash
python -m pip install notebook
jupyter notebook demo/smartx_demo.ipynb
```

Place the checkpoint at `checkpoints/smartx_large.ckpt`, or set
`CHECKPOINT = None` in the notebook to download it automatically.

## Downstream tasks

| Task | Public implementation |
| --- | --- |
| Zero-shot classification | [Prompt-based evaluation](downstream/zeroshot/README.md) |
| Image-text retrieval | [Embedding extraction and retrieval metrics](downstream/retrieval/README.md) |
| Linear probe | [Feature extraction, training, and evaluation](downstream/linear_probe/README.md) |
| MIL linear probe | [Study-level multiple-instance learning](downstream/mil_linear_probe/README.md) |
| Object detection | [SmartX-backed VitDet with COCO-format annotations](downstream/detection/README.md) |
| Report generation | [LLaVA-Rad training and evaluation](downstream/report_generation/README.md) |

All tasks use the [shared SmartX loaders](downstream/modules/README.md).
Dataset-agnostic interfaces and release conventions are documented in the
[downstream task guide](docs/downstream_task_guide.md).

## Pretraining

TopK pretraining code and an example configuration are available under
[`pretrain/`](pretrain/).

```bash
python pretrain/run.py --config pretrain/configs/topk_ark_base.example.yaml
```

The example configuration contains placeholder paths. Replace them with local
public or approved data paths; never commit private data paths.

## Tests

Run the lightweight release checks:

```bash
python -m pytest -q
```

These checks cover repository structure, public configuration templates, and
lightweight utilities. Full training and benchmark evaluation require their
respective datasets and optional dependencies.

## Documentation

- [Model zoo](docs/model_zoo.md)
- [Model card](docs/model_card.md)
- [Data availability](docs/data_availability.md)
- [Code availability](docs/code_availability.md)
- [Reproducibility guide](docs/reproducibility.md)
- [Downstream task guide](docs/downstream_task_guide.md)
- [Release checklist](docs/release_checklist.md)

Release tags and an archival DOI will be added to the code-availability record
when a formal release is created.

## Citation

If you use SmartX, please cite the associated manuscript. Machine-readable
citation metadata and the complete author list are provided in
[`CITATION.cff`](CITATION.cff).

## License

The code is released under the [MIT License](LICENSE). Third-party models and
datasets remain subject to their respective licenses and terms of use.
