# Detection

This folder provides a SmartX-backed VitDet workflow using Detectron2
LazyConfig. The detector combines the SmartX Swin-L encoder with an FPN and
the VitDet Mask R-CNN configuration (with its mask branch disabled) for
COCO-format object detection.

## Installation

Install the repository dependencies, the detection helpers, and the validated
Detectron2 source revision:

```bash
python -m pip install -r requirements.txt
python -m pip install -r downstream/detection/requirements-optional.txt
python -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/detectron2.git@a2f4a8771ab77e8411c26b27f24f9489a28a2453"
```

This revision was validated with the versions pinned in the root
`requirements.txt`. The tagged Detectron2 0.6 release is insufficient because
it does not contain `common/models/mask_rcnn_vitdet.py` or
`detectron2.modeling.backbone.vit`.

## Data and checkpoint

Set the data root and, optionally, a local SmartX checkpoint:

```bash
export SMARTX_DATA_ROOT=/path/to/data-root
export SMARTX_CHECKPOINT=/path/to/smartx_large.ckpt
```

The example configuration expects this layout:

```text
${SMARTX_DATA_ROOT}/detection/
├── annotations_train.json
├── annotations_val.json
├── annotations_test.json
├── images_train/
├── images_val/
└── images_test/
```

Annotations must use COCO format. Update `dataset.num_classes` and
`dataset.class_names` in the YAML to match the categories in the annotation
files. If `model.checkpoint` is empty, the shared SmartX loader downloads the
checkpoint from Hugging Face through
[`downstream.modules.create_clip_style_model_global()`](../modules/README.md).

## Run

```bash
python downstream/detection/train_detection.py train --config downstream/detection/configs/detection.example.yaml
python downstream/detection/train_detection.py test --config downstream/detection/configs/detection.example.yaml --checkpoint outputs/detection/xray_detection/model_best.pth
```

See [`docs/downstream_task_guide.md`](../../docs/downstream_task_guide.md) for input formats and task workflows.
