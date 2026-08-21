# Detection

This folder provides a SmartX-backed VitDet workflow adapted from the evaluation repository. It preserves the Detectron2/LazyConfig training structure while requiring users to provide COCO annotations, image roots, checkpoint path, and output directory in YAML.

```bash
python downstream/detection/train_detection.py train --config downstream/detection/configs/detection.example.yaml
python downstream/detection/train_detection.py test --config downstream/detection/configs/detection.example.yaml --checkpoint outputs/detection/xray_detection/model_best.pth
```

The detector uses the SmartX Swin-L backbone. If `model.checkpoint` is empty, the backbone resolves the SmartX checkpoint through [`downstream.modules.create_clip_style_model_global()`](../modules/README.md) and the shared Hugging Face defaults.

## Optional Dependencies

Detection dependencies are intentionally not forced into the base install. Install the helpers validated in the `vlp` environment, then install Detectron2 (which also provides `fvcore`) for your CUDA/PyTorch platform:

```bash
pip install -r downstream/detection/requirements-optional.txt
```

Follow the official Detectron2 installation instructions for a version compatible with your CUDA/PyTorch combination. Detectron2 is not pinned because it is not installed in the validated `vlp` environment.

See [`docs/downstream_task_guide.md`](../../docs/downstream_task_guide.md) for the release conventions. Do not commit COCO annotations, dataset images, generated outputs, checkpoints, or machine-specific paths.
