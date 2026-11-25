# Downstream Tasks

This directory contains the shared [SmartX loaders](modules/README.md) and implementations of all six downstream evaluation workflows. See the [Downstream Task Guide](../docs/downstream_task_guide.md) for full input schemas and end-to-end workflows.

| Directory | Task | Summary |
| :--- | :--- | :--- |
| [`zero_shot_classification/`](zero_shot_classification/README.md) | **Zero-shot classification** | Prompt-based classification without task training |
| [`retrieval/`](retrieval/README.md) | **Image–text retrieval** | Bidirectional radiograph–report retrieval on case manifests |
| [`linear_probe_classification/`](linear_probe_classification/README.md) | **Linear-probe classification** | Frozen-feature extraction, linear training, and evaluation |
| [`examination_level_classification/`](examination_level_classification/README.md) | **Examination-level classification** | Multi-view attention-based MIL classification |
| [`detection/`](detection/README.md) | **Detection** | SmartX-backed ViTDet for COCO-format abnormality localization |
| [`report_generation/`](report_generation/README.md) | **Report generation** | LLaVA-Rad two-stage training and clinical report evaluation |
