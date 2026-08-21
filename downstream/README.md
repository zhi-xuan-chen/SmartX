# Downstream Evaluation

This directory contains shared modules and downstream evaluation code. See `../docs/downstream_task_guide.md` for release conventions.

- `linear_probe/`: cached-feature extraction, training, and evaluation for standard linear probes.
- `zeroshot/`: prompt-based zero-shot classification.
- `retrieval/`: image-text retrieval on generic case manifests.
- `detection/`: SmartX-backed VitDet training/evaluation for COCO-format annotations.
- `mil_linear_probe/`: case-level multi-instance learning screening.
- `report_generation/`: LLaVA-Rad-based report generation for MIMIC-CXR and IU X-Ray.
