# Downstream Evaluation

This directory contains shared downstream locations and manuscript evaluation code. See `../docs/downstream_task_guide.md` for release conventions.

- `linear_probe/`: public cached-feature extraction, training, and evaluation for standard linear probes.
- `zeroshot/`: public prompt-based zero-shot classification.
- `retrieval/`: public image-text retrieval on generic case manifests.
- `detection/`: public SmartX-backed VitDet training/evaluation for COCO-format annotations.
- `mil_linear_probe/`: included release code for multi-instance learning screening, with private outputs/data removed.
- `report_generation/`: included release code for LLaVA-Rad report-generation experiments, with predictions, checkpoints, W&B runs, and generated results removed.
