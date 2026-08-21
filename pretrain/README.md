# Pretraining

This directory contains the TopK vision-language pretraining implementation used for SmartX.

From the repository root, set the shared dataset root and run:

```bash
export SMARTX_DATA_ROOT=/path/to/datasets
python pretrain/run.py --config pretrain/configs/topk_ark_base.example.yaml --smoke_test
```

The example expects CheXpert-Plus, PadChest, MIMIC-CXR and ReXGradient-160K
under `SMARTX_DATA_ROOT`. Edit or clear individual dataset paths when using a
different combination. Set `SMARTX_OUTPUT_DIR` to override the default
