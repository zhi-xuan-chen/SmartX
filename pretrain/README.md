# Pretraining

Continue pretraining from the released SmartX weights on your own radiograph–report corpus using [`configs/smartx_large.example.yaml`](configs/smartx_large.example.yaml).

## Quick Smoke Test

From the repository root, configure dataset paths and run a 1-epoch verification check:

```bash
export SMARTX_DATA_ROOT=/path/to/datasets
# Optional local checkpoint (downloads from Trusure/SmartX automatically if omitted):
# export SMARTX_CHECKPOINT=/path/to/smartx_large.ckpt

python pretrain/run.py --config pretrain/configs/smartx_large.example.yaml   --smoke_test --batch_size 1 --smoke_test_max_epochs 1
```

## Distributed Training

Launch one process per GPU with `torchrun` (for multi-node runs, also set `SLURM_NNODES` and cluster rendezvous variables):

```bash
torchrun --standalone --nproc_per_node=8 pretrain/run.py   --config pretrain/configs/smartx_large.example.yaml
```

## Resuming Training

Resume an interrupted run from its Lightning checkpoint (restores model weights, optimizer state, and training progress without re-downloading initialization weights):

```bash
python pretrain/run.py --config pretrain/configs/smartx_large.example.yaml   --resume_from /path/to/run/checkpoints/last.ckpt
```

## Configuration Notes

| Setting | Behavior |
| :--- | :--- |
| **Initialization** | Loads full SmartX weights with strict architecture compatibility; new runs start with a fresh optimizer and LR schedule |
| **Precision** | Defaults to `16-mixed`; set `precision: bf16-mixed` in YAML on BF16-capable GPUs |
| **Datasets** | Template expects `CheXpert-Plus`, `PadChest`, `MIMIC-CXR`, and `ReXGradient-160K` under `SMARTX_DATA_ROOT`; edit or clear entries for your local data |
| **Outputs & Logging** | Writes to `SMARTX_OUTPUT_DIR` (default: `experiments/`); W&B logging is disabled by default (`wandb_mode: disabled`) |
