# MIL Linear Probe

This directory contains the SmartX multi-instance learning screening code.

Included:
- feature extraction and MIL train/test entry points
- dataset loaders and MIL trainer code
- threshold/statistical-analysis utilities
- runnable config templates with release placeholders

Not included:
- private clinical data, private splits, reader-study images, generated results, features, checkpoints, or W&B runs
- internal reader-study and manuscript-table helper scripts

Before running, replace `${SMARTX_DATA_ROOT}`, `${SMARTX_OUTPUT_ROOT}`, and checkpoint placeholders with local paths that you are allowed to use.
