# Release Checklist

This file tracks the remaining work before SmartX is ready for public review or submission.

## High Priority

- [x] Make `demo/` runnable.
  - `demo/smartx_demo.ipynb` runs zero-shot classification and image–text retrieval on the
    pretrained weights via `downstream.modules.create_clip_style_model_global()`.
  - Public-domain X-ray demo images under `demo/assets/images/` (see `demo/assets/README.md`).
  - Keep demo outputs outside git-tracked files.

- [x] Add public downstream task entry points.
  - `downstream/zeroshot/` uses SmartX prompt-based zero-shot evaluation.
  - `downstream/retrieval/` uses generic image-text case manifests and Recall@K metrics.
  - `downstream/linear_probe/` uses cached SmartX feature extraction, training, and testing.
  - `downstream/detection/` uses SmartX-backed VitDet with user-provided COCO annotations.
  - All four use `downstream.modules.create_clip_style_model_global()` or the same public SmartX checkpoint resolver.

- [ ] Trim `downstream/mil_linear_probe/` for release clarity.
  - Keep core train/test/data-schema code.
  - Remove or archive internal batch scripts, reader-study scripts, and analysis-output defaults that reviewers cannot run.
  - Do not add private data, private splits, cached features, checkpoints, W&B runs, or generated results.

## Metadata Before Submission

- [x] Fill `CITATION.cff` author and repository fields.
- [x] Fill the public repository in `docs/code_availability.md`.
- [ ] Create a release tag and record it in `docs/code_availability.md`.
- [ ] Archive the release and record its DOI in `docs/code_availability.md`.
- [x] Confirm `docs/model_zoo.md` matches the final Hugging Face files.

## Final Audit

Run these before packaging or submitting:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlp
python -m compileall -q docs demo downstream pretrain tests utils checkpoints
python -m pytest -q
find . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name 'wandb' -o -name 'analysis_results' -o -name 'results' \) -print
find . -type f -size +1M -print
python -m pytest tests/test_downstream_release_tasks.py::test_downstream_release_hygiene -q
rg -n "TBD|NotImplemented|Implementation pending|scaffold" README.md CITATION.cff docs demo downstream pretrain tests utils
```

Expected before release:

- No private paths or credentials.
- No generated results, caches, W&B runs, or checkpoints in git.
- No `TBD` in submission-facing metadata.
- No `NotImplementedError` in reviewer-facing demos.
