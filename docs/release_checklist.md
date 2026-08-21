# Release Checklist

This file tracks the remaining work before SmartX is ready for public review or submission.

## High Priority

- [x] Make `demo/` runnable.
  - `demo/smartx_demo.ipynb` runs zero-shot classification and image–text retrieval on the
    pretrained weights via `downstream.modules.create_clip_style_model_global()`.
  - Public-domain X-ray demo images under `demo/assets/images/` (see `demo/assets/README.md`).
  - Keep demo outputs outside git-tracked files.
  - Release smoke test completed all eight code cells with the official checkpoint;
    image-to-text Recall@1 was 4/4 on the bundled examples.

- [x] Add downstream task entry points.
  - `downstream/zeroshot/` uses SmartX prompt-based zero-shot evaluation.
  - `downstream/retrieval/` uses generic image-text case manifests and Recall@K metrics.
  - `downstream/linear_probe/` uses cached SmartX feature extraction, training, and testing.
  - `downstream/detection/` uses SmartX-backed VitDet with user-provided COCO annotations.
  - `downstream/mil_linear_probe/` uses generic case manifests and a CLAM-style attention head.
  - `downstream/report_generation/` provides LLaVA-Rad-based two-stage training and
    evaluation for MIMIC-CXR and IU X-Ray.
  - All six use the SmartX checkpoint resolver.

- [x] Trim `downstream/mil_linear_probe/` for release clarity.
  - Keep only the generic manifest, feature extraction, CLAM train/test, config, and documentation.
  - Removed internal cohort adapters, batch scripts, reader-study helpers, and analysis code.
  - Do not add private data, private splits, cached features, checkpoints, W&B runs, or generated results.

- [x] Run the pytest suite on pushes to `main` and pull requests with a
  lightweight GitHub Actions workflow.

## Metadata Before Submission

- [x] Fill `CITATION.cff` author and repository fields.
- [x] Fill the public repository in `docs/code_availability.md`.
- [ ] Create a release tag and record it in `docs/code_availability.md`.
- [ ] Archive the release and record its DOI in `docs/code_availability.md`.
- [x] Confirm `docs/model_zoo.md` matches the final Hugging Face files.

## Final Audit

Run these before packaging or submitting:

```bash
python -m compileall -q docs demo downstream pretrain tests utils checkpoints
python -m pytest -q
find . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name 'wandb' -o -name 'analysis_results' -o -name 'results' \) -print
find . -type f -size +1M -print
python -m pytest tests/test_downstream_release_tasks.py::test_downstream_release_hygiene -q
rg -n "TBD|Implementation pending|scaffold" README.md CITATION.cff docs --glob '!release_checklist.md'
rg -n "NotImplemented(Error)?" demo
```

Expected before release:

- No private paths or credentials.
- No generated results, caches, W&B runs, or checkpoints in git.
- No `TBD` in submission-facing metadata.
- No `NotImplementedError` in demos.
