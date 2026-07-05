# SmartX Model Card

## Model Details

SmartX is an X-ray vision-language representation model trained with the TopK objective described in the manuscript.

## Intended Use

- X-ray image representation learning.
- Image-text retrieval.
- Zero-shot classification.
- Feature extraction for downstream classifiers.

## Not Intended For

- Direct clinical diagnosis.
- Deployment without local validation.
- Patient-level inference or private data reconstruction.

## Limitations

Performance may vary under domain shift, scanner/protocol changes, label definitions, and non-radiographic imaging distributions. Private training/evaluation data are not redistributed.
