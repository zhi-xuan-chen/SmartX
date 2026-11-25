# SmartX Model Card

**SmartX** is an open vision–language foundation model for real-world multi-anatomy X-ray interpretation, pretrained on **2.7 million radiographs** with paired clinical reports across **10 anatomical regions**. Using progressive global-to-local alignment, it learns transferable vision–language representations directly from native clinical records without requiring extra curated image-level annotations.

## Model Summary

| Attribute | Details |
| :--- | :--- |
| **Model Name** | SmartX (`smartx_large`) |
| **Vision Encoder** | Swin-L (768×768 input resolution) |
| **Text Encoder** | PubMedBERT |
| **Pretraining Corpus** | 2.7 million radiographs with paired reports spanning 10 anatomical regions |
| **Checkpoint** | [`Trusure/SmartX`](https://huggingface.co/Trusure/SmartX) (`smartx_large.ckpt`) |
| **Code License** | MIT License |

## Intended Use

SmartX is intended for research in radiographic representation learning and downstream medical imaging workflows, including:

- Feature extraction and representation analysis
- Zero-shot abnormality classification and bidirectional image–text retrieval
- Linear-probe and multi-view examination-level classification
- Abnormality detection and clinical report generation

Dataset-specific adaptation, threshold selection, and clinical validation remain the responsibility of the user.

## Out-of-Scope Use

- Direct clinical diagnosis, triage, or treatment decisions without prospective clinical validation and regulatory approval
- Unvalidated deployment across new hospitals, scanners, or patient populations
- Patient re-identification or reconstruction of private training records

## Limitations

- Performance may vary under domain shift, scanner or protocol differences, anatomical coverage, label definitions, and non-radiographic inputs.
- Bundled demo examples serve as functional verification only and do not constitute clinical validation.
- Detection and report generation require additional third-party frameworks, weights, datasets, and GPU compute beyond the lightweight test suite.

## Data and Governance

Private clinical cohorts and evaluation splits are not redistributed. Public benchmark datasets and third-party models must be obtained from their original providers under their respective licenses and data-use terms. See [Data Availability](data_availability.md) and [Model Zoo](model_zoo.md) for details.
