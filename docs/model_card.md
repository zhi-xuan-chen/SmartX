# SmartX Model Card

## Model Details

SmartX is an X-ray vision-language representation model trained with a progressive global-to-local TopK alignment strategy. It learns from native case-level radiograph-report records spanning ten anatomical regions without requiring curated image-level annotations. The SmartX-Large checkpoint uses an ARK ViT-L vision encoder and is available from [Trusure/SmartX](https://huggingface.co/Trusure/SmartX).

## Intended Use

SmartX is intended for research in X-ray representation learning, feature extraction, zero-shot classification, image-text retrieval, linear and case-level MIL probing, detection and report-generation experiments. Dataset-specific training and evaluation remain the responsibility of the user.

## Inputs and Outputs

The shared loader accepts radiographs through the preprocessing defined by each task and produces projected image features compatible with the SmartX text encoder. Multi-view tasks aggregate images at the case level. See the task README files for manifests, image sizes and output formats.

## Not Intended For

- Direct clinical diagnosis, triage or treatment decisions.
- Deployment without external and local validation.
- Patient re-identification or reconstruction of private training data.

## Limitations

Performance may vary under domain shift, scanner and protocol changes, anatomical regions, label definitions and non-radiographic inputs. Demo results are functional checks, not clinical validation. Detection and report generation require additional frameworks, models, datasets and compute that are not included in the lightweight test suite.

## Data and Licences

Private clinical data and evaluation splits are not redistributed. Obtain public datasets and third-party models from their original providers and follow their licences and access terms. See `docs/data_availability.md`, `docs/model_zoo.md` and the Hugging Face model repository for details.
