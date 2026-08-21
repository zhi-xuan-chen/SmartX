## Vision Encoders from OpenCLIP

This adapter supports timm vision transformers pretrained through OpenCLIP. Install the repository requirements and the report-generation optional dependencies before using it.

Hub-hosted encoders, such as [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224), are selected with `--vision_tower`. For a local encoder, also provide `--vision_tower_config` and `--vision_tower_checkpoint`.

Checkpoint paths must be supplied by the user and must not be committed to this repository.
