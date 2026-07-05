## Vision Encoders from Open CLIP

### Requirements
```
pip install torch==2.0.1 torchvision==0.15.2 open-clip-torch==2.23.0 timm==0.9.12
```

### Supported Vision Encoders
Right now, we only support [Timm Vision Transformer](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py), **which has been pretrained by [open_clip](https://github.com/mlfoundations/open_clip)**. However, it has already included a lot of good vision encoders. Among them, there are two types of open_clip vision encoders we'd like to try in LLaVA:
- **Public vision encoders**: they are all available on the 🤗 HuggingFace Hub, e.g., [microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224), [timm/ViT-B-16-SigLIP-512](https://huggingface.co/timm/ViT-B-16-SigLIP-512). For these models, we only need their model name to load them, e.g., timm/ViT-B-16-SigLIP-512. To use them in LLaVA pretraining, we need to specify `--vision_tower` (e.g., 'hf-hub:timm/ViT-B-16-SigLIP-512') and `--vision_tower_name` (any name is ok; this is just used to track different experiments). We also need to remove `--vision_tower_config` and `--vision_tower_checkpoint` from the command.
- **Local checkpoint vision encoders**: for encoders that are not available directly from a public Hub identifier, provide both a config file and a checkpoint path via `--vision_tower_config` and `--vision_tower_checkpoint`. Use public or reviewer-accessible checkpoint locations only.
    ```json
    {
        "embed_dim": 512,
        "vision_cfg": {
            "timm_model_name": "vit_base_patch16_384",
            "timm_model_pretrained": false,
            "timm_pool": "",
            "timm_proj": "linear",
            "image_size": 384
        },
        "text_cfg": {
            "hf_model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            "hf_tokenizer_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            "proj": "mlp",
            "pooler_type": "cls_pooler",
            "context_length": 256
        }
    }
    ```

### Test

Before we use a new vision encoder in LLaVA, it is important to test that we can correctly load it and its basic functionality is working.
To do so, we run `test.py`, where we test its basic functionality such as zero-shot classficiation, cross-modal retrieval.
```
python test.py
```
By default, it tests BiomedCLIP's zero-shot classfication. You can uncomment or add other encoders to test other fuctionality or encoders. 

