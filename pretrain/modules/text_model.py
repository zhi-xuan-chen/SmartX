from transformers import AutoModel, AutoTokenizer, AutoConfig
try:
    # Newer transformers with SigLIP2 support
    from transformers import Siglip2TextModel
except Exception:
    Siglip2TextModel = None


TEXT_ENCODERS = {
    "clinical-bert": "emilyalsentzer/Bio_ClinicalBERT",
    "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "bert-base-uncased": "bert-base-uncased",
    "bert-base-cased": "bert-base-cased",
    # SigLIP2 text tower (shares checkpoint with the vision tower)
    "siglip2-so400m-naflex": "google/siglip2-so400m-patch16-naflex",
    "qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-embedding-4b": "Qwen/Qwen3-Embedding-4B",
    "qwen3-embedding-8b": "Qwen/Qwen3-Embedding-8B",
}


def create_text_encoder(text_encoder_name, load_pretrained=True):
    repo_path = TEXT_ENCODERS[text_encoder_name]
    # Dedicated handling for SigLIP2 text encoder
    if text_encoder_name.startswith("siglip2-"):
        if Siglip2TextModel is None:
            raise ImportError(
                "The installed transformers version does not support Siglip2TextModel. "
                "Please upgrade transformers to >= 4.49 to use the SigLIP2 text encoder."
            )
        model = Siglip2TextModel.from_pretrained(
            repo_path,
            trust_remote_code=True,
            revision="main",
            output_hidden_states=True,
            output_attentions=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            repo_path, trust_remote_code=True, revision="main"
        )
        return model, tokenizer

    # Default to standard AutoModel/AutoTokenizer path
    if load_pretrained:
        model = AutoModel.from_pretrained(
            repo_path,
            trust_remote_code=True,
            revision="main",
            output_hidden_states=True,
            output_attentions=True,
        )
    else:
        config = AutoConfig.from_pretrained(repo_path, output_hidden_states=True,
                                           output_attentions=True)
        model = AutoModel.from_config(config)
    
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        
    tokenizer = AutoTokenizer.from_pretrained(
        repo_path, trust_remote_code=True, revision="main"
    )
    limit = getattr(model.config, "max_position_embeddings", None)
    if isinstance(limit, int) and limit > 0:
        tokenizer.model_max_length = min(tokenizer.model_max_length, limit)
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = create_text_encoder(text_encoder_name="pubmedbert")
    print(model)
    print(tokenizer)
    print(model.config)
