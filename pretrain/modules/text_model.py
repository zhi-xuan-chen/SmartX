from transformers import AutoModel, AutoTokenizer
try:
    # 新版 transformers（支持 SigLIP2）
    from transformers import Siglip2TextModel
except Exception:
    Siglip2TextModel = None


TEXT_ENCODERS = {
    "clinical-bert": "emilyalsentzer/Bio_ClinicalBERT",
    "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "bert-base-uncased": "bert-base-uncased",
    "bert-base-cased": "bert-base-cased",
    # SigLIP2 文本塔（与视觉塔共享同一个 ckpt）
    "siglip2-so400m-naflex": "google/siglip2-so400m-patch16-naflex",
    "qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-embedding-4b": "Qwen/Qwen3-Embedding-4B",
    "qwen3-embedding-8b": "Qwen/Qwen3-Embedding-8B",
}


def create_text_encoder(text_encoder_name):
    repo_path = TEXT_ENCODERS[text_encoder_name]
    # 专门适配 SigLIP2 文本 encoder
    if text_encoder_name.startswith("siglip2-"):
        if Siglip2TextModel is None:
            raise ImportError(
                "当前 transformers 版本不支持 Siglip2TextModel，"
                "请升级 transformers 到 >= 4.49（或对应该模型的版本）后再使用 SigLIP2 文本 encoder。"
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

    # 默认走原有 AutoModel/AutoTokenizer 路径
    model = AutoModel.from_pretrained(
        repo_path,
        trust_remote_code=True,
        revision="main",
        output_hidden_states=True,
        output_attentions=True,
    )
    
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        
    tokenizer = AutoTokenizer.from_pretrained(
        repo_path, trust_remote_code=True, revision="main"
    )
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = create_text_encoder(text_encoder_name="pubmedbert")
    print(model)
    print(tokenizer)
    print(model.config)
