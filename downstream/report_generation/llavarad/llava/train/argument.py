from dataclasses import dataclass, field
from typing import Optional
import transformers


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to LLaVA-Rad merged checkpoint (Vicuna + LLaVA-Rad LoRA merged)."}
    )
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    # Vision tower — set to custom_vision_encoder value before initialize_vision_modules()
    vision_tower: Optional[str] = field(default=None)
    vision_tower_config: Optional[str] = field(default=None)
    vision_tower_checkpoint: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    # Custom encoder fields
    custom_vision_encoder: Optional[str] = field(
        default=None,
        metadata={"help": "smartx_large; separate benchmark encoders are supported"}
    )
    custom_vision_pretrained_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local vision checkpoint path; required for smartx_large when calling this entry point directly."}
    )


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to training JSON."})
    loader: str = field(default="default")
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = field(default="square")
    image_grid_pinpoints: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    model_max_length: int = field(default=2048)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)
    # LoRA
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    # Two-stage
    stage: str = field(
        default="stage1",
        metadata={"help": "stage1: train projector only. stage2: projector + LLM LoRA."}
    )
    stage1_output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Stage 1 output_dir; Stage 2 auto-loads latest checkpoint's non_lora_trainables.bin."}
    )
    save_non_lora_each_checkpoint: bool = field(
        default=True,
        metadata={
            "help": "Stage2+LoRA: save non_lora_trainables.bin inside each checkpoint-* folder "
            "(mm_projector aligned with that step). Disable to only keep root non_lora after training."
        },
    )
    group_by_modality_length: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
