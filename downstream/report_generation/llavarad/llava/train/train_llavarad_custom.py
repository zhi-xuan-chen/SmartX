import os
import copy
import glob
import logging
import pathlib
import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import transformers

from llava.constants import (
    IGNORE_INDEX, IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
)
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer
from llava import conversation as conversation_lib
from llava.model import LlavaLlamaForCausalLM, LlavaMPTForCausalLM
from llava.mm_utils import tokenizer_image_token, open_image_with_retry
from llava.utils import data_loaders
from llava.train.argument import ModelArguments, DataArguments, TrainingArguments

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# ── ZeRO-3 helpers ───────────────────────────────────────────────────────────

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    else:
        raise NotImplementedError
    return {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    return {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key in k for key in keys_to_match)}
    return {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    lora_module_names.discard("lm_head")
    return list(lora_module_names)


# ── Tokenizer / embedding resize ─────────────────────────────────────────────

def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_emb = model.get_input_embeddings().weight.data
        output_emb = model.get_output_embeddings().weight.data
        input_emb[-num_new_tokens:] = input_emb[:-num_new_tokens].mean(0, keepdim=True)
        output_emb[-num_new_tokens:] = output_emb[:-num_new_tokens].mean(0, keepdim=True)


# ── Tokenization / label masking (mirrors train.py) ──────────────────────────

def _tokenize_fn(strings, tokenizer):
    tokenized_list = [
        tokenizer(text, return_tensors="pt", padding="longest",
                  max_length=tokenizer.model_max_length, truncation=True)
        for text in strings
    ]
    input_ids = [t.input_ids[0] for t in tokenized_list]
    input_ids_lens = [t.input_ids.ne(tokenizer.pad_token_id).sum().item() for t in tokenized_list]
    return dict(input_ids=input_ids, labels=input_ids,
                input_ids_lens=input_ids_lens, labels_lens=input_ids_lens)


def _mask_targets(target, tokenized_lens, speakers):
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(sources, data_args):
    if not data_args.is_multimodal:
        return sources
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    return sources


def preprocess_v1(sources, tokenizer, has_image=False):
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(p, tokenizer, return_tensors="pt") for p in conversations], dim=0
        )
    else:
        input_ids = tokenizer(conversations, return_tensors="pt", padding="longest",
                              max_length=tokenizer.model_max_length, truncation=True).input_ids
    targets = input_ids.clone()
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for rou in rounds:
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length and cur_len != total_len:
            target[:] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=targets)


def preprocess_plain(sources, tokenizer):
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = (source[0]["value"] + source[1]["value"]
                        + conversation_lib.default_conversation.sep)
        conversations.append(conversation)
    input_ids = [tokenizer_image_token(p, tokenizer, return_tensors="pt") for p in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=targets)


def preprocess(sources, tokenizer, has_image=False):
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        from llava.train.train import preprocess_llama_2
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    # Default: add speaker/signal
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    if has_image:
        input_ids = [tokenizer_image_token(p, tokenizer, return_tensors="pt") for p in conversations]
    else:
        input_ids = _tokenize_fn(conversations, tokenizer)["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        header = f"{conversation_lib.default_conversation.system}\n\n"
        if has_image:
            tokenized_lens = [len(tokenizer_image_token(header, tokenizer))] + \
                             [len(tokenizer_image_token(s["value"], tokenizer)) for s in source]
        else:
            tokenized_lens = _tokenize_fn(
                [header] + [s["value"] for s in source], tokenizer
            )["input_ids_lens"]
        _mask_targets(target, tokenized_lens, [s["from"] for s in source])
    return dict(input_ids=input_ids, labels=targets)


# ── Dataset ───────────────────────────────────────────────────────────────────

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path, tokenizer, data_args):
        super().__init__()
        self.list_data_dict = data_loaders[data_args.loader](data_path)
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        return [
            sum(len(c["value"].split()) for c in s["conversations"]) + (128 if "image" in s else 0)
            for s in self.list_data_dict
        ]

    @property
    def modality_lengths(self):
        result = []
        for s in self.list_data_dict:
            cur = sum(len(c["value"].split()) for c in s["conversations"])
            result.append(cur if "image" in s else -cur)
        return result

    def __getitem__(self, i):
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1

        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            processor = self.data_args.image_processor
            image = open_image_with_retry(
                os.path.join(self.data_args.image_folder, image_file)
            )
            if image is None:
                logging.error("Use an empty image.")
                image = Image.new("RGB", (224, 224),
                                  tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args
            )
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(sources, self.tokenizer,
                               has_image=("image" in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple(
            [inst[k] for inst in instances] for k in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        if "image" in instances[0]:
            images = [inst["image"] for inst in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images
        return batch


def make_supervised_data_module(tokenizer, data_args):
    train_dataset = LazySupervisedDataset(
        data_path=data_args.data_path, tokenizer=tokenizer, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    assert model_args.custom_vision_encoder is not None, \
        "--custom_vision_encoder must be set (e.g. smartx_large, rad_dino)"

    # ── Load LLaVA-Rad merged base ────────────────────────────────────────────
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    # HF >=4.37 validates generation_config during save_pretrained and can raise
    # when do_sample=False while temperature/top_p keep sampling values.
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        do_sample = getattr(gen_cfg, "do_sample", None)
        temperature = getattr(gen_cfg, "temperature", None)
        top_p = getattr(gen_cfg, "top_p", None)
        if do_sample is False:
            if temperature is not None and temperature != 1.0:
                gen_cfg.temperature = 1.0
            if top_p is not None and top_p != 1.0:
                gen_cfg.top_p = 1.0

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # ── Initialize custom vision tower + new projector ────────────────────────
    model_args.vision_tower = model_args.custom_vision_encoder
    model.get_model().initialize_vision_modules(
        model_args=model_args, fsdp=training_args.fsdp
    )
    # Persist to config so inference can reload without extra CLI flags
    model.config.custom_vision_pretrained_path = model_args.custom_vision_pretrained_path

    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=compute_dtype, device=training_args.device)

    data_args.image_processor = vision_tower.image_processor  # CustomImageProcessor
    data_args.is_multimodal = True
    data_args.mm_use_im_start_end = model_args.mm_use_im_start_end

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # ── Stage 2: optionally load Stage 1 projector weights ───────────────────
    if training_args.stage == "stage2":
        if training_args.stage1_output_dir:
            proj_bin = os.path.join(training_args.stage1_output_dir, "non_lora_trainables.bin")
            if not os.path.exists(proj_bin):
                raise FileNotFoundError(
                    f"Stage 2: no non_lora_trainables.bin found in {training_args.stage1_output_dir}"
                )
            rank0_print(f"Stage 2: loading projector weights from {proj_bin}")
            proj_weights = torch.load(proj_bin, map_location="cpu")
            # Strip base_model/model. prefixes that deepspeed/PEFT may add
            proj_weights = {
                (k[11:] if k.startswith("base_model.") else k): v
                for k, v in proj_weights.items()
            }
            if any(k.startswith("model.model.") for k in proj_weights):
                proj_weights = {(k[6:] if k.startswith("model.") else k): v
                                for k, v in proj_weights.items()}
            model.load_state_dict(proj_weights, strict=False)
            rank0_print("Stage 2: projector weights loaded OK")
        else:
            rank0_print("Stage 2: --stage1_output_dir is empty, skip projector loading and train from initialized mm_projector")

    # ── Stage 2: apply LoRA to LLM ───────────────────────────────────────────
    if training_args.stage == "stage2" and training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type=TaskType.CAUSAL_LM,
        )
        if training_args.bf16:
            model.to(torch.bfloat16)
        elif training_args.fp16:
            model.to(torch.float16)
        # Explicitly freeze base model before wrapping with PEFT, so that
        # get_peft_model() only leaves LoRA adapter weights trainable.
        model.requires_grad_(False)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    # ── Freeze / unfreeze by stage ────────────────────────────────────────────
    if training_args.stage == "stage1":
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        rank0_print("Stage 1: only mm_projector is trainable")

    elif training_args.stage == "stage2":
        # LoRA params already trainable. Keep projector trainable too.
        try:
            projector = model.base_model.model.get_model().mm_projector
        except AttributeError:
            projector = model.get_model().mm_projector
        for p in projector.parameters():
            p.requires_grad = True
        rank0_print("Stage 2: LoRA + mm_projector trainable")

    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    rank0_print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Data ──────────────────────────────────────────────────────────────────
    # use_reentrant=False avoids "marked ready twice" DDP errors with LoRA
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = LLaVATrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    model.config.use_cache = True

    # ── Save weights ──────────────────────────────────────────────────────────
    if training_args.lora_enable and training_args.stage == "stage2":
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank in (0, -1):
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict,
                       os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
    else:
        # Stage 1: save mm_projector weights
        keys_to_match = ["mm_projector"]
        weight_to_save = get_mm_adapter_state_maybe_zero_3(
            model.named_parameters(), keys_to_match
        )
        model.config.save_pretrained(training_args.output_dir)
        if training_args.local_rank in (0, -1):
            current_folder = training_args.output_dir.rstrip("/").split("/")[-1]
            parent_folder = os.path.dirname(training_args.output_dir)
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save,
                           os.path.join(mm_projector_folder, f"{current_folder}.bin"))
            else:
                torch.save(weight_to_save,
                           os.path.join(training_args.output_dir, "non_lora_trainables.bin"))


if __name__ == "__main__":
    train()
