#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import LlavaLlamaForCausalLM, LlavaMPTForCausalLM
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.language_model.llava_llama import LlavaConfig
from llava.model.multimodal_projector.builder import build_vision_projector


def _resolve_local_weight_file(model_path, filename):
    """
    Find `filename` next to `model_path` or in ancestor directories.

    HuggingFace Trainer writes `checkpoint-*` subfolders; custom training saves
    `non_lora_trainables.bin` only at the run output root, so loading a LoRA
    checkpoint must fall back to parent dirs. Never treat absolute paths as HF
    repo ids (newer huggingface_hub validates repo_id and rejects filesystem paths).
    """
    if not model_path:
        return None
    root = model_path
    if os.path.isfile(root):
        root = os.path.dirname(root)
    if not os.path.isdir(root):
        return None

    visited = set()
    cur = os.path.abspath(root)
    for _ in range(24):
        if cur in visited:
            break
        visited.add(cur)
        candidate = os.path.join(cur, filename)
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def _find_dir_with_file(start_path: str, basename: str) -> Optional[str]:
    """
    HF Trainer saves adapters under checkpoint-*; those folders often lack config.json.
    Walk upward until we find a directory containing `basename` (e.g. config.json).
    """
    if not start_path:
        return None
    root = start_path
    if os.path.isfile(root):
        root = os.path.dirname(root)
    if not os.path.isdir(root):
        return None
    visited = set()
    cur = os.path.abspath(root)
    for _ in range(24):
        if cur in visited:
            break
        visited.add(cur)
        if os.path.isfile(os.path.join(cur, basename)):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def _looks_like_local_filesystem_path(model_path):
    """True if we must not pass model_path to hf_hub_download as repo_id."""
    if not model_path:
        return False
    if os.path.isabs(model_path):
        return True
    if os.path.exists(model_path):
        return True
    if model_path.startswith(("./", ".\\")) or model_path.startswith(".."):
        return True
    return False


def _load_weight_file(model_path, filename):
    resolved = _resolve_local_weight_file(model_path, filename)
    if resolved is not None:
        return torch.load(resolved, map_location="cpu")

    if _looks_like_local_filesystem_path(model_path):
        raise FileNotFoundError(
            f"Could not find '{filename}' under '{model_path}' or any parent directory. "
            "For LoRA checkpoints under checkpoint-*, ensure non_lora_trainables.bin exists "
            "at the training output root (same directory that contains checkpoint-*)."
        )

    from huggingface_hub import hf_hub_download

    cache_file = hf_hub_download(repo_id=model_path, filename=filename)
    return torch.load(cache_file, map_location="cpu")


def _normalize_non_lora_state_dict_keys(state_dict):
    state_dict = {(k[11:] if k.startswith('base_model.') else k): v for k, v in state_dict.items()}
    if any(k.startswith('model.model.') for k in state_dict):
        state_dict = {(k[6:] if k.startswith('model.') else k): v for k, v in state_dict.items()}
    return state_dict


def _infer_mm_hidden_size_from_state_dict(state_dict):
    for suffix in ('mm_projector.0.weight', 'mm_projector.weight'):
        for key, value in state_dict.items():
            if key.endswith(suffix) and hasattr(value, "ndim") and value.ndim == 2:
                return value.shape[1]
    return None


def _infer_mm_hidden_size_from_model(model):
    mm_projector = model.get_model().mm_projector
    if isinstance(mm_projector, torch.nn.Linear):
        # Use actual weight shape rather than in_features to work around
        # accelerate dispatch re-materializing weights with checkpoint shapes.
        return mm_projector.weight.shape[1] if mm_projector.weight.device.type != "meta" else mm_projector.in_features
    if isinstance(mm_projector, torch.nn.Sequential):
        for module in mm_projector:
            if isinstance(module, torch.nn.Linear):
                if module.weight.device.type != "meta":
                    return module.weight.shape[1]
                return module.in_features
    return None


def _align_mm_projector_if_needed(model, expected_mm_hidden_size):
    if expected_mm_hidden_size is None:
        return

    model_mm_hidden_size = _infer_mm_hidden_size_from_model(model)
    if model_mm_hidden_size == expected_mm_hidden_size:
        return

    print(
        f"Warning: runtime mm_projector input dim={model_mm_hidden_size} mismatches "
        f"checkpoint mm_projector input dim={expected_mm_hidden_size}. "
        "Rebuilding mm_projector to match checkpoint."
    )
    model.config.mm_hidden_size = expected_mm_hidden_size

    old_projector = model.get_model().mm_projector
    projector_device = None
    projector_dtype = None
    for param in old_projector.parameters():
        projector_device = param.device
        projector_dtype = param.dtype
        break

    rebuilt_projector = build_vision_projector(model.config)
    if projector_device is not None:
        rebuilt_projector = rebuilt_projector.to(device=projector_device, dtype=projector_dtype)
    model.get_model().mm_projector = rebuilt_projector


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
    kwargs = {"device_map": device_map}
    has_lora_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    if not has_lora_adapter and not os.path.isdir(model_path):
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id=model_path, filename="adapter_config.json")
            has_lora_adapter = True
        except Exception:
            pass
    is_lora_checkpoint = ('lora' in model_name.lower()) or has_lora_adapter

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    
    if 'llava' in model_name.lower():
        # Load LLaVA model
        if is_lora_checkpoint and model_base is None:
            warnings.warn('Detected LoRA adapter checkpoint but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if is_lora_checkpoint and model_base is not None:
            non_lora_trainables = _load_weight_file(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = _normalize_non_lora_state_dict_keys(non_lora_trainables)

            # Prefer local LlavaConfig to avoid picking Transformers built-in llava config.
            # checkpoint-* often has no config.json; use stage output root or base.
            cfg_source_dir = _find_dir_with_file(model_path, "config.json") or model_base
            lora_cfg_pretrained = LlavaConfig.from_pretrained(cfg_source_dir)
            inferred_mm_hidden_size = _infer_mm_hidden_size_from_state_dict(non_lora_trainables)
            if inferred_mm_hidden_size is not None:
                cfg_mm_hidden_size = getattr(lora_cfg_pretrained, "mm_hidden_size", None)
                if cfg_mm_hidden_size != inferred_mm_hidden_size:
                    print(
                        f"Warning: config mm_hidden_size={cfg_mm_hidden_size} mismatches "
                        f"checkpoint mm_projector input dim={inferred_mm_hidden_size}. "
                        "Overriding config.mm_hidden_size for loading."
                    )
                    lora_cfg_pretrained.mm_hidden_size = inferred_mm_hidden_size

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            # Sync vocab_size with model_base so lm_head and embed_tokens load without
            # shape mismatch. The embeddings will be properly extended later via
            # resize_token_embeddings() after special tokens are added to the tokenizer.
            base_config = AutoConfig.from_pretrained(model_base, trust_remote_code=True)
            base_vocab_size = getattr(base_config, "vocab_size", None)
            if base_vocab_size is not None and getattr(lora_cfg_pretrained, "vocab_size", None) != base_vocab_size:
                print(
                    f"Warning: lora config vocab_size={lora_cfg_pretrained.vocab_size} != "
                    f"model_base vocab_size={base_vocab_size}. "
                    "Syncing to model_base to avoid lm_head shape mismatch during loading."
                )
                lora_cfg_pretrained.vocab_size = base_vocab_size
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            _align_mm_projector_if_needed(model, inferred_mm_hidden_size)
            print('Loading additional LLaVA weights...')
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            mm_projector_weights = None
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_src = _find_dir_with_file(model_path, "config.json") or model_base
                cfg_pretrained = AutoConfig.from_pretrained(cfg_src, trust_remote_code=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                # Prefer local LlavaConfig to preserve custom vision tower settings.
                cfg_src = _find_dir_with_file(model_path, "config.json") or model_base
                cfg_pretrained = LlavaConfig.from_pretrained(cfg_src)
                mm_projector_weights = None
                if os.path.exists(os.path.join(model_path, 'mm_projector.bin')):
                    mm_projector_weights = _load_weight_file(model_path, 'mm_projector.bin')
                elif os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                    mm_projector_weights = _load_weight_file(model_path, 'non_lora_trainables.bin')
                    mm_projector_weights = _normalize_non_lora_state_dict_keys(mm_projector_weights)

                inferred_mm_hidden_size = _infer_mm_hidden_size_from_state_dict(mm_projector_weights or {})
                if inferred_mm_hidden_size is not None:
                    cfg_mm_hidden_size = getattr(cfg_pretrained, "mm_hidden_size", None)
                    if cfg_mm_hidden_size != inferred_mm_hidden_size:
                        print(
                            f"Warning: config mm_hidden_size={cfg_mm_hidden_size} mismatches "
                            f"checkpoint mm_projector input dim={inferred_mm_hidden_size}. "
                            "Overriding config.mm_hidden_size for loading."
                        )
                        cfg_pretrained.mm_hidden_size = inferred_mm_hidden_size
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                _align_mm_projector_if_needed(model, inferred_mm_hidden_size)

            if mm_projector_weights is None:
                mm_projector_weights = _load_weight_file(model_path, 'mm_projector.bin')
            
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None
    
    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
