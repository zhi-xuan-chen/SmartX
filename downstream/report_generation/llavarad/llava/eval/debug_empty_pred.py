"""
Debug script: check why model generates empty predictions.
Run: python -m llava.eval.debug_empty_pred
"""
import os, sys, torch
from PIL import Image

os.makedirs("logs", exist_ok=True)

CKPT  = "/scratch/medimgfmod/zchenhi/checkpoints/XRay-VLP/experiments/RG/mimic_llavarad/llavarad_ark_base_stage2"
BASE  = "${SMARTX_OUTPUT_ROOT}/llavarad_merged"
MIMIC = os.environ.get("MIMIC_ROOT", "${SMARTX_DATA_ROOT}/MIMIC-CXR")
QUERY = f"{MIMIC}/labels/mimic_cxr_test_llava_rad_frontal_2461.csv"
IMG_DIR = f"{MIMIC}/imgs-1024"

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import data_loaders

model_name = get_model_name_from_path(CKPT)
print(f"model_name: {model_name}")

tokenizer, model, image_processor, context_len = load_pretrained_model(
    CKPT, BASE, model_name, device="cuda"
)

# ---------- 1. Check mm_projector weights ----------
proj = model.get_model().mm_projector
for i, m in enumerate(proj):
    if hasattr(m, 'weight'):
        w = m.weight
        print(f"mm_projector[{i}].weight: shape={w.shape} dtype={w.dtype} "
              f"device={w.device} norm={w.float().norm():.4f} "
              f"zeros={(w==0).sum().item()}/{w.numel()}")

# ---------- 2. Check vision tower output ----------
queries = data_loaders["mimic_test_findings"](QUERY)
q0 = queries[0]
img_path = os.path.join(IMG_DIR, q0["image"])
img = Image.open(img_path).convert("RGB")
img_tensor = image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
print(f"image_tensor: shape={img_tensor.shape} dtype={img_tensor.dtype} "
      f"min={img_tensor.min():.4f} max={img_tensor.max():.4f}")

with torch.inference_mode():
    vt = model.get_vision_tower()
    feats = vt(img_tensor.unsqueeze(0).half().cuda())
    print(f"vision_tower output: shape={feats.shape} norm={feats.float().norm():.4f} "
          f"zeros={(feats==0).sum().item()}/{feats.numel()}")
    projected = model.get_model().mm_projector(feats)
    print(f"mm_projector output: shape={projected.shape} norm={projected.float().norm():.4f}")

# ---------- 3. Check lm_head and embed_tokens for NaN ----------
lm_head = model.lm_head
emb     = model.get_model().embed_tokens
lm_w = lm_head.weight
emb_w = emb.weight
print(f"lm_head.weight:      shape={lm_w.shape} dtype={lm_w.dtype} "
      f"nan={torch.isnan(lm_w.float()).sum().item()} inf={torch.isinf(lm_w.float()).sum().item()} "
      f"norm={lm_w.float().norm():.4f}")
print(f"embed_tokens.weight: shape={emb_w.shape} dtype={emb_w.dtype} "
      f"nan={torch.isnan(emb_w.float()).sum().item()} inf={torch.isinf(emb_w.float()).sum().item()} "
      f"norm={emb_w.float().norm():.4f}")

# ---------- 4. Text-only forward to check if LLM itself works ----------
text_only = "Hello, the patient has"
text_ids = tokenizer(text_only, return_tensors="pt").input_ids.cuda()
with torch.inference_mode():
    text_out = model.generate(text_ids, do_sample=False, max_new_tokens=20).cpu()
print(f"text-only decode: {repr(tokenizer.decode(text_out[0, text_ids.shape[1]:], skip_special_tokens=True))}")

# ---------- 5. One forward pass to check logit NaN ----------
text2 = DEFAULT_IMAGE_TOKEN + "\nProvide a description of the findings in the radiology image."
conv = conv_templates["v1"].copy()
conv.append_message(conv.roles[0], text2)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
print(f"input_ids: shape={input_ids.shape}, len={len(input_ids)}")

with torch.inference_mode():
    fwd = model(
        input_ids=input_ids.unsqueeze(0).cuda(),
        images=img_tensor.unsqueeze(0).half().cuda(),
    )
    logits = fwd.logits  # [1, seq_len, vocab_size]
    last_logits = logits[0, -1]  # logits for the next token prediction
    print(f"logits[-1]: shape={last_logits.shape} "
          f"nan={torch.isnan(last_logits.float()).sum().item()} "
          f"inf={torch.isinf(last_logits.float()).sum().item()} "
          f"min={last_logits.float().min():.4f} max={last_logits.float().max():.4f}")
    top5 = last_logits.float().topk(5)
    print(f"top5 token ids: {top5.indices.tolist()}")
    print(f"top5 tokens: {[tokenizer.decode([i]) for i in top5.indices.tolist()]}")

# ---------- 6. generate() raw output ----------
with torch.inference_mode():
    output_ids = model.generate(
        input_ids.unsqueeze(0).cuda(),
        images=img_tensor.unsqueeze(0).half().cuda(),
        do_sample=False, temperature=0, max_new_tokens=64, use_cache=True
    ).cpu()

new_tokens = output_ids[0, len(input_ids):]
print(f"generate() new tokens: {new_tokens.tolist()[:20]}...")
print(f"prediction: {repr(tokenizer.decode(new_tokens, skip_special_tokens=True))}")
