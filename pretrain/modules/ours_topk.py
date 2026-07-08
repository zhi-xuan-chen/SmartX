from einops import rearrange
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pos_embed import get_2d_sincos_pos_embed
from .text_model import create_text_encoder
from .vision_model import create_vision_encoder
try:  # works when repo root is on sys.path (downstream/demo)
    from pretrain.utils import print_rank_zero
except ModuleNotFoundError:  # works when pretrain/ is the import root (training)
    from utils import print_rank_zero

class XRayTopKVLP(torch.nn.Module):
    def __init__(
        self,
        vision_encoder,
        text_encoder,
        tokenizer,
        text_freeze_layers=0,
        output_dim=512,
        pretrained_weight_path=None,
        pretrained_load="vision_encoder",
        vision_encoder_name="vit_base_patch16_224",
        text_feature_mode="pooling",  # "pooling" or "reencode"
        max_text_length=64,  # Max length for single region text
        max_text_length_reencode=256,  # Max length for concatenated text in reencode mode
        accumulate_ratio=0.8,  # Ratio for cumulative sentence selection; rho>=1.0 means no selection.
        topk_min_k=1,  # Lower bound on K to guarantee at least 1 sentence is selected.
    ):
        super(XRayTopKVLP, self).__init__()
        self.vision_encoder_name = vision_encoder_name
        self.local_loss = False
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.text_feature_mode = text_feature_mode
        self.max_text_length = max_text_length
        self.max_text_length_reencode = max_text_length_reencode
        self.accumulate_ratio = float(accumulate_ratio)
        self.topk_min_k = int(max(1, topk_min_k))
        print_rank_zero(f"text_feature_mode: {text_feature_mode}")
        if self.vision_encoder_name == "ark_base":
            self.vision_embed_dim = 1024
        elif self.vision_encoder_name == "ark_large":
            self.vision_embed_dim = 1536
        else:
            try:
                self.vision_embed_dim = vision_encoder.embed_dim  # 768
            except:
                self.vision_embed_dim = 768
        # 兼容 HF SigLIP（wrapper 提供了 patch_embed shim）
        # self.vision_patch_size = int(getattr(getattr(vision_encoder, 'patch_embed', None), 'patch_size', (16,))[0])
        self.vision_patch_size = 14
        # self.vision_in_chans = vision_encoder.in_chans
        self.vision_in_chans = 3 # vision_encoder.in_chans
        # self.num_patches = int(getattr(getattr(vision_encoder, 'patch_embed', None), 'num_patches', 0)) or getattr(getattr(vision_encoder, 'patch_embed', None), 'num_patches', 0)
        self.num_patches = 1369
        if not self.num_patches:
            # 回退：根据 img_size 与 patch_size 估计（仅用于初始化位置编码等，不影响 HF 路径）
            self.num_patches = (self.vision_encoder.patch_embed.num_patches if hasattr(self.vision_encoder, 'patch_embed') and hasattr(self.vision_encoder.patch_embed, 'num_patches') else 0)
        self.text_embed_dim = self.text_encoder.config.hidden_size  # 768
        self.text_freeze_layers = text_freeze_layers
        self.proj_out = output_dim  # 512

        # Use vision_embed_dim for image projection since we're not using fusion
        self.proj_img = nn.Linear(self.vision_embed_dim, self.proj_out)
        self.proj_t = nn.Linear(self.text_embed_dim, self.proj_out)

        # Learnable temperature and bias parameter for contrastive learning
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))
        self.logit_bias = nn.Parameter(torch.ones([]) * -10.0)


        self.device_id = None
        self._device_cache = None  # Cache device to avoid repeated lookups
        if self.training:
            self.text_encoder.train()

        # Freeze text encoder layers
        if self.text_freeze_layers == -1:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        else:
            for layer_idx in range(self.text_freeze_layers):
                for param in list(
                    self.text_encoder.encoder.layer[layer_idx].parameters()
                ):
                    param.requires_grad = False
        # Initialize weights - 对 HF SigLIP 视觉编码器跳过视觉随机初始化
        skip_vision = (
            (pretrained_weight_path is not None and pretrained_load in ["vision_encoder", "vision", "encoder_only"]) or
            getattr(self.vision_encoder, 'is_hf_pretrained', False)
        )
        self.initialize_weights(skip_vision_encoder=True)

    def _get_device(self):
        """Get cached device to avoid repeated parameter lookups"""
        if self._device_cache is None:
            self._device_cache = next(self.parameters()).device
        return self._device_cache

    def _tokenize(self, text, device=None, max_length=None):
        """
        Tokenize text with configurable max_length.
        
        Args:
            text: Text to tokenize
            device: Device to move tensors to
            max_length: Maximum sequence length. If None, uses self.max_text_length
        """
        if max_length is None:
            max_length = self.max_text_length
        
        if self.training:
            tokenizer_output = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
        else:
            tokenizer_output = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )

        if device is not None:
            tokenizer_output = {k: v.to(device, non_blocking=True) for k, v in tokenizer_output.items()}
        return tokenizer_output

    def _simple_encode_text(self, text, max_length=None):
        """
        Encode text using simple encoding.
        
        Args:
            text: Text to encode
            max_length: Maximum sequence length. If None, uses self.max_text_length_reencode for reencode mode,
                       or self.max_text_length for normal mode
        """
        # Ensure text is a list of strings for batch_encode_plus
        if not isinstance(text, (list, tuple)):
            text = [text]  # Convert single string to list
        
        # Filter out any non-string items and convert to strings
        text = [str(item) if item is not None else "" for item in text]
        device = self._get_device()
        
        # Use max_text_length_reencode if in reencode mode and max_length not specified
        if max_length is None and self.text_feature_mode == "reencode":
            max_length = self.max_text_length_reencode
        
        tokenized_output = self._tokenize(text, device=device, max_length=max_length)
        input_ids = tokenized_output["input_ids"]
        attention_mask = tokenized_output["attention_mask"]
        if self.training:
            self.text_encoder.train()
        output = self.text_encoder(
            input_ids=input_ids.to(device, non_blocking=True), 
            attention_mask=attention_mask.to(device, non_blocking=True)
        )
        # Mean Pooling
        last = output.last_hidden_state  # [B, T, d]
        mask_f = attention_mask.unsqueeze(-1).float()
        # text_emb = (last * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1e-6)

        # CLS token
        text_emb = output.pooler_output

        return text_emb, last, mask_f

    def encode_text(self, text):
        """Encode text regions.

        Args:
            text: List of [batch_size, max_regions] where each element is a text string

        Returns:
            region_emb: [B, N_r, D] D: text embedding dimension, region embeddings only
        """
        batch_size = len(text)
        device = self._get_device()

        if self.training:
            self.text_encoder.train()

        # Determine max_regions from text structure
        max_regions = max(len(region_texts) for region_texts in text) if text else 0
        
        # Flatten all text regions for batch encoding
        flat_texts = []
        for region_text in text:
            region_text = region_text if isinstance(region_text, (list, tuple)) else []
            for i in range(max_regions):
                if i < len(region_text):
                    flat_texts.append(region_text[i] if region_text[i] else "")  # Handle None/empty
                else:
                    flat_texts.append("")  # padding

        # Tokenize and encode all texts at once
        toks = self._tokenize(flat_texts, device=device)

        out = self.text_encoder(**toks)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            flat_emb = out.pooler_output  # [B*max_regions, D]
        else:
            # Use [CLS] token (position 0) from last hidden state
            flat_emb = out.last_hidden_state[:, 0, :]  # [B*max_regions, D]

        # Reshape to [B, max_regions, D]
        D = flat_emb.shape[-1]
        region_emb = flat_emb.view(batch_size, max_regions, D)

        return region_emb

    def encode_image(self, img):
        """
        Encode image.

        Args:
            img: [B, V, C, H, W] where V = number of views per case
        Returns:
            cls_tkn: [B, V, D] encoded cls token for each X-ray
            patch_tokens: [B, V, N, D] encoded patch tokens for each X-ray
        """
        B, V, C, H, W = img.shape
        # Reshape to process all images at once using einops
        images_flat = rearrange(img, "b v c h w -> (b v) c h w")

        with torch.no_grad() if not self.training else torch.enable_grad():
            # cls_tkn, patch_tokens 
            if self.vision_encoder_name == "vit_base_patch16_224":
                x = self.vision_encoder.forward_features(
                    images_flat,
                    attn_mask=None,
                )
                cls_tkn = x[:, 0, :]
                patch_tokens = x[:, 1:, :]
            else:
                cls_tkn, patch_tokens = self.vision_encoder.forward_features(
                    images_flat,
                    attn_mask=None,
                )
        cls_tkn = rearrange(cls_tkn, "(b v) d -> b v d", b=B, v=V)
        patch_tokens = rearrange(patch_tokens, "(b v) n d -> b v n d", b=B, v=V)
        return cls_tkn, patch_tokens

    def encode_image_rad_dino(self, img):
        B, V, C, H, W = img.shape
        images_flat = rearrange(img, "b v c h w -> (b v) c h w")
        # Prepare inputs using RadDINO's processor; keep autograd enabled
        if self.training:
            self.vision_encoder.model.train()
        else:
            self.vision_encoder.model.eval()
        inputs = self.vision_encoder.processor(images_flat, return_tensors="pt")
        inputs = {k: v.to(self._get_device(), non_blocking=True) for k, v in inputs.items()}
        outputs = self.vision_encoder.model(**inputs)
        last_hidden = outputs.last_hidden_state  # [(B*V), 1+N, D]
        cls_tkn = last_hidden[:, 0, :]
        patch_tokens = last_hidden[:, 1:, :]  # [(B*V), N, D]
        cls_tkn = rearrange(cls_tkn, "(b v) d -> b v d", b=B, v=V)
        patch_tokens = rearrange(patch_tokens, "(b v) n d -> b v n d", b=B, v=V)
        return cls_tkn, patch_tokens

    def encode_image_ark(self, img):
        B, V, C, H, W = img.shape
        images_flat = rearrange(img, "b v c h w -> (b v) c h w")
        with torch.no_grad() if not self.training else torch.enable_grad():
            cls_tkn = self.vision_encoder.forward_features(
                images_flat
            )
        cls_tkn = rearrange(cls_tkn, "(b v) d -> b v d", b=B, v=V)
        return cls_tkn, None
     
    def warmup_forward(self, img, text, img_mask=None, text_mask=None):
        # single_image = img[:, 0, :, :, :].unsqueeze(1)
        # Automatically select encoding method based on vision encoder type
        if self.vision_encoder_name == "rad-dino":
            img_cls_tkn, img_patch_tokens = self.encode_image_rad_dino(img)
        elif self.vision_encoder_name in ["ark_base", "ark_large"]:
            img_cls_tkn, img_patch_tokens = self.encode_image_ark(img)
        else:
            img_cls_tkn, img_patch_tokens = self.encode_image(img)
        # img_cls_tkn = img_cls_tkn.squeeze(1)

        # Encode each sentence separately, then pool per sample
        text_region_tkn = self.encode_text(text)  # [B, N_r, D] - region tokens only
        
        # case_img_features = torch.mean(img_cls_tkn, dim=1)
        img_cls_features = self.proj_img(img_cls_tkn)
        text_region_features = self.proj_t(text_region_tkn)
        
        img_cls_features = F.normalize(img_cls_features, dim=-1)
        text_region_features = F.normalize(text_region_features, dim=-1)
        
        # only pooling the valid image and text regions according to the mask
        B = img_cls_features.shape[0]
        device = img_cls_features.device
        
        # Pool image features with mask: [B, V, D] -> [B, D]
        if img_mask is not None:
            # img_mask: [B, V], True=valid, False=padding
            valid_img_mask = img_mask.float()  # [B, V]
            img_cls_features_masked = img_cls_features * valid_img_mask.unsqueeze(-1)  # [B, V, D]
            denom_img = valid_img_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [B, 1]
            case_img_features = img_cls_features_masked.sum(dim=1) / denom_img  # [B, D]
        else:
            case_img_features = img_cls_features.mean(dim=1)  # [B, D]
        
        # Pool text region features with mask: [B, N_r, D] -> [B, D]
        if text_mask is not None:
            # text_mask: [B, N_r], True=valid, False=padding
            valid_text_mask = text_mask.float()  # [B, N_r]
            text_region_features_masked = text_region_features * valid_text_mask.unsqueeze(-1)  # [B, N_r, D]
            denom_text = valid_text_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [B, 1]
            text_features = text_region_features_masked.sum(dim=1) / denom_text  # [B, D]
        else:
            text_features = text_region_features.mean(dim=1)  # [B, D]
        
        logit_scale = self.logit_scale.exp()
        logit_scale = logit_scale.clamp(min=1e-6, max=100)
        logit_bias = self.logit_bias.clamp(min=-20.0, max=5.0)
        
        return {
            "img_cls_tkn": case_img_features,
            "img_patch_tokens": img_patch_tokens,
            "text_embeddings": text_features,
            "logit_scale": logit_scale,
            "logit_bias": logit_bias,
        }

    def forward(self, img, text, img_mask=None, text_mask=None):
        """
        Args:
            img: [B, V (=max_views), C, H, W]
            text: List of [batch_size, max_regions] where each element is a text string
            img_mask: torch.Tensor [B, V] where True=valid image, False=padding
            text_mask: torch.Tensor [B, max_regions] where True=real text, False=padding

        Returns:
            dict: Dictionary containing the following keys:
        """
        # Encode image and text
        # Automatically select encoding method based on vision encoder type
        if self.vision_encoder_name == "rad-dino":
            img_cls_tkn, img_patch_tokens = self.encode_image_rad_dino(img)
        elif self.vision_encoder_name in ["ark_base", "ark_large"]:
            img_cls_tkn, img_patch_tokens = self.encode_image_ark(img)
        else:
            img_cls_tkn, img_patch_tokens = self.encode_image(img)
        # [B, V, D], [B, V, N, D]

        text_region_tkn = self.encode_text(text)  # [B, N_r, D] - region tokens only

        # Project to common embedding space for similarity computation
        img_cls_features = self.proj_img(img_cls_tkn)  # [B, V, proj_out (512)]
        text_region_features = self.proj_t(text_region_tkn)  # [B, N_r, proj_out (512)]

        # Normalize features for similarity computation
        img_cls_features = F.normalize(img_cls_features, dim=-1) # [B, V, proj_out (512)]
        text_region_features = F.normalize(text_region_features, dim=-1) # [B, N_r, proj_out (512)]

        # Randomly select an image, then compute soft weights for text regions based on similarity
        B, V = img_cls_tkn.shape[:2]
        N_r = text_region_features.shape[1]
        
        # Get learnable temperature for similarity computation
        logit_scale = self.logit_scale.exp()
        logit_scale = logit_scale.clamp(min=1e-6, max=100)
        logit_bias = self.logit_bias.clamp(min=-20.0, max=5.0)
        
        # Compute similarities and select pairs for each case
        selected_img_features = []
        selected_text_features = []
        selected_img_idx = []
        selected_text_idx = []
        output_similarities = []
        soft_weights_list = []  # Store soft weights for each batch sample
        
        for b in range(B):
            # Get valid images and text regions for this case
            if img_mask is not None:
                valid_images = img_mask[b]  # [V] boolean mask
            else:
                valid_images = torch.ones(V, dtype=torch.bool, device=img_cls_tkn.device)
            
            if text_mask is not None:
                valid_texts = text_mask[b]  # [N_r] boolean mask
            else:
                valid_texts = torch.ones(N_r, dtype=torch.bool, device=text_region_features.device)
            
            # Get valid image and text features
            valid_img_indices = torch.where(valid_images)[0]  # Indices of valid images
            valid_text_indices = torch.where(valid_texts)[0]  # Indices of valid texts
            
            # if len(valid_img_indices) == 0 or len(valid_text_indices) == 0:
            #     # Fallback to first image and first text if no valid ones
            #     selected_img_features.append(img_cls_features[b][0])
            #     selected_text_features.append(text_region_features[b][0])
            #     selected_img_idx.append(0)
            #     selected_text_idx.append([0])
            #     continue
            
            # Randomly select one valid image
            num_valid_imgs = len(valid_img_indices)
            random_img_idx_in_valid = torch.randint(0, num_valid_imgs, (1,), device=img_cls_tkn.device).item()
            selected_img_idx_in_batch = valid_img_indices[random_img_idx_in_valid].item()
            selected_img_feature = img_cls_features[b][selected_img_idx_in_batch]  # [D]
            
            # Get valid text features
            valid_text_features = text_region_features[b][valid_texts]  # [N_r_valid, D]
            
            # Compute similarities between selected image and all valid text regions
            # selected_img_feature: [D], valid_text_features: [N_r_valid, D]
            similarities_logits = logit_scale * (selected_img_feature @ valid_text_features.t()) + logit_bias  # [N_r_valid]
            
            # Normalize similarities to [0, 1] range using sigmoid
            similarities = torch.sigmoid(similarities_logits)  # [N_r_valid], range (0, 1)
            # Compute soft weights directly from similarities
            similarities_sum = similarities.sum()
            # Add numerical stability check: if sum is too small, use uniform weights
            if similarities_sum < 1e-6:
                # Fallback: if all similarities are too small, use uniform weights
                soft_weights = torch.ones_like(similarities) / len(similarities)
            else:
                soft_weights = similarities / similarities_sum  # [N_r_valid]
            
            # # Ensure no NaN or Inf values in weights
            # soft_weights = torch.nan_to_num(soft_weights, nan=1.0 / len(soft_weights), posinf=1.0 / len(soft_weights), neginf=0.0)
            # # Renormalize after NaN handling
            # soft_weights = soft_weights / (soft_weights.sum() + 1e-8)
            
            # TopK pooling based on accumulated ratio of soft weights
            # Sort weights in descending order
            sorted_weights, sorted_indices = torch.sort(soft_weights, descending=True)
            # Compute cumulative sum
            cumsum_weights = torch.cumsum(sorted_weights, dim=0)
            # rho >= 1.0 means "no selection": all sentences participate.
            if self.accumulate_ratio >= 1.0:
                k = torch.tensor(len(soft_weights), device=soft_weights.device)
            else:
                k = torch.searchsorted(cumsum_weights, self.accumulate_ratio, right=False) + 1
            # Floor by topk_min_k (>=1) and cap by number of valid features.
            k = torch.clamp(k, min=self.topk_min_k, max=len(soft_weights))

            # k = 1
            # print(f"k: {k}")

            # Select topk indices and weights
            topk_indices = sorted_indices[:k]
            topk_weights = sorted_weights[:k]
            # Renormalize topk weights
            topk_weights = topk_weights / (topk_weights.sum() + 1e-8)
            
            # Get topk text feature based on text_feature_mode
            if self.text_feature_mode == "reencode":
                # Map topk_indices back to original batch indices
                topk_original_indices = valid_text_indices[topk_indices]  # [k]
                # Retrieve original text strings for topk regions
                topk_texts = []
                for idx in topk_original_indices.cpu().tolist():
                    if idx < len(text[b]):
                        topk_texts.append(text[b][idx] if text[b][idx] else "")
                    else:
                        topk_texts.append("")
                
                # Shuffle the texts randomly
                shuffled_texts = topk_texts.copy()
                random.shuffle(shuffled_texts)
                
                # Concatenate shuffled texts into a new sentence
                # Use a separator (e.g., ". ") to join sentences
                concatenated_text = ". ".join([t for t in shuffled_texts if t.strip()])
                if not concatenated_text.strip():
                    concatenated_text = " "  # Fallback to empty string if all texts are empty
                
                # Re-encode the concatenated text using text encoder
                text_emb, _, _ = self._simple_encode_text([concatenated_text])  # [1, D]
                text_emb = text_emb.squeeze(0)  # [D]
                
                # Project to common embedding space
                pooled_text_feature = self.proj_t(text_emb)  # [D]
                # Normalize
                pooled_text_feature = F.normalize(pooled_text_feature, dim=-1)
            else:
                # Default: pooling mode
                # Weighted aggregation of topk text features
                topk_text_features = valid_text_features[topk_indices]  # [k, D]
                # pooled_text_feature = (topk_text_features * topk_weights.unsqueeze(-1)).sum(dim=0)  # [D]
                pooled_text_feature = topk_text_features.mean(dim=0)  # [D]
            
            # For logging: return all valid text indices with their soft weights
            # Map all valid text indices back to original batch indices
            all_text_indices_in_batch = valid_text_indices.tolist()  # All valid text region indices
            
            # Store results
            selected_img_features.append(selected_img_feature)
            selected_text_features.append(pooled_text_feature)
            selected_img_idx.append(selected_img_idx_in_batch)
            selected_text_idx.append(all_text_indices_in_batch)  # All valid text indices for logging
            output_similarities.append(similarities)
            # Store soft weights for each text region (contribution to pooled feature)
            # Map soft_weights back to original indices (pad with 0 for invalid regions)
            soft_weights_full = torch.zeros(N_r, device=soft_weights.device, dtype=soft_weights.dtype)
            soft_weights_full[valid_text_indices] = soft_weights
            soft_weights_list.append(soft_weights_full)
        # Stack selected features
        selected_img_features = torch.stack(selected_img_features, dim=0)  # [B, D]
        selected_text_features = torch.stack(selected_text_features, dim=0)  # [B, D]

        return {
            "img_cls_features": img_cls_features,
            "text_region_features": text_region_features,
            "selected_img_features": selected_img_features,  # [B, D] - randomly selected image features
            "selected_text_features": selected_text_features,  # [B, D] - pooled text features from soft-weighted aggregation
            "logit_scale": logit_scale,
            "logit_bias": logit_bias,
            "selected_img_idx": selected_img_idx,  # List[int] - indices of randomly selected images
            "selected_text_idx": selected_text_idx,  # List[List[int]] - all valid text region indices for each sample
            "output_similarities": output_similarities,  # List[Tensor] - similarities between selected image and text regions for each sample
            "soft_weights": soft_weights_list,  # List[Tensor] - soft weights for each text region (contribution to pooled feature) [B, N_r]
        }

    # Inference
    def get_image_features(self, images):
        """Extract normalized image features for inference"""
        with torch.no_grad():
            img_emb, _ = self.encode_image(images)
            image_features = self.proj_v(img_emb)
            image_features = F.normalize(image_features, dim=-1)
        return image_features

    def get_text_features(self, texts):
        """Extract normalized text features for inference"""
        with torch.no_grad():
            text_features = self.proj_t(self.encode_text(texts))
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def compute_similarity(self, image_features, text_features):
        """Compute similarity scores between image and text features"""
        logit_scale = self.logit_scale.exp()
        return logit_scale * image_features @ text_features.t() + self.logit_bias

    def zero_shot_predict(self, images, texts):
        """Zero-shot prediction given images and candidate texts"""
        image_features = self.get_image_features(images)
        text_features = self.get_text_features(texts)

        logits_per_image = self.compute_similarity(image_features, text_features)
        probs = F.softmax(logits_per_image, dim=-1)

        return {
            "logits": logits_per_image,
            "probs": probs,
            "image_features": image_features,
            "text_features": text_features,
        }

    def initialize_weights(self, skip_vision_encoder=True):
        # if not skip_vision_encoder:
        #     # Initialize position embeddings
        #     encoder_pos_embed = get_2d_sincos_pos_embed(
        #         self.vision_encoder.pos_embed.shape[-1],
        #         int(self.num_patches**0.5),
        #         cls_token=True,
        #     )
        #     self.vision_encoder.pos_embed.data.copy_(
        #         torch.from_numpy(encoder_pos_embed).float().unsqueeze(0)
        #     )
            
        #     # Initialize patch embedding weights
        #     w = self.vision_encoder.patch_embed.proj.weight.data
        #     torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        #     print_rank_zero("Vision encoder weights randomly initialized")
        # else:
        #     print_rank_zero("Vision encoder weights will be initialized from pretrained weights")

        self._init_weights(self.proj_img)
        if self.local_loss:
            self._init_weights(self.proj_v)
        self._init_weights(self.proj_t)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


# TODO Current implementation creates encoder based on parameters not model name.
# Not necessarily a problem, but should be noted.
def create_topk_vlp_model(
    vision_encoder_name="vit_base_patch16",
    vision_encoder_dim=768,
    vision_encoder_depth=12,
    vision_encoder_num_heads=12,
    vision_encoder_mlp_ratio=4,
    vision_encoder_drop_path_rate=0.1,
    in_chans=1,
    img_size=224,
    patch_size=16,
    output_dim=512,
    text_encoder_name="pubmedbert",
    text_freeze_layers=0,
    pretrained_weight_path=None,
    pretrained_load="vision_encoder",
    hf_siglip_model_id=None,
    text_feature_mode="pooling",  # "pooling" or "reencode"
    max_text_length=64,  # Max length for single region text
    max_text_length_reencode=256,  # Max length for concatenated text in reencode mode
    accumulate_ratio=0.8,
    topk_min_k=1,
):
    """Create VLP model with vision and text encoders"""
    # Create vision encoder
    if vision_encoder_name == "rad-dino":
        from .rad_dino import RadDino
        vision_encoder = RadDino()
        del vision_encoder.model.embeddings.mask_token
    elif vision_encoder_name == "ark_base":
        from timm.models.swin_transformer import SwinTransformer
        vision_encoder = SwinTransformer(
            img_size=224,
            patch_size=4,
            window_size=7,
            embed_dim=128,
            depths=(2, 2, 18, 2),
            num_heads=(4, 8, 16, 32),
            use_checkpoint=True
        )
        weight = pretrained_weight_path or os.environ.get("ARK6_SWINBASE_WEIGHT")
        if not weight:
            raise ValueError(
                "ARK base checkpoint path is required. Set pretrained_weight_path "
                "in the config or ARK6_SWINBASE_WEIGHT in the environment."
            )
        checkpoint = torch.load(weight, map_location=torch.device('cpu'), weights_only=False)
        state_dict = checkpoint["teacher"]
        if any([True if 'module.' in k else False for k in state_dict.keys()]):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k.startswith('module.')}       
        missing, unexpected = vision_encoder.load_state_dict(state_dict, strict=False)
        print_rank_zero(f"Vision encoder weights loaded from: {weight}")
        print_rank_zero(f"missing: {missing}")
        print_rank_zero(f"unexpected: {unexpected}")
    elif vision_encoder_name == "ark_large":
        from timm.models.swin_transformer import SwinTransformer
        vision_encoder = SwinTransformer(
            img_size=768,
            patch_size=4,
            window_size=12,
            embed_dim=192,
            depths=(2, 2, 18, 2),
            num_heads=(6, 12, 24, 48),
            use_checkpoint=True
        )
        weight = pretrained_weight_path or os.environ.get("ARK6_SWINLARGE_WEIGHT")
        if not weight:
            raise ValueError(
                "ARK large checkpoint path is required. Set pretrained_weight_path "
                "in the config or ARK6_SWINLARGE_WEIGHT in the environment."
            )
        checkpoint = torch.load(weight, map_location=torch.device('cpu'), weights_only=False)
        state_dict = checkpoint["teacher"]
        if any([True if 'module.' in k else False for k in state_dict.keys()]):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k.startswith('module.')}       
        missing, unexpected = vision_encoder.load_state_dict(state_dict, strict=False)
        print_rank_zero(f"Vision encoder weights loaded from: {weight}")
        print_rank_zero(f"missing: {missing}")
        print_rank_zero(f"unexpected: {unexpected}")
    else:
        vision_encoder = create_vision_encoder(
            vision_encoder_name=vision_encoder_name,
            img_size=img_size,
            in_chans=in_chans,
            patch_size=patch_size,
            embed_dim=vision_encoder_dim,
            depth=vision_encoder_depth,
            num_heads=vision_encoder_num_heads,
            mlp_ratio=vision_encoder_mlp_ratio,
            drop_path_rate=vision_encoder_drop_path_rate,
            qkv_bias=True,
            global_pool=False,
            hf_siglip_model_id=hf_siglip_model_id,
        )
 
    # Remove unused classification head from vision encoder
    if hasattr(vision_encoder, 'head'):
        del vision_encoder.head

    # Create text encoder and tokenizer
    text_encoder, tokenizer = create_text_encoder(text_encoder_name=text_encoder_name)
    # Create VLP model
    model = XRayTopKVLP(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_freeze_layers=text_freeze_layers,
        output_dim=output_dim,
        pretrained_weight_path=pretrained_weight_path,
        pretrained_load=pretrained_load,
        vision_encoder_name=vision_encoder_name,
        text_feature_mode=text_feature_mode,
        max_text_length=max_text_length,
        max_text_length_reencode=max_text_length_reencode,
        accumulate_ratio=accumulate_ratio,
        topk_min_k=topk_min_k,
    )
    if pretrained_weight_path is not None:
        weight = load_weight(pretrained_weight_path, scope=pretrained_load)
        if pretrained_load in ["vision_encoder", "vision", "encoder_only"]:
            missing, unexpected = model.vision_encoder.load_state_dict(weight, strict=False)
        elif pretrained_load in ["text_encoder", "text"]:
            missing, unexpected = model.text_encoder.load_state_dict(weight, strict=False)
        else:
            # Load into the whole model for scopes like 'all', 'encoders'
            missing, unexpected = model.load_state_dict(weight, strict=False)
        print_rank_zero(f"Missing: {missing}")
        print_rank_zero(f"Unexpected: {unexpected}")
    return model

def load_weight(ckpt_path, scope="vision_encoder"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]

    # Remove the top-level module prefix saved by Lightning: 'model.'
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    def _remove_head_keys(sd):
        # Drop any classification heads
        keys_to_drop = [k for k in sd.keys() if k.endswith("head.weight") or k.endswith("head.bias")]
        for k in keys_to_drop:
            sd.pop(k, None)
        return sd

    normalized_scope = str(scope or "vision_encoder").lower()

    if normalized_scope in ["vision_encoder", "vision", "encoder_only"]:
        has_vision_encoder = any(k.startswith("vision_encoder.") for k in state_dict.keys())
        has_encoder = any(k.startswith("encoder.") for k in state_dict.keys())
        if has_vision_encoder:
            sd = {k.replace("vision_encoder.", ""): v for k, v in state_dict.items() if k.startswith("vision_encoder.")}
        elif has_encoder:
            sd = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}
        else:
            # Fallback: try keys that belong to the vision encoder without explicit prefix
            sd = {k: v for k, v in state_dict.items() if not (k.startswith("text_encoder.") or k.startswith("fusion.") or k.startswith("proj_") or k == "logit_scale")}
        return _remove_head_keys(sd)

    if normalized_scope in ["text_encoder", "text"]:
        sd = {k.replace("text_encoder.", ""): v for k, v in state_dict.items() if k.startswith("text_encoder.")}
        return sd

    if normalized_scope in ["fusion", "fusion_only"]:
        allowed_prefixes = ("fusion.", "proj_v.", "proj_t.", "proj_case.", "logit_scale")
        sd = {k: v for k, v in state_dict.items() if k.startswith(allowed_prefixes) or k == "logit_scale"}
        return sd

    if normalized_scope in ["encoders", "both_encoders"]:
        sd = {k: v for k, v in state_dict.items() if k.startswith("vision_encoder.") or k.startswith("text_encoder.") or k.startswith("encoder.")}
        # Normalize possible MAE-style 'encoder.' to 'vision_encoder.'
        sd = { (k.replace("encoder.", "vision_encoder.") if k.startswith("encoder.") else k): v for k, v in sd.items() }
        return _remove_head_keys(sd)

    # Default: load the whole model (all)
    # Also normalize possible MAE-style 'encoder.' prefix to our 'vision_encoder.'
    sd = { (k.replace("encoder.", "vision_encoder.") if k.startswith("encoder.") else k): v for k, v in state_dict.items() }
    # In 'all' mode, also remove head keys to avoid unused parameters in DDP
    return _remove_head_keys(sd)

if __name__ == "__main__":
    model = create_topk_vlp_model(
        vision_encoder_name="ark_large",
        vision_encoder_dim=768,
        vision_encoder_depth=12,
        vision_encoder_num_heads=12,
        vision_encoder_mlp_ratio=4,
        vision_encoder_drop_path_rate=0.1,
        in_chans=1,
        img_size=256,
        patch_size=16,
        output_dim=256,
        text_encoder_name="pubmedbert",
        text_freeze_layers=0,
        pretrained_weight_path=None,  # No pretrained weights for testing
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

    import torch

    dummy_input = torch.randn(5, 4, 3, 768, 768)
    dummy_text = [
        [
            "Patient presents with shortness of breath and chest pain.",
            "No prior imaging available.",
        ],
        ["Patient presents.", "No prior imaging available."],
        [
            "Patient presents with shortness of breath and chest pain.",
            "No prior imaging available.",
        ],
        ["Patient presents.", "No prior imaging available."],
        [
            "Patient presents with shortness of breath and chest pain.",
            "No prior imaging available.",
        ],
    ]
    
    # Create masks for testing
    img_mask = torch.ones(5, 4, dtype=torch.bool)  # All images are valid
    text_mask = torch.ones(5, 2, dtype=torch.bool)  # All text regions are valid
    
    output = model(dummy_input, dummy_text, img_mask=img_mask, text_mask=text_mask)
    if output['img_cls_features'] is not None:
        print(f"Image CLS Features: {output['img_cls_features'].shape}")  # [B, V, D]
    else:
        print("Image CLS Features: None (local_loss=False)")
    print(f"Text Region Features: {output['text_region_features'].shape}")  # [B, N_r, D]
    print(f"Selected Image Features: {output['selected_img_features'].shape}")  # [B, D]
    print(f"Selected Text Features: {output['selected_text_features'].shape}")  # [B, D]

    from utils.loss import clip_loss, local_loss

    # Test contrastive loss with selected pairs
    contrastive_loss, acc1, acc5, i2t_acc1, i2t_acc5, t2i_acc1, t2i_acc5 = clip_loss(
        output["selected_img_features"],
        output["selected_text_features"],
        temperature=1.0 / output["logit_scale"].item(),
        device=dummy_input.device,
    )
    print(f"Selected Pairs Contrastive Loss: {contrastive_loss}")
    print(f"Selected Pairs Acc1: {acc1}")
    print(f"Selected Pairs Acc5: {acc5}")

    # Test local loss if available
    if output["img_cls_features"] is not None:
        local_contrastive_loss, local_acc1, local_acc5, attention_maps = local_loss(
            output["img_cls_features"],
            output["text_region_features"],
            attn_temp=4.0,
            text_agg_temp=5.0,
            final_temp=10.0,
            agg="sum",
            device=dummy_input.device,
        )
        print(f"Local Contrastive Loss: {local_contrastive_loss}")
        print(f"Local Acc1: {local_acc1}")
        print(f"Local Acc5: {local_acc5}")
