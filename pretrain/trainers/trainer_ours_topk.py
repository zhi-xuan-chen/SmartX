import math
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb

from modules import create_topk_vlp_model
from utils import print_rank_zero
from utils.loss import (
    siglip_loss, 
    clip_loss, 
    siglip_loss_ring, 
    siglip_loss_gather, 
)

matplotlib.use("Agg")  # Use non-interactive backend for server environments


def param_groups_layer_decay(model, weight_decay=0.0, layer_decay=0.75, no_weight_decay_list=None):
    """
    Custom implementation of layer decay for parameter groups.
    Groups parameters by layer depth and applies layer decay scaling.
    
    Args:
        model: The model to extract parameters from
        weight_decay: Weight decay value
        layer_decay: Layer decay factor (e.g., 0.75 means each layer has 0.75x LR of previous)
        no_weight_decay_list: List of parameter names that should not have weight decay
    
    Returns:
        List of parameter group dicts with 'params', 'weight_decay', and 'lr_scale' keys
    """
    if no_weight_decay_list is None:
        no_weight_decay_list = []
    
    # Group parameters by layer depth
    layer_groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Extract layer number from parameter name
        # Patterns: layers.0, blocks.0, layer_0, encoder.layer.0, etc.
        layer_match = re.search(r'(?:layers|blocks|layer|encoder\.layer)[._](\d+)', name)
        if layer_match:
            layer_idx = int(layer_match.group(1))
        else:
            # For parameters without explicit layer (e.g., patch_embed, norm, embedding), assign to layer 0
            layer_idx = 0
        
        if layer_idx not in layer_groups:
            layer_groups[layer_idx] = {'decay': [], 'no_decay': []}
        
        # Check if this parameter should have weight decay
        should_decay = not any(nd in name for nd in no_weight_decay_list)
        
        if should_decay:
            layer_groups[layer_idx]['decay'].append(param)
        else:
            layer_groups[layer_idx]['no_decay'].append(param)
    
    # Create parameter groups with layer decay scaling
    param_groups = []
    max_layer = max(layer_groups.keys()) if layer_groups else 0
    
    for layer_idx in sorted(layer_groups.keys()):
        # Calculate LR scale: deeper layers have smaller LR
        # Layer 0 (shallowest) has scale 1.0, deeper layers have scale layer_decay^(max_layer - layer_idx)
        lr_scale = layer_decay ** (max_layer - layer_idx)
        
        # Add decay group
        if layer_groups[layer_idx]['decay']:
            param_groups.append({
                'params': layer_groups[layer_idx]['decay'],
                'weight_decay': weight_decay,
                'lr_scale': lr_scale,
            })
        
        # Add no_decay group
        if layer_groups[layer_idx]['no_decay']:
            param_groups.append({
                'params': layer_groups[layer_idx]['no_decay'],
                'weight_decay': 0.0,
                'lr_scale': lr_scale,
            })
    
    return param_groups


class OursTopKWrapper(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.lr, self.vision_lr, self.text_lr, self.fusion_lr = self._calculate_scaled_learning_rates(cfg)
        self.weight_decay = cfg["weight_decay"]
        self.warmup_steps = cfg["warmup_steps"]
        self.total_steps = cfg["total_steps"]
        self.log_image_freq = cfg.get("log_image_freq", 100)
        self.log_train_similarity = bool(cfg.get("log_train_similarity", True))
        self.img_size = cfg["img_size"]
        self.global_loss = cfg["global_loss"]
        # Ensure weight is a Python float for safe arithmetic
        self.vision_layer_decay = cfg.get("vision_layer_decay", 0)
        self.text_layer_decay = cfg.get("text_layer_decay", 0)
        # Projection warmup: number of epochs to train only projection layers
        self.projection_warmup_epochs = cfg.get("projection_warmup_epochs", 1)
        # Decoupled warmup schedule:
        # - encoder: lr=0 during projection_warmup_epochs, then warmup + cosine
        # - projection: normal LR from step 0 (no long warmup)
        # - logit_scale/logit_bias: short step-based warmup (default 500)
        self.encoder_warmup_epochs = cfg.get("encoder_warmup_epochs", cfg.get("warmup_epochs", 1))
        self.logit_warmup_steps = int(cfg.get("logit_warmup_steps", 500))
        self.topk_transition_epochs = float(cfg.get("local_alignment_transition_epochs", cfg.get("topk_transition_epochs", 0)))
        # Decouple transition window from dual-forward mixing:
        # by default, transition only affects encoder LR (no second forward).
        self.topk_transition_mix_loss = bool(cfg.get("local_alignment_transition_mix_loss", cfg.get("topk_transition_mix_loss", False)))
        self.transition_encoder_lr_scale = float(cfg.get("transition_encoder_lr_scale", 1.0))
        self.transition_encoder_lr_ramp = bool(cfg.get("transition_encoder_lr_ramp", True))
        max_epochs = max(1, int(cfg.get("max_epochs", 1)))
        self.topk_transition_steps = int(
            max(0.0, self.topk_transition_epochs) * max(1, self.total_steps) / max_epochs
        )
        self._in_projection_warmup = False
        self._encoder_warmup_start_step = None  # Track the global_step when encoder training starts
        # Create Ours TopK model (CLIP with TopK selection)
        self.model = create_topk_vlp_model(
            vision_encoder_name=cfg["vision_encoder_name"],
            vision_encoder_dim=cfg["vision_encoder_dim"],
            vision_encoder_depth=cfg["vision_encoder_depth"],
            vision_encoder_num_heads=cfg["vision_encoder_num_heads"],
            vision_encoder_mlp_ratio=cfg["vision_encoder_mlp_ratio"],
            vision_encoder_drop_path_rate=cfg["vision_encoder_drop_path_rate"],
            in_chans=cfg["in_chans"],
            img_size=cfg["img_size"],
            patch_size=cfg["patch_size"],
            output_dim=cfg["output_dim"],
            text_encoder_name=cfg["text_encoder_name"],
            text_freeze_layers=cfg["text_freeze_layers"],
            pretrained_weight_path=cfg.get("pretrained_weight_path", cfg.get("vision_encoder_pretrained", None)),
            pretrained_load=cfg.get("pretrained_load", "vision_encoder"),
            vision_init_path=cfg.get("vision_init_path"),
            smartx_checkpoint=cfg.get("smartx_checkpoint", "auto"),
            resume_training=cfg.get("resume_training", False),
            hf_siglip_model_id=cfg.get("hf_siglip_model_id", None),
            text_feature_mode=cfg.get("text_feature_mode", "pooling"),  # "pooling" or "reencode"
            max_text_length=cfg.get("max_text_length", 64),  # Max length for single region text
            max_text_length_reencode=cfg.get("max_text_length_reencode", 256),  # Max length for concatenated text in reencode mode
            accumulate_ratio=cfg.get("accumulate_ratio", 0.8),
            topk_min_k=cfg.get("topk_min_k", 1),
        )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["smartx_training_stage"] = {
            "in_projection_warmup": self._in_projection_warmup,
            "encoder_warmup_start_step": self._encoder_warmup_start_step,
        }

    def on_load_checkpoint(self, checkpoint):
        stage = checkpoint.get("smartx_training_stage")
        if stage is not None:
            self._in_projection_warmup = stage["in_projection_warmup"]
            self._encoder_warmup_start_step = stage["encoder_warmup_start_step"]
        elif self.projection_warmup_epochs == 0:
            self._encoder_warmup_start_step = 0
        else:
            # Older checkpoints lack stage metadata; retain the transition at
            # the next epoch rather than leaving encoder learning rates at zero.
            print_rank_zero("Checkpoint lacks training-stage metadata; encoder warmup restarts at the resumed epoch.")
            self._in_projection_warmup = True

    def forward(self, img, text, image_attention_mask, text_attention_mask=None):
        return self.model(img, text, img_mask=image_attention_mask, text_mask=text_attention_mask)
    
    def on_train_epoch_start(self):
        """Called at the start of each training epoch"""
        prev_flag = self._in_projection_warmup
        self._in_projection_warmup = (
            self.projection_warmup_epochs > 0
            and self.current_epoch < self.projection_warmup_epochs
        )

        if self._in_projection_warmup and not prev_flag:
            print_rank_zero(
                f"[Epoch {self.current_epoch}] Projection warmup: encoder lr=0, only projection/logit training."
            )
        # Record encoder warmup start step at the first epoch after projection warmup
        if (not self._in_projection_warmup) and prev_flag:
            if self._encoder_warmup_start_step is None:
                self._encoder_warmup_start_step = int(self.global_step)
                print_rank_zero(
                    f"[Epoch {self.current_epoch}] Projection warmup finished. "
                    f"Encoder warmup will start from global_step={self._encoder_warmup_start_step}."
                )
        # Edge case: when projection_warmup_epochs==0, the transition above never fires.
        # Set encoder warmup start step to 0 at epoch 0.
        if (
            self.projection_warmup_epochs <= 0
            and self._encoder_warmup_start_step is None
        ):
            self._encoder_warmup_start_step = 0
            print_rank_zero(
                "[Epoch 0] projection_warmup_epochs=0 → encoder warmup starts at global_step=0."
            )

    def training_step(self, batch, batch_idx):
        imgs = batch["image"]
        image_attention_mask = batch["image_attention_mask"]
        texts = batch["text"]
        text_attention_mask = batch["text_attention_mask"]

        mixed_alpha = 0.0
        warmup_acc1 = None
        topk_acc1 = None

        if self.global_step < self.warmup_steps:
            outputs = self.model.warmup_forward(imgs, texts, img_mask=image_attention_mask, text_mask=text_attention_mask)
            img_cls_tkn = outputs["img_cls_tkn"]
            text_embeddings = outputs["text_embeddings"]
            logit_scale = outputs["logit_scale"]
            logit_bias = outputs["logit_bias"]
            contrastive_loss, acc1, acc5, i2t_acc1, i2t_acc5, t2i_acc1, t2i_acc5 = self._compute_global_loss(
                selected_img_features=img_cls_tkn,
                selected_text_features=text_embeddings,
                logit_scale=logit_scale,
                logit_bias=logit_bias,
                device=imgs.device,
            )
            warmup_acc1 = acc1
        else:
            in_topk_transition = (
                self.topk_transition_mix_loss
                and self.topk_transition_steps > 0
                and self.global_step < self.warmup_steps + self.topk_transition_steps
            )
            if in_topk_transition:
                local_step = self.global_step - self.warmup_steps + 1
                mixed_alpha = min(1.0, max(0.0, local_step / max(1, self.topk_transition_steps)))

                warmup_outputs = self.model.warmup_forward(
                    imgs, texts, img_mask=image_attention_mask, text_mask=text_attention_mask
                )
                warmup_loss, warmup_acc1, warmup_acc5, warmup_i2t_acc1, warmup_i2t_acc5, warmup_t2i_acc1, warmup_t2i_acc5 = self._compute_global_loss(
                    selected_img_features=warmup_outputs["img_cls_tkn"],
                    selected_text_features=warmup_outputs["text_embeddings"],
                    logit_scale=warmup_outputs["logit_scale"],
                    logit_bias=warmup_outputs["logit_bias"],
                    device=imgs.device,
                )

            outputs = self.model.forward(imgs, texts, img_mask=image_attention_mask, text_mask=text_attention_mask)
            # Extract outputs
            selected_img_features = outputs["selected_img_features"]
            selected_text_features = outputs["selected_text_features"]
            logit_scale = outputs["logit_scale"]
            logit_bias = outputs["logit_bias"]
        
            # 1. Global Contrastive Loss with selected pairs
            device = selected_img_features.device
            contrastive_loss, acc1, acc5, i2t_acc1, i2t_acc5, t2i_acc1, t2i_acc5 = self._compute_global_loss(
                selected_img_features=selected_img_features,
                selected_text_features=selected_text_features,
                logit_scale=logit_scale,
                logit_bias=logit_bias,
                device=device,
            )
            topk_acc1 = acc1

            if in_topk_transition:
                contrastive_loss = (1.0 - mixed_alpha) * warmup_loss + mixed_alpha * contrastive_loss
                acc1 = (1.0 - mixed_alpha) * warmup_acc1 + mixed_alpha * topk_acc1
                acc5 = (1.0 - mixed_alpha) * warmup_acc5 + mixed_alpha * acc5
                i2t_acc1 = (1.0 - mixed_alpha) * warmup_i2t_acc1 + mixed_alpha * i2t_acc1
                i2t_acc5 = (1.0 - mixed_alpha) * warmup_i2t_acc5 + mixed_alpha * i2t_acc5
                t2i_acc1 = (1.0 - mixed_alpha) * warmup_t2i_acc1 + mixed_alpha * t2i_acc1
                t2i_acc5 = (1.0 - mixed_alpha) * warmup_t2i_acc5 + mixed_alpha * t2i_acc5
            
        # Combined loss (conditional components)
        total_loss = contrastive_loss
        # Log scalar metrics
        self.log(
            "train/total_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/contrastive_loss",
            contrastive_loss,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/acc1", 
            acc1, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True,
        )
        self.log(
            "train/acc5", 
            acc5, 
            on_step=True, 
            on_epoch=True,
        )
        self.log(
            "train/i2t_acc1", 
            i2t_acc1, 
            on_step=True, 
            on_epoch=True,
        )
        self.log(
            "train/i2t_acc5", 
            i2t_acc5, 
            on_step=True, 
            on_epoch=True,
        )
        self.log(
            "train/t2i_acc1", 
            t2i_acc1, 
            on_step=True, 
            on_epoch=True,
        )
        self.log(
            "train/t2i_acc5", 
            t2i_acc5, 
            on_step=True, 
            on_epoch=True,
        )
        self.log(
            "train/logit_scale", 
            logit_scale, 
            on_step=True, 
            on_epoch=True,
        )
        self.log(
            "train/logit_bias", 
            logit_bias, 
            on_step=True, 
            on_epoch=True,
        )
        # Log whether we're in projection warmup mode
        self.log(
            "train/projection_warmup_mode",
            float(self._in_projection_warmup),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/mixed_alpha",
            float(mixed_alpha),
            on_step=True,
            on_epoch=True,
        )
        if warmup_acc1 is not None:
            self.log(
                "train/warmup_acc1",
                warmup_acc1,
                on_step=True,
                on_epoch=True,
            )
        if topk_acc1 is not None:
            self.log(
                "train/topk_acc1",
                topk_acc1,
                on_step=True,
                on_epoch=True,
            )

        # Log learning rates for each component
        opt = self.optimizers(use_pl_optimizer=True)
        if opt:
            # Log learning rates by parameter group name
            for param_group in opt.param_groups:
                group_name = param_group.get('name', 'unknown')
                lr = param_group["lr"]
                
                if 'vision' in group_name.lower():
                    self.log("lr/vision_lr", lr, on_step=True)
                elif 'text' in group_name.lower():
                    self.log("lr/text_lr", lr, on_step=True)
                elif 'projection' in group_name.lower():
                    self.log("lr/projection_lr", lr, on_step=True)
                elif 'logit_scale' in group_name.lower():
                    self.log("lr/logit_scale_lr", lr, on_step=True)
                    
            # Also log the overall/first group LR for backward compatibility
            self.log("lr/lr", opt.param_groups[0]["lr"], on_step=True)

        # Log visualizations once per log_image_freq steps (only on rank 0)
        if self.log_image_freq > 0 and self.global_step % self.log_image_freq == 0 and self.global_rank == 0:
            if self.global_step < self.warmup_steps:
                selected_img_features = img_cls_tkn
                selected_text_features = text_embeddings
            else:
                # Log all text regions with their soft weights
                selected_text_indices = outputs["selected_text_idx"][0]  # All valid text indices
                soft_weights = outputs.get("soft_weights", [None])[0]  # Soft weights for each text region
                
                # Create caption with weights for each sentence (truncated for wandb display)
                selected_text = self._format_text_with_weights(
                    selected_text_indices, soft_weights, texts[0] if texts else []
                )
                
                selected_img_idx = outputs["selected_img_idx"][0]
                self._log_selected_pairs(
                    imgs[0][selected_img_idx], 
                    selected_text,
                    batch_idx=batch_idx,
                    image_paths=batch.get("image_paths"),
                    selected_img_idx=selected_img_idx
                )
                if self.log_train_similarity:
                    # log similarity
                    output_similarities = outputs["output_similarities"][0]
                    self._log_similarity(
                        output_similarities
                    )

            if self.log_train_similarity:
                self._log_visualizations(
                    selected_img_features,
                    selected_text_features
                )

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step - compute loss and metrics"""
        imgs = batch["image"]
        image_attention_mask = batch["image_attention_mask"]
        texts = batch["text"]
        text_attention_mask = batch["text_attention_mask"]

        outputs = self.model(imgs, texts, img_mask=image_attention_mask, text_mask=text_attention_mask)

        # Extract outputs
        selected_img_features = outputs["selected_img_features"]
        selected_text_features = outputs["selected_text_features"]
        logit_scale = outputs["logit_scale"]
        logit_bias = outputs["logit_bias"]

        # 1. Global Contrastive Loss with selected pairs
        device = selected_img_features.device
        contrastive_loss, acc1, acc5, i2t_acc1, i2t_acc5, t2i_acc1, t2i_acc5 = self._compute_global_loss(
            selected_img_features=selected_img_features,
            selected_text_features=selected_text_features,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
            device=device,
        )
        total_loss = contrastive_loss

        # Log validation metrics
        if self.log_image_freq > 0 and self.global_rank == 0 and batch_idx % self.log_image_freq == 0:
            if self.global_step > self.warmup_steps:
                # Log all text regions with their soft weights
                selected_text_indices = outputs["selected_text_idx"][0]  # All valid text indices
                soft_weights = outputs.get("soft_weights", [None])[0]  # Soft weights for each text region
                
                # Create caption with weights for each sentence (truncated for wandb display)
                selected_text = self._format_text_with_weights(
                    selected_text_indices, soft_weights, texts[0] if texts else []
                )
                
                selected_img_idx = outputs["selected_img_idx"][0]
                self._log_selected_pairs(
                    imgs[0][selected_img_idx], 
                    selected_text,
                    batch_idx=batch_idx,
                    image_paths=batch.get("image_paths"),
                    selected_img_idx=selected_img_idx
                )
                # log similarity
                output_similarities = outputs["output_similarities"][0]
                self._log_similarity(
                    output_similarities
                )
            self._log_visualizations(
                selected_img_features,
                selected_text_features,
                log_name="Validation",
            )
        self.log(
            "val/total_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val/contrastive_loss",
            contrastive_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/acc1",
            acc1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val/acc5", 
            acc5, 
            on_step=False, 
            on_epoch=True, 
            sync_dist=True,
        )
        self.log(
            "val/i2t_acc1", 
            i2t_acc1, 
            on_step=False, 
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/i2t_acc5", 
            i2t_acc5, 
            on_step=False, 
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/t2i_acc1", 
            t2i_acc1, 
            on_step=False, 
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/t2i_acc5", 
            t2i_acc5, 
            on_step=False, 
            on_epoch=True,
            sync_dist=True,
        )
        return total_loss
    
    def _compute_global_loss(
        self, 
        selected_img_features, 
        selected_text_features, 
        logit_scale, 
        logit_bias,
        device,
    ):
        if self.global_loss == "siglip_ring":
            contrastive_loss, acc1, acc5 = siglip_loss_ring(
                selected_img_features,
                selected_text_features,
                temperature=1.0 / logit_scale,
                logit_bias=logit_bias,
            )
        elif self.global_loss == "siglip":
            contrastive_loss, acc1, acc5 = siglip_loss(
                selected_img_features,
                selected_text_features,
                temperature=1.0 / logit_scale,
                logit_bias=logit_bias,
                device=device,
            )
        elif self.global_loss == "clip":
            contrastive_loss, acc1, acc5 = clip_loss(
                selected_img_features,
                selected_text_features,
                temperature=1.0 / logit_scale,
                device=device,
            )
        elif self.global_loss == "siglip_all_gather":
            # Gather features from all GPUs for larger effective batch size
            selected_img_features_all = self.all_gather(
                selected_img_features, sync_grads=False
            )
            selected_text_features_all = self.all_gather(
                selected_text_features, sync_grads=False
            )

            # Flatten to concatenate batches: [num_gpus, batch_size, dim] -> [num_gpus * batch_size, dim]
            selected_img_features_all = selected_img_features_all.view(
                -1, selected_img_features_all.shape[-1]
            )
            selected_text_features_all = selected_text_features_all.view(
                -1, selected_text_features_all.shape[-1]
            )
            contrastive_loss, acc1, acc5, i2t_acc1, i2t_acc5, t2i_acc1, t2i_acc5 = siglip_loss_gather(
                selected_img_features,
                selected_img_features_all,
                selected_text_features,
                selected_text_features_all,
                temperature=1.0 / logit_scale,
                logit_bias=logit_bias,
                device=device,
            )
        elif self.global_loss == "clip_all_gather":
            # Gather features from all GPUs for larger effective batch size
            selected_img_features_all = self.all_gather(
                selected_img_features, sync_grads=True
            )
            selected_text_features_all = self.all_gather(
                selected_text_features, sync_grads=True
            )

            # Flatten to concatenate batches: [num_gpus, batch_size, dim] -> [num_gpus * batch_size, dim]
            selected_img_features_all = selected_img_features_all.view(
                -1, selected_img_features_all.shape[-1]
            )
            selected_text_features_all = selected_text_features_all.view(
                -1, selected_text_features_all.shape[-1]
            )

            contrastive_loss, acc1, acc5 = clip_loss(
                selected_img_features_all,
                selected_text_features_all,
                temperature=1.0 / logit_scale,
                device=device,
            )
        return contrastive_loss, acc1, acc5, i2t_acc1, i2t_acc5, t2i_acc1, t2i_acc5
    
    def _format_text_with_weights(self, selected_text_indices, soft_weights, texts):
        """
        Format text with weights for wandb display, sorted by weight (descending).
        
        Args:
            selected_text_indices: List of text region indices
            soft_weights: Tensor of soft weights for each text region
            texts: List of text strings
            
        Returns:
            Formatted string with weights, sorted by weight
        """
        if not selected_text_indices:
            return ""
        
        if soft_weights is not None:
            # Create list of (weight, idx) pairs and sort by weight (descending)
            weight_idx_pairs = []
            for idx in selected_text_indices:
                if idx < len(texts):
                    weight = soft_weights[idx].item() if isinstance(soft_weights, torch.Tensor) else 0.0
                    weight_idx_pairs.append((weight, idx))
            
            # Sort by weight (descending)
            weight_idx_pairs.sort(key=lambda x: x[0], reverse=True)
            
            # Build caption with weights, sorted by weight
            text_with_weights = []
            for weight, idx in weight_idx_pairs:
                text = texts[idx]
                text_with_weights.append(f"[w={weight:.3f}] {text}")
            
            selected_text = " | ".join(text_with_weights)
            return selected_text
        else:
            # Fallback: just join texts without weights
            texts_list = []
            for idx in selected_text_indices:
                if idx < len(texts):
                    texts_list.append(texts[idx])
            selected_text = " | ".join(texts_list)
            return selected_text
    
    def _log_similarity(self, similarities):
        """Log similarity values as histogram and bar chart"""
        try:
            # Convert tensor to numpy array
            if isinstance(similarities, torch.Tensor):
                similarities_np = similarities.detach().cpu().numpy()
            else:
                similarities_np = np.array(similarities)
            
            # Log as histogram to show distribution
            self.logger.experiment.log(
                {
                    "similarity/histogram": wandb.Histogram(similarities_np),
                    "similarity/mean": float(np.mean(similarities_np)),
                    "similarity/std": float(np.std(similarities_np)),
                    "similarity/min": float(np.min(similarities_np)),
                    "similarity/max": float(np.max(similarities_np)),
                },
                step=self.global_step,
            )
            
            # Optionally create a bar chart visualization for small vectors
            if len(similarities_np) <= 20:  # Only visualize if not too many values
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(range(len(similarities_np)), similarities_np)
                ax.set_title("Similarity Scores for Selected Text Regions")
                ax.set_xlabel("Text Region Index")
                ax.set_ylabel("Similarity Score")
                ax.set_ylim(0, 1)
                
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                plot_img = np.asarray(buf)
                plot_img = plot_img[:, :, :3]  # Remove alpha channel
                plt.close(fig)
                
                self.logger.experiment.log(
                    {
                        "similarity/bar_chart": wandb.Image(
                            plot_img, caption="Similarity scores for each text region"
                        ),
                    },
                    step=self.global_step,
                )
        except Exception as e:
            # Log error but don't crash training if visualization fails
            print_rank_zero(f"Warning: Failed to log similarity: {e}")
            return False
        return True
    
    def _log_selected_pairs(self, selected_img, selected_text, batch_idx=None, image_paths=None, selected_img_idx=None):
        """Log selected pairs with image identification information
        
        Args:
            selected_img: Selected image tensor
            selected_text: Selected text caption
            batch_idx: Batch index (reserved, unused)
            image_paths: List of image paths for the batch (from batch["image_paths"])
            selected_img_idx: Index of selected image within the sample's views
        
        Note:
            Identifier string includes only img_idx and full path.
        """
        try:
            # Build identifier string (img_idx and full path)
            identifier_parts = []
            if selected_img_idx is not None:
                identifier_parts.append(f"img_idx={selected_img_idx}")
            if image_paths is not None and selected_img_idx is not None:
                # image_paths shape: [B, max_views]; take the first sample
                if len(image_paths) > 0 and selected_img_idx < len(image_paths[0]):
                    img_path = image_paths[0][selected_img_idx]
                    # Use full path
                    if img_path:
                        identifier_parts.append(f"path={img_path}")
            
            identifier_str = " | ".join(identifier_parts) if identifier_parts else ""
            
            # Append identifier info to caption
            if identifier_str:
                full_caption = f"[{identifier_str}]\n{selected_text}"
            else:
                full_caption = selected_text
            
            # Denormalize image for visualization
            # Determine number of image channels
            in_chans = self.hparams.get("in_chans", 3)
            
            # Select normalization stats by channel count
            if in_chans == 1:
                # Single-channel image: use default medical image normalization stats
                mean = torch.tensor([0.482])
                std = torch.tensor([0.279])
            elif (
                self.hparams.get("vision_encoder_name") == "siglip2"
                or "siglip2" in str(self.hparams.get("hf_siglip_model_id", "")).lower()
            ):
                mean = torch.tensor([0.5, 0.5, 0.5])
                std = torch.tensor([0.5, 0.5, 0.5])
            else:
                # Multi-channel image (typically 3-channel): use ImageNet stats
                mean = torch.tensor([0.485, 0.456, 0.406])
                std = torch.tensor([0.229, 0.224, 0.225])
            
            # Ensure mean and std are on the right device
            if isinstance(selected_img, torch.Tensor):
                device = selected_img.device
                mean = mean.to(device)
                std = std.to(device)
                
                # Denormalize: unnormalized = normalized * std + mean
                # Reshape mean and std to [C, 1, 1] for broadcasting
                if selected_img.dim() == 4:  # [B, C, H, W]
                    mean = mean.view(1, -1, 1, 1)
                    std = std.view(1, -1, 1, 1)
                elif selected_img.dim() == 3:  # [C, H, W]
                    mean = mean.view(-1, 1, 1)
                    std = std.view(-1, 1, 1)
                
                # Denormalize
                denormalized_img = selected_img * std + mean
                
                # Clamp values to [0, 1] for wandb.Image
                denormalized_img = torch.clamp(denormalized_img, 0, 1)
                
                # Convert to CPU numpy array for wandb
                if denormalized_img.dim() == 4:
                    denormalized_img = denormalized_img[0]  # Take the first sample
                denormalized_img = denormalized_img.detach().cpu().numpy()
                
                # Single-channel: convert to [H, W] or [H, W, 1]
                if denormalized_img.shape[0] == 1:
                    denormalized_img = denormalized_img[0]
                else:
                    # Multi-channel: convert to [H, W, C]
                    denormalized_img = denormalized_img.transpose(1, 2, 0)
            else:
                # Already a numpy array; assume denormalized
                denormalized_img = selected_img
            
            self.logger.experiment.log(
                {
                    "selected_pairs": [
                        wandb.Image(
                            denormalized_img, caption=full_caption
                        )
                    ],
                },
                step=self.global_step,
            )
        except Exception as e:
            # Log error but don't crash training if visualization fails
            print_rank_zero(f"Warning: Failed to log selected pairs: {e}")
            return False
        return True
    
    def _log_visualizations(self, img_features, text_features, log_name='Train'):
        """Log similarity matrix"""
        try:
            with torch.no_grad():
                # 1. Log similarity matrix (consistent with actual loss computation)
                # visualize_all = True
                if False:
                    img_features_all = self.all_gather(img_features, sync_grads=False).detach()
                    text_features_all = self.all_gather(text_features, sync_grads=False).detach()
                    img_features = img_features_all.view(-1, img_features_all.shape[-1])
                    text_features = text_features_all.view(-1, text_features_all.shape[-1])
                    logits_per_image = img_features @ text_features.t()
                    similarity_matrix = logits_per_image.cpu().numpy()
                else:
                    logits_per_image = img_features @ text_features.t()
                    similarity_matrix = logits_per_image.detach().cpu().numpy()

                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(
                    similarity_matrix, cmap="viridis", aspect="auto"
                )
                ax.set_title("Image-Text Similarity Matrix")
                ax.set_xlabel("Text Index")
                ax.set_ylabel("Image Index")
                plt.colorbar(im, ax=ax)

                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                plot_img = np.asarray(buf)
                plot_img = plot_img[:, :, :3]  # Remove alpha channel
                plt.close(fig)

                self.logger.experiment.log(
                    {
                        f"{log_name}": [
                            wandb.Image(
                                plot_img, caption="Image-Text Similarity Matrix"
                            )
                        ],
                    },
                    step=self.global_step,
                )

        except Exception as e:
            # Log error but don't crash training if visualization fails
            print_rank_zero(f"Warning: Failed to log visualization: {e}")

    def configure_optimizers(self):
        """Configure optimizer with different learning rates for different modules"""
        param_groups = []
        
        # 1. Vision encoder parameters with optional layer decay (Pre-trained weights)
        if self.vision_layer_decay != 0:
            vision_groups = param_groups_layer_decay(
                self.model.vision_encoder,
                weight_decay=self.weight_decay,
                layer_decay=self.vision_layer_decay,
            )
            # Apply vision learning rate with layer decay scaling
            # Unique names per group: Lightning LearningRateMonitor forbids duplicate group names.
            for i, g in enumerate(vision_groups):
                scale = g.pop("lr_scale", 1.0)  # timm sets this
                g["lr"] = self.vision_lr * float(scale)
                g["name"] = f"vision_encoder_{i}"
            param_groups.extend(vision_groups)
        else:
            # Simple vision encoder group without layer decay
            vision_params = list(self.model.vision_encoder.parameters())
            if vision_params:
                param_groups.append({
                    'params': vision_params,
                    'lr': self.vision_lr,
                    'weight_decay': self.weight_decay,
                    'name': 'vision_encoder'
                })
        
        # 2. Text encoder parameters with optional layer decay (Pre-trained weights)
        if self.text_layer_decay != 0:
            text_groups = param_groups_layer_decay(
                self.model.text_encoder,
                weight_decay=self.weight_decay,
                layer_decay=self.text_layer_decay,
            )
            # Apply text learning rate with layer decay scaling
            for i, g in enumerate(text_groups):
                scale = g.pop("lr_scale", 1.0)  # timm sets this
                g["lr"] = self.text_lr * float(scale)
                g["name"] = f"text_encoder_{i}"
            param_groups.extend(text_groups)
        else:
            # Simple text encoder group without layer decay
            text_params = list(self.model.text_encoder.parameters())
            if text_params:
                param_groups.append({
                    'params': text_params,
                    'lr': self.text_lr,
                    'weight_decay': self.weight_decay,
                    'name': 'text_encoder'
                })
        
        # 3. Projection layers
        projection_params = []
        
        # Projection layers
        if hasattr(self.model, 'proj_t'):
            projection_params.extend(list(self.model.proj_t.parameters()))
        if hasattr(self.model, 'proj_img'):
            projection_params.extend(list(self.model.proj_img.parameters()))
            
        # Logit scale parameter (no weight decay for scalar parameters)
        if hasattr(self.model, 'logit_scale'):
            scalar_params = [self.model.logit_scale, self.model.logit_bias]
            param_groups.append({
                'params': scalar_params,
                'lr': self.fusion_lr,
                'weight_decay': 0.0,  # No weight decay for logit scale
                # 'weight_decay': self.weight_decay,  # No weight decay for logit scale
                'name': 'logit_scale'
            })
            
        if projection_params:
            param_groups.append({
                'params': projection_params,
                'lr': self.fusion_lr,
                'weight_decay': self.weight_decay,
                'name': 'projection'
            })
        
        # Create optimizer with parameter groups
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.98),  # CLIP typically uses (0.9, 0.98)
            eps=1e-6,
        )
        if self.total_steps:

            def _cosine_after_warmup(local_step: int, warmup_steps: int, total_steps: int) -> float:
                """Linear warmup followed by cosine decay, with local_step starting at 0."""
                if total_steps <= 0:
                    return 1.0
                if local_step < warmup_steps:
                    return float(local_step) / max(1, warmup_steps)
                progress = float(local_step - warmup_steps) / float(
                    max(1, total_steps - warmup_steps)
                )
                progress = max(0.0, min(1.0, progress))
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            def _encoder_lr_lambda(step: int) -> float:
                # Projection warmup phase: encoder lr=0
                if self._in_projection_warmup:
                    return 0.0

                # Keep encoder frozen until encoder_warmup_start_step is recorded
                if self._encoder_warmup_start_step is None or step < self._encoder_warmup_start_step:
                    return 0.0

                local_step = step - self._encoder_warmup_start_step
                enc_total = max(1, self.total_steps - self._encoder_warmup_start_step)
                enc_warmup_steps = int(
                    enc_total * float(self.encoder_warmup_epochs) / max(1, self.hparams.get("max_epochs", 1))
                )
                enc_warmup_steps = max(1, enc_warmup_steps)
                base_factor = _cosine_after_warmup(local_step, enc_warmup_steps, enc_total)

                # Optional encoder LR damping during topk transition window.
                if self.topk_transition_steps > 0:
                    t_start = int(self.warmup_steps)
                    t_end = int(self.warmup_steps + self.topk_transition_steps)
                    if t_start <= step < t_end:
                        s0 = max(0.0, min(1.0, float(self.transition_encoder_lr_scale)))
                        if self.transition_encoder_lr_ramp:
                            p = float(step - t_start + 1) / float(max(1, self.topk_transition_steps))
                            s = s0 + (1.0 - s0) * max(0.0, min(1.0, p))
                        else:
                            s = s0
                        base_factor *= s
                return base_factor

            def _projection_lr_lambda(step: int) -> float:
                # projection: cosine decay from step 0 without long warmup
                local_step = step
                return _cosine_after_warmup(local_step, 0, self.total_steps)

            def _logit_lr_lambda(step: int) -> float:
                # logit: short warmup (logit_warmup_steps) + cosine
                local_step = step
                warmup = max(0, self.logit_warmup_steps)
                return _cosine_after_warmup(local_step, warmup, self.total_steps)

            lr_lambdas = []
            for g in optimizer.param_groups:
                name = str(g.get("name", "") or "").lower()
                if name.startswith("vision_encoder") or name.startswith("text_encoder"):
                    lr_lambdas.append(_encoder_lr_lambda)
                elif name == "projection":
                    lr_lambdas.append(_projection_lr_lambda)
                elif name == "logit_scale":
                    lr_lambdas.append(_logit_lr_lambda)
                else:
                    # Any remaining parameter groups follow the projection schedule
                    lr_lambdas.append(_projection_lr_lambda)

            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas),
                "interval": "step",
                "frequency": 1,
            }
            return [optimizer], [scheduler]

        return optimizer

    def on_after_backward(self):
        """Log parameters without gradients after the first backward pass."""
        if getattr(self, "_unused_params_reported", False):
            return
        try:
            unused = []
            for name, p in self.model.named_parameters():
                if p.requires_grad and p.grad is None:
                    unused.append(name)
            if unused:
                # Log on rank 0 only
                from utils import print_rank_zero
                print_rank_zero(f"Unused parameters (no grad in this step, first report): {len(unused)} items")
                for n in unused[:200]:  # Print at most 200 items
                    print_rank_zero(f"  - {n}")
                if len(unused) > 200:
                    print_rank_zero(f"  ... and {len(unused) - 200} more")
            self._unused_params_reported = True
        except Exception:
            # Diagnostic helper only; does not affect training
            self._unused_params_reported = True

    def _calculate_scaled_learning_rates(self, cfg):
        """Calculate learning rates scaled by batch size using specific scaling logic"""
        scale_lr = bool(cfg.get("scale_lr", True))

        # Calculate effective batch size (B_global)
        batch_size = cfg["batch_size"]
        accumulate_grad_batches = cfg.get("accumulate_grad_batches", 1)
        B_global = batch_size * accumulate_grad_batches * int(os.environ.get("WORLD_SIZE", 1))
        if scale_lr:
            # Apply linear scaling rule for large-scale CLIP pretraining (reference batch size: 256)
            LR_v = cfg['vision_lr'] * (B_global / 256)  # Vision encoder
            LR_t = cfg['text_lr'] * (B_global / 256)  # Text encoder
        else:
            LR_v = cfg['vision_lr']
            LR_t = cfg['text_lr']
        LR_img_head = LR_v    # Image projection head: 5 × LR_v
        LR_txt_head = LR_t    # Text projection head: 5 × LR_t
        
        # For fusion layers, use the higher of the two projection head rates
        fusion_lr = max(LR_img_head, LR_txt_head)
        
        return (
            LR_v,        # base_lr (for backward compatibility)
            LR_v,        # vision_lr
            LR_t,        # text_lr  
            fusion_lr    # fusion_lr (projection heads)
        )
        
    def get_image_features(self, images):
        """Extract image features for inference"""
        return self.model.get_image_features(images)

    def get_text_features(self, texts):
        """Extract text features for inference"""
        return self.model.get_text_features(texts)

    def zero_shot_predict(self, images, texts):
        """Perform zero-shot prediction"""
        return self.model.zero_shot_predict(images, texts)
 