import math

import torch
import torch.nn.functional as F
import torch.distributed as dist

from .utils import attention_fn


def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
    """Compute the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def siglip_loss(x, y, temperature=0.1, logit_bias=-10.0, device="cuda", pos_mask=None):
    """
    SigLIP pairwise-sigmoid loss.
    - temperature here means 1 / scale (so scale = 1/temperature). Recommended: 0.1
    - logit_bias default: -10.0
    Returns: loss, acc1, acc5
    """
    # Features are already normalized in the model - no need to normalize again
    # x = F.normalize(x, dim=-1)  # Redundant
    # y = F.normalize(y, dim=-1)  # Redundant

    tau = 1.0 / temperature
    sim = torch.einsum("i d, j d -> i j", x, y)
    logits = tau * sim + logit_bias  # [B, B]

    # Targets (identity) or many-to-many mask
    if pos_mask is None:
        target = torch.eye(logits.size(0), device=logits.device)
    else:
        target = pos_mask.to(logits.dtype).to(logits.device)

    loss_xy = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
    loss_yx = F.binary_cross_entropy_with_logits(
        logits.t(), target.t(), reduction="mean"
    )
    loss = 0.5 * (loss_xy + loss_yx)

    labels = torch.arange(x.shape[0]).to(device)
    max_top_k = min(5, x.shape[0])
    top_k = (1, min(max_top_k, 5)) if max_top_k >= 1 else (1, 1)
    i2t_acc1, i2t_acc5 = precision_at_k(logits, labels, top_k=top_k)
    t2i_acc1, t2i_acc5 = precision_at_k(logits.t(), labels, top_k=top_k)

    acc1 = (i2t_acc1 + t2i_acc1) / 2.0
    acc5 = (i2t_acc5 + t2i_acc5) / 2.0

    return loss, acc1, acc5, i2t_acc1, i2t_acc5, t2i_acc1, t2i_acc5


def siglip_loss_gather(x, x_all, y, y_all, temperature=0.1, logit_bias=-10.0, device="cuda", pos_mask=None, symmetric=True):
    """
    SigLIP pairwise-sigmoid loss with all_gather.
    Each GPU only computes its local part of the logits [B, N] instead of [N, N].
    
    Args:
        x: Local image embeddings [B, D]
        x_all: All gathered image embeddings [N, D] where N = world_size * B
        y: Local text embeddings [B, D]
        y_all: All gathered text embeddings [N, D]
        temperature: Temperature parameter (default: 0.1)
        logit_bias: Logit bias for SigLIP (default: -10.0)
        device: Device for computation
        pos_mask: Optional positive pair mask [B, N] or [N, N]
        symmetric: If True, compute both i2t and t2i losses (default: True)
    
    Returns:
        loss, acc1, acc5, i2t_acc1, i2t_acc5, t2i_acc1, t2i_acc5
    """
    B = x.shape[0]  # Local batch size
    N = x_all.shape[0]  # Global batch size (world_size * B)
    D = x.shape[1]  # Embedding dimension
    
    # Sanity checks
    assert x_all.shape == (N, D), f"x_all shape mismatch: {x_all.shape} vs ({N}, {D})"
    assert y_all.shape == (N, D), f"y_all shape mismatch: {y_all.shape} vs ({N}, {D})"
    assert y.shape == (B, D), f"y shape mismatch: {y.shape} vs ({B}, {D})"
    
    # Get rank for determining positive pairs
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    tau = 1.0 / temperature
    
    # Each GPU computes local logits: [B, N]
    # logits_img2txt = local_img @ all_txt.T / tau
    # logits_txt2img = local_txt @ all_img.T / tau
    logits_i2t = tau * (x @ y_all.t()) + logit_bias  # [B, N]
    logits_t2i = tau * (y @ x_all.t()) + logit_bias if symmetric else None  # [B, N]
    
    # Build targets / masks
    # Default: identity alignment with global indices
    if pos_mask is None:
        # For each local row b, its positive global column is at global index (rank*B + b)
        global_pos = torch.arange(B, device=device) + rank * B  # [B]
        
        # Create target matrices [B, N]
        target_i2t = torch.zeros((B, N), device=device, dtype=x.dtype)
        target_i2t.scatter_(1, global_pos.view(-1, 1), 1.0)
        
        if symmetric:
            target_t2i = torch.zeros((B, N), device=device, dtype=x.dtype)
            target_t2i.scatter_(1, global_pos.view(-1, 1), 1.0)
    else:
        # User-supplied mask.
        # Accept [B, N] (local rows) or [N, N] (global). If [N, N], slice rows for this rank.
        pm = pos_mask.to(device).to(x.dtype)
        if pm.dim() != 2:
            raise ValueError("pos_mask must be 2D [B,N] or [N,N].")
        
        if pm.shape == (B, N):
            target_i2t = pm
        elif pm.shape == (N, N):
            row_start = rank * B
            row_end = row_start + B
            target_i2t = pm[row_start:row_end, :]
        else:
            raise ValueError(f"pos_mask shape {pm.shape} incompatible with local [B,N]={B,N} or global [N,N]={N,N}.")
        
        if symmetric:
            # For symmetry with many-to-many masks: t->i simply uses the transposed global mask,
            # then slice the corresponding rows for local y anchors.
            if pm.shape == (B, N):
                target_t2i = pm  # when using identity or shared mask, this is fine
            else:
                target_t2i = pm[row_start:row_end, :]  # same slice for local y anchors
    
    # BCE losses on local blocks [B, N]
    loss_i2t = F.binary_cross_entropy_with_logits(logits_i2t, target_i2t, reduction="mean")
    if symmetric:
        loss_t2i = F.binary_cross_entropy_with_logits(logits_t2i, target_t2i, reduction="mean")
        loss = 0.5 * (loss_i2t + loss_t2i)
    else:
        loss = loss_i2t
    
    # Accuracies (use identity labels for standard retrieval metric)
    labels_global = torch.arange(B, device=device) + rank * B  # [B]
    max_top_k = min(5, N)
    top_k = (1, max_top_k if max_top_k >= 1 else 1)
    
    i2t_acc1, i2t_acc5 = precision_at_k(logits_i2t, labels_global, top_k=top_k)
    if symmetric:
        t2i_acc1, t2i_acc5 = precision_at_k(logits_t2i, labels_global, top_k=top_k)
        acc1 = (i2t_acc1 + t2i_acc1) / 2.0
        acc5 = (i2t_acc5 + t2i_acc5) / 2.0
    else:
        t2i_acc1 = t2i_acc5 = torch.tensor(0.0, device=device)
        acc1, acc5 = i2t_acc1, i2t_acc5
    
    return loss, acc1, acc5, i2t_acc1, i2t_acc5, t2i_acc1, t2i_acc5


def clip_loss(x, y, temperature=0.07, device="cuda"):
    # Features are already normalized in the model - no need to normalize again
    # x = F.normalize(x, dim=-1)  # Redundant  
    # y = F.normalize(y, dim=-1)  # Redundant

    sim = torch.einsum("i d, j d -> i j", x, y) * 1 / temperature

    labels = torch.arange(x.shape[0]).to(device)

    loss_t = F.cross_entropy(sim, labels)
    loss_i = F.cross_entropy(sim.T, labels)
    
    max_top_k = min(5, x.shape[0])
    top_k = (1, min(max_top_k, 5)) if max_top_k >= 1 else (1, 1)
    i2t_acc1, i2t_acc5 = precision_at_k(sim, labels, top_k=top_k)
    t2i_acc1, t2i_acc5 = precision_at_k(sim.T, labels, top_k=top_k)
    acc1 = (i2t_acc1 + t2i_acc1) / 2.0
    acc5 = (i2t_acc5 + t2i_acc5) / 2.0

    return (loss_t + loss_i), acc1, acc5, i2t_acc1, i2t_acc5, t2i_acc1, t2i_acc5


def local_loss(
    img_features,  # [B, (V*N), D]
    text_features,  # [B, N_r, D]
    attn_temp=4.0,
    text_agg_temp=5.0,
    final_temp=10.0,
    agg="sum",  # "sum" = logsumexp; "mean" = logsumexp - log(num_texts),
    device="cuda",
    img_mask=None,  # [B, N_p] Optional mask for valid image patches, True indicates valid patches
    text_mask=None,  # [B, N_r] Optional mask for valid texts, True indicates valid tokens
):
    B, N_p, D = img_features.shape
    B, N_r, D = text_features.shape
    assert text_features.shape[0] == B

    sims_cols = []  # will stack into [B, B]
    attention_maps = []  # list of [1, N_r, N_p]

    for i in range(B):
        # Slice texts for caption i: [1, N_r, D] -> repeat to match B images: [B, N_r, D]
        q = text_features[i].unsqueeze(0).expand(B, -1, -1)  # [B, N_r, D]

        # Get mask for caption i if provided
        q_mask = None
        if text_mask is not None:
            q_mask = text_mask[i].unsqueeze(0).expand(B, -1)  # [B, N_r]

        # Get image mask if provided
        img_mask_batch = None
        if img_mask is not None:
            img_mask_batch = img_mask  # [B, N_p]

        # Attend caption i over all images (B): returns [B, N_r, D], [B, N_r, N_p]
        ctx_i, attn_i = attention_fn(q, img_features, attn_temp, q_mask, img_mask_batch)

        # Cosine similarity per text (normalize on channel dim)
        q_n = F.normalize(q, dim=-1)  # [B, N_r, D]
        ctx_n = F.normalize(ctx_i, dim=-1)  # [B, N_r, D]
        text_sims = (q_n * ctx_n).sum(dim=-1)  # [B, N_r]

        # Apply mask to text similarities if provided
        # This masks the [B, N_r] tensor along dim=1 (text dimension)
        if q_mask is not None:
            text_sims = text_sims.masked_fill(~q_mask, float("-inf"))

        # Aggregate across texts with temperature text_agg_temp
        z = text_agg_temp * text_sims  # [B, N_r]

        if agg == "mean":
            # For mean aggregation, we need to account for the actual number of valid texts
            if q_mask is not None:
                num_valid_texts = q_mask.sum(dim=1, keepdim=True).float()  # [B, 1]
                # Avoid division by zero
                num_valid_texts = num_valid_texts.clamp(min=1.0)
                sim_vec = torch.logsumexp(z, dim=1, keepdim=True) - torch.log(
                    num_valid_texts
                )  # [B, 1]
            else:
                sim_vec = torch.logsumexp(z, dim=1, keepdim=True) - math.log(
                    N_r
                )  # [B, 1]
        elif agg == "sum":
            sim_vec = torch.logsumexp(z, dim=1, keepdim=True)  # [B, 1]
        else:
            raise ValueError(f"Invalid aggregation method: {agg}")

        # Store similarity vector (NaN handling done later in batch)
        sims_cols.append(sim_vec)  # column i of the BxB matrix

        # Save the diagonal attention map for visualization (image i vs caption i)
        attention_maps.append(
            attn_i[i].unsqueeze(0).detach().contiguous().cpu()
        )  # [1, N_r, N_p]

    # Build similarity matrix using stack (more efficient than cat)
    sim = torch.stack(sims_cols, dim=1).squeeze(-1) * final_temp  # [B, B]

    # Efficient NaN handling: replace NaN with large negative values for numerical stability
    sim = torch.nan_to_num(sim, nan=-1e4, posinf=1e4, neginf=-1e4)

    # Create binary target matrix (identity matrix like SigLIP)
    target = torch.eye(B, device=device, dtype=sim.dtype)

    # Calculate losses (NaN already handled in sim matrix)
    loss_t = F.binary_cross_entropy_with_logits(
        sim, target, reduction="mean"
    )  # text -> image
    loss_i = F.binary_cross_entropy_with_logits(
        sim.t(), target.t(), reduction="mean"
    )  # image -> text

    # Final NaN check for losses (should be rare after preprocessing)
    loss_t = torch.nan_to_num(loss_t, nan=0.0)
    loss_i = torch.nan_to_num(loss_i, nan=0.0)

    # Handle small batch sizes by adjusting top_k (simplified)
    max_top_k = min(5, B)
    top_k = (1, min(max_top_k, 5)) if max_top_k >= 1 else (1, 1)

    # Calculate accuracies with simplified error handling
    labels = torch.arange(B, device=device)  # Still need integer labels for accuracy
    i2t_acc1, i2t_acc5 = precision_at_k(sim, labels, top_k=top_k)
    t2i_acc1, t2i_acc5 = precision_at_k(sim.t(), labels, top_k=top_k)
    acc1 = (i2t_acc1 + t2i_acc1) / 2.0
    acc5 = (i2t_acc5 + t2i_acc5) / 2.0

    # Handle NaN in accuracies if they occur
    acc1 = torch.nan_to_num(acc1, nan=0.0)
    acc5 = torch.nan_to_num(acc5, nan=0.0)

    return (loss_t + loss_i), acc1, acc5, attention_maps


# Vectorized for speed but consumes more memory
def local_loss_vectorized(
    img_features,  # [B, (V*N), D]
    text_features,  # [B, N_r, D]
    attn_temp=4.0,
    text_agg_temp=5.0,
    final_temp=10.0,
    agg="sum",  # "sum" = logsumexp; "mean" = logsumexp - log(num_texts),
    device="cuda",
    img_mask=None,  # [B, N_p] Optional mask for valid image patches, True indicates valid patches
    text_mask=None,  # [B, N_r] Optional mask for valid texts, True indicates valid tokens
):
    B, N_p, D = img_features.shape
    B, N_r, D = text_features.shape
    assert text_features.shape[0] == B

    # Vectorized computation: process all text-image pairs at once

    # Expand text_features: [B, N_r, D] -> [B, 1, N_r, D] -> [B, B, N_r, D]
    text_expanded = text_features.unsqueeze(1).expand(B, B, N_r, D)

    # Expand img_features: [B, N_p, D] -> [1, B, N_p, D] -> [B, B, N_p, D]
    img_expanded = img_features.unsqueeze(0).expand(B, B, N_p, D)

    # Reshape for batch processing: [B*B, N_r, D] and [B*B, N_p, D]
    text_batch = text_expanded.reshape(B * B, N_r, D)
    img_batch = img_expanded.reshape(B * B, N_p, D)

    # Handle text masks
    text_mask_batch = None
    if text_mask is not None:
        # Expand text_mask: [B, N_r] -> [B, 1, N_r] -> [B, B, N_r] -> [B*B, N_r]
        text_mask_expanded = text_mask.unsqueeze(1).expand(B, B, N_r)
        text_mask_batch = text_mask_expanded.reshape(B * B, N_r)

    # Handle image masks
    img_mask_batch = None
    if img_mask is not None:
        # Expand img_mask: [B, N_p] -> [1, B, N_p] -> [B, B, N_p] -> [B*B, N_p]
        img_mask_expanded = img_mask.unsqueeze(0).expand(B, B, N_p)
        img_mask_batch = img_mask_expanded.reshape(B * B, N_p)

    # Process all pairs at once
    ctx_batch, attn_batch = attention_fn(
        text_batch, img_batch, attn_temp, text_mask_batch, img_mask_batch
    )

    # Reshape back: [B*B, N_r, D] -> [B, B, N_r, D]
    ctx_all = ctx_batch.reshape(B, B, N_r, D)
    attn_all = attn_batch.reshape(B, B, N_r, N_p)

    # Compute cosine similarities for all pairs
    text_norm = F.normalize(text_expanded, dim=-1)  # [B, B, N_r, D]
    ctx_norm = F.normalize(ctx_all, dim=-1)  # [B, B, N_r, D]
    text_sims_all = (text_norm * ctx_norm).sum(dim=-1)  # [B, B, N_r]

    # Apply text masking if provided
    if text_mask is not None:
        text_mask_expanded = text_mask.unsqueeze(1).expand(B, B, N_r)
        text_sims_all = text_sims_all.masked_fill(~text_mask_expanded, float("-inf"))

    # Aggregate across texts: [B, B, N_r] -> [B, B]
    z_all = text_agg_temp * text_sims_all  # [B, B, N_r]

    if agg == "mean":
        # For mean aggregation, we need to account for the actual number of valid texts
        if text_mask is not None:
            num_valid_texts = text_mask_expanded.sum(
                dim=-1, keepdim=True
            ).float()  # [B, B, 1]
            # Avoid division by zero
            num_valid_texts = num_valid_texts.clamp(min=1.0)
            sim = torch.logsumexp(z_all, dim=-1, keepdim=True) - torch.log(
                num_valid_texts
            )  # [B, B, 1]
            sim = sim.squeeze(-1)  # [B, B]
        else:
            sim = torch.logsumexp(z_all, dim=-1) - math.log(N_r)  # [B, B]
    elif agg == "sum":
        sim = torch.logsumexp(z_all, dim=-1)  # [B, B]
    else:
        raise ValueError(f"Invalid aggregation method: {agg}")

    # Extract diagonal attention maps for visualization (optimized with stack)
    # diagonal_indices = torch.arange(B, device=device)
    # attention_maps_tensor = attn_all[
    #     diagonal_indices, diagonal_indices
    # ]  # [B, N_r, N_p]
    # attention_maps = [
    #     attention_maps_tensor[i].unsqueeze(0).detach().contiguous().cpu()
    #     for i in range(B)
    # ]

    # Scale with final_temp and handle NaN efficiently
    sim = sim * final_temp  # [B, B]
    sim = torch.nan_to_num(sim, nan=-1e4, posinf=1e4, neginf=-1e4)

    # Create binary target matrix (identity matrix like SigLIP)
    target = torch.eye(B, device=device, dtype=sim.dtype)

    # Calculate losses (NaN already handled in sim matrix)
    loss_t = F.binary_cross_entropy_with_logits(
        sim, target, reduction="mean"
    )  # text -> image
    loss_i = F.binary_cross_entropy_with_logits(
        sim.t(), target.t(), reduction="mean"
    )  # image -> text

    # Final NaN check for losses (should be rare after preprocessing)
    loss_t = torch.nan_to_num(loss_t, nan=0.0)
    loss_i = torch.nan_to_num(loss_i, nan=0.0)

    # Handle small batch sizes by adjusting top_k (simplified)
    max_top_k = min(5, B)
    top_k = (1, min(max_top_k, 5)) if max_top_k >= 1 else (1, 1)

    # Calculate accuracies with simplified error handling
    labels = torch.arange(B, device=device)  # Still need integer labels for accuracy
    i2t_acc1, i2t_acc5 = precision_at_k(sim, labels, top_k=top_k)
    t2i_acc1, t2i_acc5 = precision_at_k(sim.t(), labels, top_k=top_k)
    acc1 = (i2t_acc1 + t2i_acc1) / 2.0
    acc5 = (i2t_acc5 + t2i_acc5) / 2.0

    # Handle NaN in accuracies if they occur
    acc1 = torch.nan_to_num(acc1, nan=0.0)
    acc5 = torch.nan_to_num(acc5, nan=0.0)

    return (loss_t + loss_i), acc1, acc5, None # attention_maps


def local_loss_gather(
    img_features,  # [B, (V*N), D] - local image features
    img_features_all,  # [N, (V*N), D] - all gathered image features where N = world_size * B
    text_features,  # [B, N_r, D] - local text features
    text_features_all,  # [N, N_r, D] - all gathered text features
    attn_temp=4.0,
    text_agg_temp=5.0,
    final_temp=10.0,
    agg="sum",  # "sum" = logsumexp; "mean" = logsumexp - log(num_texts),
    device="cuda",
    img_mask=None,  # [B, N_p] Optional mask for valid image patches, True indicates valid patches
    text_mask=None,  # [B, N_r] Optional mask for valid texts, True indicates valid tokens
    img_mask_all=None,  # [N, N_p] Optional mask for all gathered image patches
    text_mask_all=None,  # [N, N_r] Optional mask for all gathered texts
):
    """
    Local computation version of local_loss with all_gather.
    Each GPU only computes its local part of the logits [B, N] instead of [N, N].
    
    Args:
        img_features: Local image embeddings [B, N_p, D]
        img_features_all: All gathered image embeddings [N, N_p, D] where N = world_size * B
        text_features: Local text embeddings [B, N_r, D]
        text_features_all: All gathered text embeddings [N, N_r, D]
        attn_temp: Attention temperature parameter
        text_agg_temp: Text aggregation temperature parameter
        final_temp: Final temperature scaling
        agg: Aggregation method ("sum" or "mean")
        device: Device for computation
        img_mask: Optional mask for valid local image patches [B, N_p]
        text_mask: Optional mask for valid local texts [B, N_r]
        img_mask_all: Optional mask for valid all gathered image patches [N, N_p]
        text_mask_all: Optional mask for valid all gathered texts [N, N_r]
    
    Returns:
        loss, acc1, acc5, attention_maps
    """
    B, N_p, D = img_features.shape
    N, N_p_all, D_all = img_features_all.shape
    B_txt, N_r, D_txt = text_features.shape
    N_txt, N_r_all, D_txt_all = text_features_all.shape
    
    # Sanity checks
    assert B == B_txt, f"Batch size mismatch: img {B} vs text {B_txt}"
    assert D == D_all, f"Image embedding dim mismatch: local {D} vs all {D_all}"
    assert D_txt == D_txt_all, f"Text embedding dim mismatch: local {D_txt} vs all {D_txt_all}"
    assert N_p == N_p_all, f"Image patch count mismatch: local {N_p} vs all {N_p_all}"
    assert N_r == N_r_all, f"Text token count mismatch: local {N_r} vs all {N_r_all}"
    assert N == N_txt, f"Global batch size mismatch: img {N} vs text {N_txt}"
    
    # Get rank for determining positive pairs
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    # Each GPU computes local logits: [B, N]
    # For each local text, compute similarity with all images
    sims_cols = []  # will stack into [B, N]
    attention_maps = []  # list of [1, N_r, N_p]
    
    for i in range(B):
        # Slice texts for caption i: [1, N_r, D] -> repeat to match N images: [N, N_r, D]
        q = text_features[i].unsqueeze(0).expand(N, -1, -1)  # [N, N_r, D]
        
        # Get mask for caption i if provided
        q_mask = None
        if text_mask is not None:
            q_mask = text_mask[i].unsqueeze(0).expand(N, -1)  # [N, N_r]
        elif text_mask_all is not None:
            # Use the corresponding row from text_mask_all
            q_mask = text_mask_all  # [N, N_r]
        
        # Get image mask if provided
        img_mask_batch = None
        if img_mask_all is not None:
            img_mask_batch = img_mask_all  # [N, N_p]
        elif img_mask is not None:
            # For local computation, we only have local image mask
            # This is a limitation - we assume all gathered images have same mask structure
            img_mask_batch = img_mask.unsqueeze(0).expand(N, -1)  # [N, N_p]
        
        # Attend caption i over all images (N): returns [N, N_r, D], [N, N_r, N_p]
        ctx_i, attn_i = attention_fn(q, img_features_all, attn_temp, q_mask, img_mask_batch)
        
        # Cosine similarity per text (normalize on channel dim)
        q_n = F.normalize(q, dim=-1)  # [N, N_r, D]
        ctx_n = F.normalize(ctx_i, dim=-1)  # [N, N_r, D]
        text_sims = (q_n * ctx_n).sum(dim=-1)  # [N, N_r]
        
        # Apply mask to text similarities if provided
        if q_mask is not None:
            text_sims = text_sims.masked_fill(~q_mask, float("-inf"))
        
        # Aggregate across texts with temperature text_agg_temp
        z = text_agg_temp * text_sims  # [N, N_r]
        
        if agg == "mean":
            # For mean aggregation, we need to account for the actual number of valid texts
            if q_mask is not None:
                num_valid_texts = q_mask.sum(dim=1, keepdim=True).float()  # [N, 1]
                # Avoid division by zero
                num_valid_texts = num_valid_texts.clamp(min=1.0)
                sim_vec = torch.logsumexp(z, dim=1, keepdim=True) - torch.log(
                    num_valid_texts
                )  # [N, 1]
            else:
                sim_vec = torch.logsumexp(z, dim=1, keepdim=True) - math.log(
                    N_r
                )  # [N, 1]
        elif agg == "sum":
            sim_vec = torch.logsumexp(z, dim=1, keepdim=True)  # [N, 1]
        else:
            raise ValueError(f"Invalid aggregation method: {agg}")
        
        # Store similarity vector (column i of the BxN matrix)
        sims_cols.append(sim_vec.squeeze(-1))  # [N]
        
        # Save the attention map for the positive pair (image i vs caption i)
        # The positive image index is at rank * B + i
        pos_img_idx = rank * B + i
        if pos_img_idx < N:
            attention_maps.append(
                attn_i[pos_img_idx].unsqueeze(0).detach().contiguous().cpu()
            )  # [1, N_r, N_p]
    
    # Build similarity matrix: [B, N]
    sim = torch.stack(sims_cols, dim=0) * final_temp  # [B, N]
    
    # Efficient NaN handling: replace NaN with large negative values for numerical stability
    sim = torch.nan_to_num(sim, nan=-1e4, posinf=1e4, neginf=-1e4)
    
    # Create binary target matrix [B, N] - only positive pairs are (local_text_i, global_img_{rank*B + i})
    target = torch.zeros((B, N), device=device, dtype=sim.dtype)
    global_pos = torch.arange(B, device=device) + rank * B  # [B]
    target.scatter_(1, global_pos.view(-1, 1), 1.0)
    
    # Calculate losses (NaN already handled in sim matrix)
    loss_t = F.binary_cross_entropy_with_logits(
        sim, target, reduction="mean"
    )  # text -> image
    
    # For symmetric loss, compute image -> text direction
    # Each local image attends to all texts
    sims_rows = []  # will stack into [B, N] for image->text direction
    
    for j in range(B):
        # Slice images for image j: [1, N_p, D] -> repeat to match N texts: [N, N_p, D]
        img_j = img_features[j].unsqueeze(0).expand(N, -1, -1)  # [N, N_p, D]
        
        # Get mask for image j if provided
        img_mask_j = None
        if img_mask is not None:
            img_mask_j = img_mask[j].unsqueeze(0).expand(N, -1)  # [N, N_p]
        elif img_mask_all is not None:
            img_mask_j = img_mask_all  # [N, N_p]
        
        # Get text mask if provided
        text_mask_batch = None
        if text_mask_all is not None:
            text_mask_batch = text_mask_all  # [N, N_r]
        elif text_mask is not None:
            # For local computation, we only have local text mask
            # This is a limitation - we assume all gathered texts have same mask structure
            text_mask_batch = text_mask.unsqueeze(0).expand(N, -1)  # [N, N_r]
        
        # Attend image j over all texts (N): returns [N, N_p, D], [N, N_p, N_r]
        ctx_j, attn_j = attention_fn(img_j, text_features_all, attn_temp, img_mask_j, text_mask_batch)
        
        # Cosine similarity per image patch (normalize on channel dim)
        img_n = F.normalize(img_j, dim=-1)  # [N, N_p, D]
        ctx_n = F.normalize(ctx_j, dim=-1)  # [N, N_p, D]
        img_sims = (img_n * ctx_n).sum(dim=-1)  # [N, N_p]
        
        # Apply mask to image similarities if provided
        if img_mask_j is not None:
            img_sims = img_sims.masked_fill(~img_mask_j, float("-inf"))
        
        # Aggregate across image patches with temperature text_agg_temp
        z = text_agg_temp * img_sims  # [N, N_p]
        
        if agg == "mean":
            # For mean aggregation, we need to account for the actual number of valid patches
            if img_mask_j is not None:
                num_valid_patches = img_mask_j.sum(dim=1, keepdim=True).float()  # [N, 1]
                # Avoid division by zero
                num_valid_patches = num_valid_patches.clamp(min=1.0)
                sim_vec = torch.logsumexp(z, dim=1, keepdim=True) - torch.log(
                    num_valid_patches
                )  # [N, 1]
            else:
                sim_vec = torch.logsumexp(z, dim=1, keepdim=True) - math.log(
                    N_p
                )  # [N, 1]
        elif agg == "sum":
            sim_vec = torch.logsumexp(z, dim=1, keepdim=True)  # [N, 1]
        else:
            raise ValueError(f"Invalid aggregation method: {agg}")
        
        # Store similarity vector (row j of the BxN matrix)
        sims_rows.append(sim_vec.squeeze(-1))  # [N]
    
    # Build similarity matrix for image->text direction: [B, N]
    sim_i2t = torch.stack(sims_rows, dim=0) * final_temp  # [B, N]
    
    # Efficient NaN handling for image->text similarities
    sim_i2t = torch.nan_to_num(sim_i2t, nan=-1e4, posinf=1e4, neginf=-1e4)
    
    # Create binary target matrix for image->text direction [B, N]
    target_i2t = torch.zeros((B, N), device=device, dtype=sim_i2t.dtype)
    target_i2t.scatter_(1, global_pos.view(-1, 1), 1.0)
    
    # Calculate image->text loss
    loss_i = F.binary_cross_entropy_with_logits(
        sim_i2t, target_i2t, reduction="mean"
    )  # image -> text
    
    # Final NaN check for losses (should be rare after preprocessing)
    loss_t = torch.nan_to_num(loss_t, nan=0.0)
    loss_i = torch.nan_to_num(loss_i, nan=0.0)
    
    # Handle small batch sizes by adjusting top_k
    max_top_k = min(5, N)
    top_k = (1, max_top_k) if max_top_k >= 1 else (1, 1)
    
    # Calculate accuracies with global labels
    labels_global = torch.arange(B, device=device) + rank * B  # [B]
    i2t_acc1, i2t_acc5 = precision_at_k(sim, labels_global, top_k=top_k)
    t2i_acc1, t2i_acc5 = precision_at_k(sim_i2t, labels_global, top_k=top_k)
    
    acc1 = (i2t_acc1 + t2i_acc1) / 2.0
    acc5 = (i2t_acc5 + t2i_acc5) / 2.0
    
    # Handle NaN in accuracies if they occur
    acc1 = torch.nan_to_num(acc1, nan=0.0)
    acc5 = torch.nan_to_num(acc5, nan=0.0)
    
    return (loss_t + loss_i), acc1, acc5, attention_maps


def bceDiceLoss(pred, target, train=True):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    smooth = 1e-5
    num = target.size(0)

    # Convert logits to probabilities for Dice calculation
    pred_sigmoid = torch.sigmoid(pred)
    pred_flat = pred_sigmoid.view(num, -1)
    target_flat = target.view(num, -1)

    intersection = pred_flat * target_flat
    dice = (2.0 * intersection.sum(1) + smooth) / (
        pred_flat.sum(1) + target_flat.sum(1) + smooth
    )
    dice = 1 - dice.sum() / num

    if train:
        return dice + 0.2 * bce
    return dice


def siglip_loss_ring(img_emb, txt_emb, temperature=0.07):
    """Memory-efficient SigLIP loss with proper accuracy calculation"""
    B, D = img_emb.shape
    device = img_emb.device
    
    # Features are already normalized in the model - no need to normalize again
    # img = F.normalize(img_emb, dim=-1)  # Redundant
    # txt = F.normalize(txt_emb, dim=-1)  # Redundant
    img = img_emb
    txt = txt_emb
    
    # Compute similarities
    logits = (img @ txt.T) / temperature  # [B, B]
    
    # Single-process case
    if not dist.is_initialized():
        # Identity labels for single process
        labels = torch.eye(B, device=device, dtype=logits.dtype)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
        
        # Compute accuracies using precision_at_k
        with torch.no_grad():
            acc_labels = torch.arange(B, device=device)
            max_top_k = min(5, B)
            top_k = (1, max_top_k) if max_top_k > 1 else (1, 1)
            
            i2t_acc1, i2t_acc5 = precision_at_k(logits, acc_labels, top_k=top_k)
            t2i_acc1, t2i_acc5 = precision_at_k(logits.T, acc_labels, top_k=top_k)
            
            acc1 = (i2t_acc1 + t2i_acc1) / 2.0
            acc5 = (i2t_acc5 + t2i_acc5) / 2.0
            
        return loss, acc1, acc5
    
    # Distributed case - gather only text embeddings for memory efficiency
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # Efficient all_gather using pre-allocated tensors
    txt_gathered = torch.empty(world_size, B, D, device=device, dtype=txt.dtype)
    dist.all_gather(list(txt_gathered.unbind(0)), txt)
    txt_all = txt_gathered.view(world_size * B, D)  # [world_size * B, D]
    
    # Compute global similarities
    global_logits = (img @ txt_all.T) / temperature  # [B, world_size * B]
    
    # Create binary labels - only positive pairs are (local_img_i, global_txt_{rank*B + i})
    global_txt_indices = rank * B + torch.arange(B, device=device)
    labels = torch.zeros_like(global_logits)  # [B, world_size * B]
    labels[torch.arange(B, device=device), global_txt_indices] = 1.0
    
    # Compute loss
    loss = F.binary_cross_entropy_with_logits(global_logits, labels, reduction="mean")
    
    # Compute accuracies efficiently
    with torch.no_grad():
        max_top_k = min(5, world_size * B)
        top_k = (1, max_top_k) if max_top_k > 1 else (1, 1)
        
        # I2T: Each image finds its text among all gathered texts
        i2t_acc1, i2t_acc5 = precision_at_k(global_logits, global_txt_indices, top_k=top_k)
        
        # T2I: Each local text finds its image among local images
        # Use only the relevant slice of transposed logits
        t2i_logits = global_logits.T[rank * B:(rank + 1) * B]  # [B, B]
        local_labels = torch.arange(B, device=device)
        local_top_k = (1, min(5, B)) if B > 1 else (1, 1)
        t2i_acc1, t2i_acc5 = precision_at_k(t2i_logits, local_labels, top_k=local_top_k)
        
        # Average both directions
        acc1 = (i2t_acc1 + t2i_acc1) / 2.0
        acc5 = (i2t_acc5 + t2i_acc5) / 2.0
    
    return loss, acc1, acc5
