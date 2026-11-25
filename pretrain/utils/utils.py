import numpy as np
import torch
import torch.nn.functional as F


def create_pseudo_labels_from_attn_map(
    attn_map, img_size=224, threshold_method="quantile", quantile=0.7
):
    """
    Create pseudo labels from attention maps

    Args:
        attn_map: attention map [B, H, N + 1, N + 1]
        img_size: target image size
        threshold_method: 'mean', 'median', or 'quantile'
        quantile: quantile value if using quantile method

    Returns:
        pseudo_labels: binary masks [B, 1, H, W]
    """
    batch_size = attn_map.shape[0]
    grid_size = int(attn_map.shape[1] ** 0.5)  # e.g., 14 for 196 patches

    # Reshape to spatial grid
    attn_grid = attn_map.view(batch_size, grid_size, grid_size)  # [B, 14, 14]

    # Upsample to image size
    attn_resized = F.interpolate(
        attn_grid.unsqueeze(1).float(),
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )  # [B, 1, H, W]

    # Convert to numpy for thresholding
    pseudo_map = attn_resized.detach().cpu().numpy()

    # Apply threshold
    if threshold_method == "mean":
        threshold = np.mean(pseudo_map, axis=(2, 3), keepdims=True)  # [B, 1, 1, 1]
    elif threshold_method == "median":
        threshold = np.median(pseudo_map, axis=(2, 3), keepdims=True)
    elif threshold_method == "quantile":
        threshold = np.quantile(pseudo_map, quantile, axis=(2, 3), keepdims=True)

    # Binarize
    pseudo_map[pseudo_map > threshold] = 1
    pseudo_map[pseudo_map <= threshold] = 0

    # Convert back to tensor
    pseudo_labels = torch.from_numpy(pseudo_map).float()  # [B, 1, H, W]


    return pseudo_labels


def print_rank_zero(str):
    is_dist = torch.distributed.is_initialized()
    is_rank_zero = not is_dist or torch.distributed.get_rank() == 0
    if is_rank_zero:
        print(str)


def attention_fn(query, context, temp, query_mask=None, context_mask=None):
    """
    Args:
        query:   (B, N_r, D)   Text features
        context: (B, N_p, D)  Image features
        temp:   Temperature for attention
        query_mask: (B, N_r) Optional mask for query tokens, True indicates valid tokens
        context_mask: (B, N_p) Optional mask for context tokens, True indicates valid tokens

    Returns:
        weightedContext: (B, N_r, D)
        attn_map:        (B, N_r, N_p)
    """
    B, N_r, D = query.shape
    N_p = context.shape[1]

    # Compute similarity
    attn = torch.bmm(context, query.transpose(1, 2))  # (B, N_p, N_r)

    # Apply query mask if provided (mask out padded query tokens)
    if query_mask is not None:
        # query_mask: (B, N_r) -> (B, 1, N_r) for broadcasting
        query_mask_expanded = query_mask.unsqueeze(1)  # (B, 1, N_r)
        # Set attention to very negative value for masked positions
        attn = attn.masked_fill(~query_mask_expanded, float("-inf"))

    # Normalize over queries (word dimension)
    # distribution over text given an image
    attn = F.softmax(attn, dim=-1)

    # Apply context mask if provided (mask out padded context tokens)
    if context_mask is not None:
        # context_mask: (B, N_p) -> (B, N_p, 1) for broadcasting
        context_mask_expanded = context_mask.unsqueeze(-1)  # (B, N_p, 1)
        # Set attention to very negative value for masked positions
        attn = attn.masked_fill(~context_mask_expanded, float("-inf"))

    # Temperature scaling + normalize over image locations
    # distribution over image given a text
    attn = F.softmax(attn * temp, dim=1)  # (B, N_p, N_r)

    # Handle case where all values are masked (resulting in NaN)
    attn = torch.nan_to_num(attn, nan=0.0)

    # Compute weighted context
    # (B, D, N_p) @ (B, N_p, N_r) -> (B, D, N_r)
    weightedContext = torch.bmm(context.transpose(1, 2), attn)  # (B, D, N_r)
    weightedContext = weightedContext.transpose(1, 2)  # (B, N_r, D)

    # Attention map reshaped back: (B, N_r, N_p)
    attn_map = attn.permute(0, 2, 1).view(B, N_r, N_p)

    return weightedContext, attn_map

def adapt_checkpoint_for_new_img_size(checkpoint, new_img_size, old_img_size=224):
    """
    Adapt a checkpoint trained on one image size to work with a different image size.
    This handles positional embedding interpolation.
    """
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # Handle positional embeddings for encoder
    if 'model.encoder.pos_embed' in state_dict:
        pos_embed = state_dict['model.encoder.pos_embed']
        # Interpolate positional embeddings
        new_pos_embed = interpolate_pos_embed(pos_embed, new_img_size, old_img_size, patch_size=16)
        state_dict['model.encoder.pos_embed'] = new_pos_embed
    
    # Handle decoder positional embeddings
    if 'model.decoder_pos_embed' in state_dict:
        decoder_pos_embed = state_dict['model.decoder_pos_embed']
        new_decoder_pos_embed = interpolate_pos_embed(decoder_pos_embed, new_img_size, old_img_size, patch_size=16)
        state_dict['model.decoder_pos_embed'] = new_decoder_pos_embed
    
    return state_dict

def interpolate_pos_embed(pos_embed, new_img_size, old_img_size=224, patch_size=16):
    """Interpolate positional embeddings for different image sizes."""
    old_patches = (old_img_size // patch_size) ** 2
    new_patches = (new_img_size // patch_size) ** 2
    
    if old_patches == new_patches:
        return pos_embed
    
    # Separate class token and patch embeddings
    cls_token = pos_embed[:, :1, :]  # [1, 1, dim]
    patch_embed = pos_embed[:, 1:, :]  # [1, old_patches, dim]
    
    # Reshape and interpolate
    old_grid_size = int(old_patches ** 0.5)
    new_grid_size = int(new_patches ** 0.5)
    
    patch_embed = patch_embed.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
    patch_embed = F.interpolate(patch_embed, size=(new_grid_size, new_grid_size), mode='bicubic', align_corners=False)
    patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, new_patches, -1)
    
    return torch.cat([cls_token, patch_embed], dim=1)