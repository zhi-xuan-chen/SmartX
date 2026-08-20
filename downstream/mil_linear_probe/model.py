"""Lightweight class-specific CLAM-style pooling for frozen image features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLAMLinearProbe(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        *,
        hidden_dim: int = 512,
        attention_dim: int = 256,
        dropout: float = 0.25,
        gated: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.gated = bool(gated)
        self.feature_projection = nn.Sequential(
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.attention_v = nn.Linear(int(hidden_dim), int(attention_dim))
        self.attention_u = nn.Linear(int(hidden_dim), int(attention_dim)) if gated else None
        self.attention_weights = nn.Linear(int(attention_dim), self.num_classes)
        self.classifiers = nn.ModuleList(
            nn.Linear(int(hidden_dim), 1) for _ in range(self.num_classes)
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError(f"Expected [batch, views, features], got {tuple(features.shape)}")
        batch, views, _ = features.shape
        if mask is None:
            mask = torch.ones((batch, views), dtype=torch.bool, device=features.device)
        if mask.shape != (batch, views):
            raise ValueError(f"Expected mask shape {(batch, views)}, got {tuple(mask.shape)}")
        if not torch.all(mask.any(dim=1)):
            raise ValueError("Every MIL bag must contain at least one valid view")

        hidden = self.feature_projection(features)
        attention = torch.tanh(self.attention_v(hidden))
        if self.attention_u is not None:
            attention = attention * torch.sigmoid(self.attention_u(hidden))
        attention = self.attention_weights(attention).permute(0, 2, 1)
        attention = attention.masked_fill(~mask.unsqueeze(1), torch.finfo(attention.dtype).min)
        attention = F.softmax(attention, dim=-1)
        pooled = torch.einsum("bcv,bvh->bch", attention, hidden)
        logits = torch.cat(
            [classifier(pooled[:, idx]) for idx, classifier in enumerate(self.classifiers)],
            dim=1,
        )
        return logits, attention
