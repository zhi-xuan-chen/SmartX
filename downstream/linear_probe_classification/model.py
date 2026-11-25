"""Small linear-probe model used on cached SmartX image embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(int(feature_dim), int(num_classes))
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(features))
