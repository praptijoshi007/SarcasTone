"""Small neural pieces shared by the audio and fusion modules."""

from __future__ import annotations

import torch
from torch import nn


class AttentivePooling(nn.Module):
    """Masked attention pool. A fully padded row returns a zero vector."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # hidden: [B, T, D], mask: [B, T] True where the step is real.
        scores = self.score(hidden).squeeze(-1)
        scores = scores.masked_fill(~mask.bool(), -1e4)
        weights = torch.softmax(scores, dim=-1)
        valid = mask.bool().any(dim=-1, keepdim=True)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        pooled = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
        return pooled * valid.to(pooled.dtype)
