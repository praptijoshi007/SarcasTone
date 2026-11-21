"""Cross-Modal Attention Module for SarcasTone — member3_fusion.

Public API
----------
CrossModalAttentionBlock  – single bidirectional cross-attention block
                            (text_dim, audio_dim may differ; projects to out_dim)
StackedCrossModalAttention – N stacked CrossModalAttentionBlocks, used by
                             SarcasToneFusionModel.

BiDirectionalCrossAttention – legacy alias kept for backward compatibility.

Both output a (text_out, audio_out) tuple for direct use in fusion_model.py:
    t_out, a_out = self.cross_attn(t_seq, a_seq)

Supports:
  - Single-token pooled inputs  [B, 1, D]
  - Full sequence inputs        [B, T, D]  (variable T)
  - Different text_dim / audio_dim          (input projections align them)
  - Optional key-padding masks for padded sequences
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Single cross-attention layer (one direction)
# ---------------------------------------------------------------------------

class _CrossAttentionLayer(nn.Module):
    """One direction of cross-modal attention: query from modality A, kv from B.

    Pre-LN transformer block with residual connection + feed-forward.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,                     # [B, Sq, D]
        key_value: torch.Tensor,                 # [B, Skv, D]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, Skv] bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output:          [B, Sq, D]
            attn_weights:    [B, n_heads, Sq, Skv]  (averaged across heads)
        """
        # Pre-LN cross-attention
        q_norm = self.norm1(query)
        attn_out, attn_w = self.attn(
            query=q_norm,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
        )
        query = query + self.drop(attn_out)
        # Pre-LN feed-forward
        query = query + self.ffn(self.norm2(query))
        return query, attn_w


# ---------------------------------------------------------------------------
# CrossModalAttentionBlock — bidirectional, possibly different input dims
# ---------------------------------------------------------------------------

class CrossModalAttentionBlock(nn.Module):
    """Single bidirectional cross-modal attention block.

    Both text→audio and audio→text attention are computed within one block.
    If text_dim ≠ out_dim or audio_dim ≠ out_dim, linear input projections
    align the representations before attention.

    Args:
        text_dim:  Dimension of text features (input).
        audio_dim: Dimension of audio features (input).
        out_dim:   Dimension used *inside* attention and returned as output.
        num_heads: Number of attention heads (must divide out_dim).
        dropout:   Dropout probability.

    Forward:
        text_seq:  [B, St, text_dim]
        audio_seq: [B, Sa, audio_dim]
        text_mask:  [B, St] bool (True = padded) or None
        audio_mask: [B, Sa] bool or None

    Returns:
        (text_out [B, St, out_dim], audio_out [B, Sa, out_dim])
    """

    def __init__(
        self,
        text_dim:  int,
        audio_dim: int,
        out_dim:   int,
        num_heads: int = 8,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim

        # Input projections (identity-like when dims match)
        self.text_in_proj  = (
            nn.Linear(text_dim,  out_dim) if text_dim  != out_dim else nn.Identity()
        )
        self.audio_in_proj = (
            nn.Linear(audio_dim, out_dim) if audio_dim != out_dim else nn.Identity()
        )

        # Bidirectional cross-attention layers
        self.t2a = _CrossAttentionLayer(out_dim, num_heads, dropout)  # text ← audio
        self.a2t = _CrossAttentionLayer(out_dim, num_heads, dropout)  # audio ← text

        # Output norms
        self.text_out_norm  = nn.LayerNorm(out_dim)
        self.audio_out_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        text_seq:   torch.Tensor,
        audio_seq:  torch.Tensor,
        text_mask:  Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Project to common dimension
        t = self.text_in_proj(text_seq)    # [B, St, out_dim]
        a = self.audio_in_proj(audio_seq)  # [B, Sa, out_dim]

        # Text attends to audio; audio attends to text
        t_out, _ = self.t2a(query=t, key_value=a, key_padding_mask=audio_mask)
        a_out, _ = self.a2t(query=a, key_value=t, key_padding_mask=text_mask)

        return self.text_out_norm(t_out), self.audio_out_norm(a_out)


# ---------------------------------------------------------------------------
# StackedCrossModalAttention — N stacked blocks
# ---------------------------------------------------------------------------

class StackedCrossModalAttention(nn.Module):
    """N stacked CrossModalAttentionBlocks.

    Used by SarcasToneFusionModel as ``self.cross_attn``.

    Forward signature:
        text_seq  [B, St, text_dim]
        audio_seq [B, Sa, audio_dim]
        → (text_out [B, St, out_dim], audio_out [B, Sa, out_dim])

    Compatible with single-token pooled inputs [B, 1, D] and full sequences.
    """

    def __init__(
        self,
        text_dim:   int,
        audio_dim:  int,
        out_dim:    int,
        num_layers: int = 2,
        num_heads:  int = 8,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.out_dim    = out_dim
        self.num_layers = num_layers

        # First block: handles heterogeneous input dims
        # Subsequent blocks: out_dim → out_dim (homogeneous)
        blocks = []
        in_t, in_a = text_dim, audio_dim
        for _ in range(num_layers):
            blocks.append(
                CrossModalAttentionBlock(
                    text_dim=in_t, audio_dim=in_a,
                    out_dim=out_dim, num_heads=num_heads, dropout=dropout,
                )
            )
            in_t = out_dim
            in_a = out_dim
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        text_seq:   torch.Tensor,
        audio_seq:  torch.Tensor,
        text_mask:  Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_seq:   [B, St, text_dim]
            audio_seq:  [B, Sa, audio_dim]
            text_mask:  [B, St] bool or None
            audio_mask: [B, Sa] bool or None

        Returns:
            (text_out [B, St, out_dim], audio_out [B, Sa, out_dim])
        """
        t, a = text_seq, audio_seq
        for block in self.blocks:
            t, a = block(t, a, text_mask=text_mask, audio_mask=audio_mask)
        return t, a


# ---------------------------------------------------------------------------
# Legacy alias — keeps old code that imports BiDirectionalCrossAttention alive
# ---------------------------------------------------------------------------

class BiDirectionalCrossAttention(nn.Module):
    """Legacy wrapper around StackedCrossModalAttention.

    Kept for backward compatibility with any code that imports this name.
    Returns a dict with keys matching the original API.
    """

    def __init__(
        self,
        d_model:   int = 256,
        n_heads:   int = 8,
        n_layers:  int = 2,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()
        self._stack = StackedCrossModalAttention(
            text_dim=d_model, audio_dim=d_model, out_dim=d_model,
            num_layers=n_layers, num_heads=n_heads, dropout=dropout,
        )

    def forward(
        self,
        text_features:  torch.Tensor,
        audio_features: torch.Tensor,
        text_padding_mask:  Optional[torch.Tensor] = None,
        audio_padding_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            text_features:  [B, St, d_model]
            audio_features: [B, Sa, d_model]

        Returns dict:
            text_attended:         [B, St, d_model]
            audio_attended:        [B, Sa, d_model]
            text_to_audio_weights: []  (not retained in new impl)
            audio_to_text_weights: []
        """
        t_out, a_out = self._stack(
            text_features, audio_features,
            text_mask=text_padding_mask,
            audio_mask=audio_padding_mask,
        )
        return {
            "text_attended":         t_out,
            "audio_attended":        a_out,
            "text_to_audio_weights": [],
            "audio_to_text_weights": [],
        }
