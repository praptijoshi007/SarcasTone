"""Context-aware representation for multi-turn dialogues.

MUStARD utterances come with preceding context turns. Member 1's model
already extracts a context-pooled vector [768]. This module:

  1. Accepts the member1 context vector (or a list of raw context strings
     when the member1 checkpoint is available).
  2. Applies a learnable linear transform to distinguish it from the
     target utterance embedding.
  3. Combines it with the target via a learned gating scalar per sample.

The final context-enriched representation is added to the classifier input.
Samples with no context (has_context=False) receive zero contribution.

Design note: MUStARD++ (text-only) has no audio. The context encoder
operates purely on text embeddings and is modality-agnostic.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    """Integrate dialogue context into the target utterance representation.

    Args:
        hidden_dim:  width of both context and target vectors (must match).
        dropout:     dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Learn a context-specific transform (distinguish context from target)
        self.context_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

        # Gating: how much context to inject (per sample)
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        target: torch.Tensor,        # [B, hidden_dim]  target utterance pool
        context: torch.Tensor,       # [B, hidden_dim]  context pool from member1
        has_context: torch.Tensor,   # [B] bool
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            enriched:    [B, hidden_dim]  context-enriched target
            gate:        [B]              context injection weight per sample
        """
        ctx_t = self.context_transform(context)  # [B, hidden_dim]
        ctx_t = self.dropout(ctx_t)

        gate_input = torch.cat([target, ctx_t], dim=-1)  # [B, 2*hidden_dim]
        gate = self.context_gate(gate_input).squeeze(-1)  # [B]

        # Zero out gate where there is no context
        gate = gate * has_context.float()

        enriched = target + gate.unsqueeze(-1) * ctx_t  # [B, hidden_dim]

        return {
            "enriched": enriched,
            "gate": gate,
        }
