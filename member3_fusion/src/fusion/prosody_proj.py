"""Prosody feature projection.

Projects the PROSODY_DIM (32) handcrafted prosody vector and the full
acoustic feature vector (up to 116 dims from member2 handoff) into the
shared fusion dimension.

Both projections are independent 2-layer MLPs with batch normalisation.
The prosody projection output feeds into the PSID module's audio-side.
The acoustic-full projection is concatenated with the main classifier input
to give a richer acoustic signal.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from shared.constants import PROSODY_DIM


class ProsodyProjection(nn.Module):
    """Project hand-crafted prosody features → fusion space.

    Args:
        prosody_dim:  number of input prosody features (PROSODY_DIM).
        out_dim:      output projection width.
        dropout:      dropout rate.
    """

    def __init__(
        self,
        prosody_dim: int = PROSODY_DIM,
        out_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.prosody_dim = prosody_dim
        self.out_dim = out_dim

        self.proj = nn.Sequential(
            nn.Linear(prosody_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, prosody: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prosody: [B, prosody_dim]
        Returns:
            [B, out_dim]
        """
        return self.proj(prosody)


class AcousticFullProjection(nn.Module):
    """Project full acoustic feature vector (116-dim) → fusion space.

    This is a lightweight MLP that provides additional acoustic texture
    to the classifier beyond what Wav2Vec2 captures.

    Args:
        acoustic_dim: number of input features (default 116 from member2).
        out_dim:      output projection width.
        dropout:      dropout.
    """

    def __init__(
        self,
        acoustic_dim: int = 116,
        out_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.acoustic_dim = acoustic_dim
        self.out_dim = out_dim

        self.proj = nn.Sequential(
            nn.Linear(acoustic_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, acoustic_full: torch.Tensor) -> torch.Tensor:
        """
        Args:
            acoustic_full: [B, acoustic_dim]
        Returns:
            [B, out_dim]
        """
        return self.proj(acoustic_full)
