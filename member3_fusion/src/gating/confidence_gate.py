"""Adaptive Confidence-Gated Fusion.

Research contribution B of SarcasTone.

Theory
------
When one modality is unreliable (poor audio quality, low transcription
confidence, missing audio), the model should automatically downweight it.

The gating network learns:

    signal = concat(text_pool, audio_pool, quality_vec, [audio_avail, quality_ok])
    g_text  = σ( W_t · signal + b_t )   ∈ (0, 1)
    g_audio = σ( W_a · signal + b_a )   ∈ (0, 1)

    # Reliability priors multiply the learned gates
    r_text  = transcription_confidence (1.0 if gold, else ASR confidence)
    r_audio = composite quality score from QUALITY_DIM vector

    effective_text  = g_text  · r_text
    effective_audio = g_audio · r_audio

    # Normalise so they form a soft partition of unity (optional)
    w_text  = effective_text  / (effective_text + effective_audio + ε)
    w_audio = effective_audio / (effective_text + effective_audio + ε)

    fused = w_text · text_pool + w_audio · audio_pool

The fused vector is used together with PSID disagreement as the classifier
input.

Missing modality handling
-------------------------
If audio is entirely absent (audio_available = False), g_audio is forced to
0. The gating then degrades gracefully to text-only mode.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from shared.constants import QUALITY_DIM


class ConfidenceGatedFusion(nn.Module):
    """Adaptive confidence-gated fusion of text and audio representations.

    Args:
        text_dim:      width of text pool vector.
        audio_dim:     width of audio pool vector.
        quality_dim:   width of audio quality vector (QUALITY_DIM).
        hidden_dim:    gating network hidden width.
        dropout:       dropout rate.
        normalise_weights: softmax-normalise modality weights (default True).
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        quality_dim: int = QUALITY_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        normalise_weights: bool = True,
    ) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.quality_dim = quality_dim
        self.normalise_weights = normalise_weights

        # Extra scalar signals: [audio_available (1), quality_ok (1)]
        signal_dim = text_dim + audio_dim + quality_dim + 2

        self.gate_net = nn.Sequential(
            nn.Linear(signal_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),   # logits for [text_weight, audio_weight]
        )

        # Optional reliability modulator (learnable scalar per quality dimension)
        self.quality_reliability = nn.Linear(quality_dim, 1, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        text_emb: torch.Tensor,          # [B, text_dim]
        audio_emb: torch.Tensor,         # [B, audio_dim]
        quality_vec: torch.Tensor,       # [B, QUALITY_DIM]
        audio_available: torch.Tensor,   # [B] bool
        quality_ok: torch.Tensor,        # [B] bool
        transcription_confidence: float | torch.Tensor = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Compute modality weights and fused representation.

        Returns:
            fused:        [B, text_dim]  (same width as text_emb)
            text_weight:  [B]
            audio_weight: [B]
            gate_logits:  [B, 2]        (for analysis)
        """
        B = text_emb.size(0)
        device = text_emb.device

        a_avail = audio_available.float().unsqueeze(-1)   # [B, 1]
        q_ok    = quality_ok.float().unsqueeze(-1)         # [B, 1]

        # Reliability prior from quality vector
        # q_reliable ∈ (0, 1) — higher = better audio quality signal
        q_reliable = torch.sigmoid(
            self.quality_reliability(quality_vec)
        )  # [B, 1]

        # Build gating signal
        signal = torch.cat(
            [text_emb, audio_emb * a_avail, quality_vec, a_avail, q_ok],
            dim=-1,
        )  # [B, signal_dim]

        gate_logits = self.gate_net(signal)   # [B, 2]
        gate_weights = torch.sigmoid(gate_logits)  # [B, 2]

        g_text  = gate_weights[:, 0]   # [B]
        g_audio = gate_weights[:, 1]   # [B]

        # Apply reliability priors
        if isinstance(transcription_confidence, (int, float)):
            r_text = torch.full((B,), float(transcription_confidence), device=device)
        else:
            r_text = transcription_confidence.to(device).float().squeeze()

        r_audio = (q_reliable.squeeze(-1) * a_avail.squeeze(-1))  # [B]

        eff_text  = g_text  * r_text.clamp(0.0, 1.0)   # [B]
        eff_audio = g_audio * r_audio                   # [B]

        # Force audio weight to 0 when audio is unavailable
        eff_audio = eff_audio * a_avail.squeeze(-1)

        if self.normalise_weights:
            total = eff_text + eff_audio + 1e-8
            w_text  = eff_text  / total   # [B]
            w_audio = eff_audio / total   # [B]
        else:
            w_text  = eff_text.clamp(0.0, 1.0)
            w_audio = eff_audio.clamp(0.0, 1.0)

        # Weighted sum — project audio to text_dim if necessary
        if self.text_dim != self.audio_dim:
            raise NotImplementedError(
                "text_dim != audio_dim; add a projection before ConfidenceGatedFusion"
            )

        fused = (
            w_text.unsqueeze(-1)  * text_emb +
            w_audio.unsqueeze(-1) * audio_emb
        )  # [B, text_dim]

        return {
            "fused": fused,
            "text_weight": w_text,
            "audio_weight": w_audio,
            "gate_logits": gate_logits,
        }
