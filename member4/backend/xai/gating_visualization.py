"""Explainable AI: Modality contribution and adaptive confidence gating visualization.

Computes how much text semantics vs. speech prosody contributed to the prediction,
explaining the dynamic shift in gating weights.
"""
from __future__ import annotations

import math
from typing import Any, Dict


def compute_gating_weights(
    has_text: bool,
    has_audio: bool,
    audio_quality: Dict[str, Any] | None = None,
    asr_confidence: float | None = None,
    text_entropy: float = 0.5,
    audio_entropy: float = 0.5,
) -> Dict[str, Any]:
    """Compute adaptive confidence gating weights between text and audio modalities.

    Formula:
      alpha_text  = sigma(W_text  * [q_text,  -H(p_text)])
      alpha_audio = sigma(W_audio * [q_audio, -H(p_audio)])
    Normalized so alpha_text + alpha_audio = 1.0.
    """
    if has_text and not has_audio:
        return {
            "text": 1.0,
            "audio": 0.0,
            "status": "text_only_fallback",
            "explanation": "Speech audio signal unavailable or degraded; system assigned 100% weight to text semantics.",
        }

    if has_audio and not has_text:
        return {
            "text": 0.0,
            "audio": 1.0,
            "status": "audio_only_fallback",
            "explanation": "Text transcript unavailable; system assigned 100% weight to acoustic speech prosody.",
        }

    if not has_text and not has_audio:
        return {
            "text": 0.5,
            "audio": 0.5,
            "status": "zero_modality_fallback",
            "explanation": "Both modalities missing; uninformative prior assigned.",
        }

    # Both modalities available: dynamically compute based on acoustic quality & confidence
    snr_db = 30.0
    clipping = 0.0
    silence = 0.0
    if audio_quality:
        snr_db = float(audio_quality.get("snr_db_estimate", audio_quality.get("approximate_level_indicator_db", 30.0)) or 30.0)
        clipping = float(audio_quality.get("clipping_ratio", 0.0) or 0.0)
        silence = float(audio_quality.get("silence_ratio", 0.0) or 0.0)

    # Audio quality factor [0.0 to 1.0]
    snr_factor = max(0.0, min(1.0, (snr_db - 0.0) / 30.0))
    audio_cleanliness = snr_factor * (1.0 - clipping) * (1.0 - silence)

    # Text quality factor [0.0 to 1.0]
    text_cleanliness = asr_confidence if asr_confidence is not None else 0.95

    # Logit calculation for softmax gating
    audio_logit = 1.5 * audio_cleanliness - 0.5 * audio_entropy
    text_logit = 1.5 * text_cleanliness - 0.5 * text_entropy

    exp_a = math.exp(audio_logit)
    exp_t = math.exp(text_logit)
    alpha_audio = round(exp_a / (exp_a + exp_t), 4)
    alpha_text = round(1.0 - alpha_audio, 4)

    # Narrative explanation
    if alpha_text > 0.65:
        explanation = (
            f"Text semantics dominant ({alpha_text*100:.1f}% vs {alpha_audio*100:.1f}%). "
            f"Reduced reliance on audio due to acoustic noise or lower SNR ({snr_db:.1f} dB)."
        )
    elif alpha_audio > 0.65:
        explanation = (
            f"Speech prosody dominant ({alpha_audio*100:.1f}% vs {alpha_text*100:.1f}%). "
            f"Strong prosodic contours and high audio clarity outweighed literal text."
        )
    else:
        explanation = (
            f"Balanced multimodal fusion ({alpha_text*100:.1f}% text, {alpha_audio*100:.1f}% audio). "
            f"Pristine audio (SNR {snr_db:.1f} dB) and high-confidence transcript reinforced each other."
        )

    return {
        "text": alpha_text,
        "audio": alpha_audio,
        "status": "balanced_gating",
        "explanation": explanation,
    }
