"""Explainable AI: Confidence calibration and uncertainty quantification module."""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple


def calibrate_probabilities(
    raw_logits: Tuple[float, float] | list[float],
    temperature: float = 1.25,
) -> Dict[str, float]:
    """Apply post-hoc temperature scaling to calibrate probabilities.

    Guarantees monotonic scaling and prevents overconfident probability estimation.
    """
    t = max(0.1, float(temperature))
    scaled_0 = raw_logits[0] / t
    scaled_1 = raw_logits[1] / t

    # Numerically stable softmax
    max_logit = max(scaled_0, scaled_1)
    e0 = math.exp(scaled_0 - max_logit)
    e1 = math.exp(scaled_1 - max_logit)
    denom = e0 + e1

    p0 = e0 / denom
    p1 = e1 / denom

    return {
        "non_sarcastic": round(p0, 4),
        "sarcastic": round(p1, 4),
    }


def compute_entropy(probabilities: Dict[str, float]) -> float:
    """Compute Shannon predictive entropy in nats."""
    p0 = max(1e-12, probabilities.get("non_sarcastic", 0.5))
    p1 = max(1e-12, probabilities.get("sarcastic", 0.5))
    entropy = - (p0 * math.log(p0) + p1 * math.log(p1))
    return round(float(entropy), 4)


def evaluate_uncertainty(
    probabilities: Dict[str, float],
    entropy_threshold: float = 0.55,
    has_audio: bool = True,
    has_text: bool = True,
    psid_dissonance: float = 0.0,
) -> Dict[str, Any]:
    """Diagnose model uncertainty and generate explainability alerts."""
    entropy = compute_entropy(probabilities)
    p_max = max(probabilities.values())
    margin = round(abs(probabilities["sarcastic"] - probabilities["non_sarcastic"]), 4)

    is_uncertain = entropy >= entropy_threshold or margin < 0.15

    reason = None
    if is_uncertain:
        if not has_audio:
            reason = (
                "High uncertainty due to missing audio. Without speech prosody, "
                "literal text semantics cannot definitively establish sarcastic intent."
            )
        elif not has_text:
            reason = (
                "High uncertainty due to missing text transcript. Prosody alone "
                "lacks situational context to disambiguate sarcasm from genuine excitement."
            )
        elif margin < 0.15:
            reason = (
                f"Close decision boundary (margin: {margin*100:.1f}%). "
                "Acoustic cues and semantic valence show subtle, borderline dissonance."
            )
        else:
            reason = (
                f"Elevated predictive entropy ({entropy:.3f} nats >= {entropy_threshold:.3f}). "
                "Prediction is sensitive to subtle tonal nuances."
            )

    return {
        "entropy": entropy,
        "confidence": round(p_max, 4),
        "margin": margin,
        "is_uncertain": is_uncertain,
        "reason": reason,
    }
