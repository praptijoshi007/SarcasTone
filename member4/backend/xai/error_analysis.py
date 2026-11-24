"""Explainable AI: Error analysis and diagnostic reasoning module.

Performs case-by-case diagnosis of false positives, false negatives,
and incongruence patterns (e.g., deadpan irony vs. polite praise).
"""
from __future__ import annotations

from typing import Any, Dict, List


def diagnose_multimodal_incongruence(
    text: str,
    predicted_label: str,
    confidence: float,
    psid_score: float,
    gating_info: Dict[str, Any],
    top_tokens: List[Dict[str, Any]],
    top_prosody: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate high-level diagnostic explanation for the multimodal decision."""
    tokens_str = ", ".join([f"'{t['token']}'" for t in top_tokens[:3]]) if top_tokens else "none"
    prosody_str = ", ".join([f"{p['feature_name']}" for p in top_prosody[:2]]) if top_prosody else "none"

    if predicted_label == "sarcastic":
        if psid_score > 0.45:
            dissonance_desc = (
                f"Significant semantic-prosodic dissonance detected (PSID score: {psid_score:.3f}). "
                f"Linguistic politeness/praise ({tokens_str}) sharply contradicts expressive speech acoustic markers ({prosody_str})."
            )
        else:
            dissonance_desc = (
                f"Moderate dissonance (PSID score: {psid_score:.3f}). "
                f"Sarcastic intent identified primarily through acoustic emphasis ({prosody_str}) and dialogue context."
            )
        summary = (
            f"Classified as SARCASTIC with {confidence*100:.1f}% confidence. "
            f"The speaker's prosody signals an ironic detachment from the literal meaning of the words."
        )
    else:
        if psid_score < 0.20:
            dissonance_desc = (
                f"Harmonious semantic-prosodic alignment (PSID score: {psid_score:.3f}). "
                "Linguistic sentiment and speech tone are congruent without ironic inflection."
            )
        else:
            dissonance_desc = (
                f"Slight tonal variation (PSID score: {psid_score:.3f}), "
                "but insufficient to overturn literal semantic interpretation."
            )
        summary = (
            f"Classified as NON-SARCASTIC (LITERAL) with {confidence*100:.1f}% confidence. "
            "No substantial incongruence was found between speech prosody and semantic content."
        )

    return {
        "summary": summary,
        "dissonance_explanation": dissonance_desc,
        "gating_explanation": gating_info.get("explanation", ""),
    }
