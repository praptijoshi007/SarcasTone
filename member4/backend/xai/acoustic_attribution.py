"""Explainable AI: Acoustic prosody feature attribution module.

Calculates the contribution and importance of speech prosody features
towards sarcasm vs. literal speech.
"""
from __future__ import annotations

from typing import Any, Dict, List

# Normative baselines for conversational speech (MUStARD conversational norm estimates)
_ACOUSTIC_BASELINES = {
    "f0_std": 25.0,                   # pitch variation in Hz
    "pitch_variation_coeff": 0.15,     # CV_f0 = std/mean
    "f0_range": 60.0,                  # max - min pitch in Hz
    "f0_slope": 0.0,                   # pitch direction
    "energy_dynamic_range_db": 12.0,   # peak vs average dB
    "speaking_rate_proxy": 4.5,        # syllables / sec
    "pause_ratio": 0.15,               # silence fraction
    "emphasis_proxy": 0.20,            # local energy peak
    "jitter_local": 0.015,             # pitch perturbation
    "shimmer_proxy": 0.04,             # amplitude perturbation
    "spectral_centroid_mean": 800.0,   # vocal brightness in Hz
    "spectral_flatness_mean": 1e-4,    # noisiness vs tonality
}


def compute_acoustic_attributions(
    prosody_dict: Dict[str, float],
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """Compute importance and direction of prosodic features.

    Returns a list of dicts with:
      - feature_name: str
      - feature_value: float
      - importance_score: float (0.0 to 1.0)
      - direction: 'promotes_sarcasm' | 'promotes_literal'
      - explanation: str
    """
    if not prosody_dict:
        return []

    attributions = []

    # 1. Pitch Variation (Exaggerated pitch contour is a key sarcasm marker)
    f0_std = float(prosody_dict.get("f0_std", 0.0))
    cv_pitch = float(prosody_dict.get("pitch_variation_coeff", 0.0))
    if cv_pitch > 0.25 or f0_std > 40.0:
        attributions.append({
            "feature_name": "pitch_variation_coeff",
            "feature_value": round(cv_pitch if cv_pitch > 0 else f0_std, 4),
            "importance_score": 0.88,
            "direction": "promotes_sarcasm",
            "explanation": "Exaggerated pitch variance (melodic sing-song or hyper-articulated contour).",
        })
    elif 0 < cv_pitch < 0.05 or (0 < f0_std < 10.0):
        attributions.append({
            "feature_name": "pitch_flatness_monotone",
            "feature_value": round(cv_pitch if cv_pitch > 0 else f0_std, 4),
            "importance_score": 0.76,
            "direction": "promotes_sarcasm",
            "explanation": "Unnaturally flat/monotone pitch (deadpan sarcastic delivery).",
        })

    # 2. Energy Dynamic Range
    energy_dyn = float(prosody_dict.get("energy_dynamic_range_db", 0.0))
    if energy_dyn > 18.0:
        attributions.append({
            "feature_name": "energy_dynamic_range_db",
            "feature_value": round(energy_dyn, 2),
            "importance_score": 0.82,
            "direction": "promotes_sarcasm",
            "explanation": "High acoustic stress and energy burst on focal syllables.",
        })
    elif energy_dyn > 0:
        attributions.append({
            "feature_name": "energy_dynamic_range_db",
            "feature_value": round(energy_dyn, 2),
            "importance_score": 0.40,
            "direction": "promotes_literal",
            "explanation": "Balanced dynamic range consistent with neutral discourse.",
        })

    # 3. Speaking Rate / Vowel Elongation
    speaking_rate = float(prosody_dict.get("speaking_rate_proxy", 0.0))
    if 0 < speaking_rate < 3.2:
        attributions.append({
            "feature_name": "speaking_rate_proxy",
            "feature_value": round(speaking_rate, 2),
            "importance_score": 0.74,
            "direction": "promotes_sarcasm",
            "explanation": "Prolonged syllables and drawn-out vowel duration (sarcastic drawl).",
        })
    elif speaking_rate > 5.5:
        attributions.append({
            "feature_name": "speaking_rate_proxy",
            "feature_value": round(speaking_rate, 2),
            "importance_score": 0.55,
            "direction": "promotes_literal",
            "explanation": "Rapid conversational tempo without sarcastic vowel lengthening.",
        })

    # 4. Emphasis Proxy
    emphasis = float(prosody_dict.get("emphasis_proxy", 0.0))
    if emphasis > 0.35:
        attributions.append({
            "feature_name": "emphasis_proxy",
            "feature_value": round(emphasis, 3),
            "importance_score": 0.70,
            "direction": "promotes_sarcasm",
            "explanation": "Disproportionate acoustic prominence on contrastive word.",
        })

    # 5. Pauses and Rhythm
    pause_ratio = float(prosody_dict.get("pause_ratio", 0.0))
    if pause_ratio > 0.30:
        attributions.append({
            "feature_name": "pause_ratio",
            "feature_value": round(pause_ratio, 3),
            "importance_score": 0.62,
            "direction": "promotes_sarcasm",
            "explanation": "Hesitation pauses preceding the ironic comment.",
        })

    # 6. Pitch Slope (terminal boundary tones)
    slope = float(prosody_dict.get("f0_slope", 0.0))
    if abs(slope) > 50.0:
        attributions.append({
            "feature_name": "f0_slope",
            "feature_value": round(slope, 2),
            "importance_score": 0.58,
            "direction": "promotes_sarcasm",
            "explanation": "Steep terminal pitch shift (fall-rise boundary tone).",
        })

    # Fill remaining features with calibrated divergences
    for feat_name, val in prosody_dict.items():
        if len(attributions) >= top_k:
            break
        if any(a["feature_name"] == feat_name for a in attributions):
            continue
        fval = float(val)
        base = _ACOUSTIC_BASELINES.get(feat_name, 1.0)
        if base != 0:
            rel_div = abs(fval - base) / abs(base)
            imp = min(0.65, round(rel_div * 0.3, 3))
            direction = "promotes_sarcasm" if rel_div > 0.5 else "promotes_literal"
            attributions.append({
                "feature_name": feat_name,
                "feature_value": round(fval, 4),
                "importance_score": imp,
                "direction": direction,
                "explanation": f"Observed value {fval:.3f} relative to conversational baseline {base:.3f}.",
            })

    # Sort by importance descending
    attributions.sort(key=lambda x: x["importance_score"], reverse=True)
    return attributions[:top_k]
