"""Shared tensor, label, and feature contracts.

Projection layers in each module must convert native encoder widths into
these sizes. Keep ``configs/interface_contract.yaml`` in sync when changing a
dimension, sample rate, or label.
"""

from __future__ import annotations

TEXT_HIDDEN = 768
AUDIO_HIDDEN = 768
CONTEXT_HIDDEN = 768
FUSION_LOGITS = 2
DISAGREE_DIM = 128
PROSODY_PROJ_DIM = 128

TARGET_SAMPLE_RATE = 16_000
MIN_AUDIO_SECONDS = 0.2
MAX_AUDIO_SECONDS = 120.0

NUM_SARCASM_LABELS = 2
SARCASM_LABELS = ("non_sarcastic", "sarcastic")

# Fixed order. Member 2 writes this vector; Member 3 reads it by position.
PROSODY_FEATURES: tuple[str, ...] = (
    "duration_sec",
    "f0_mean",
    "f0_std",
    "f0_min",
    "f0_max",
    "f0_range",
    "f0_slope",
    "jitter_local",
    "rms_mean",
    "rms_std",
    "rms_max",
    "energy_dynamic_range_db",
    "zcr_mean",
    "zcr_std",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "spectral_bandwidth_mean",
    "spectral_flatness_mean",
    "mfcc1_mean",
    "mfcc2_mean",
    "mfcc1_std",
    "speaking_rate_proxy",
    "pause_ratio",
    "pause_count_per_sec",
    "mean_pause_sec",
    "voiced_ratio",
    "intonation_range_semitones",
    "shimmer_proxy",
    "pitch_variation_coeff",
    "emphasis_proxy",
)

# Heuristic signal descriptors. They are not a calibrated audio-quality model.
QUALITY_FEATURES: tuple[str, ...] = (
    "snr_db_estimate",
    "clipping_ratio",
    "silence_ratio",
    "duration_in_range",
    "rms",
    "spectral_flatness",
)

PROSODY_DIM = len(PROSODY_FEATURES)
QUALITY_DIM = len(QUALITY_FEATURES)

EMOTION_LABELS: tuple[str, ...] = (
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
    "surprise",
)

# Speaker strings that do not identify a person. Each example is its own group
# so these rows cannot glue unrelated clips into one "speaker".
GENERIC_SPEAKERS = frozenset({"", "PERSON", "UNKNOWN", "UNK", "NARRATOR", "NONE"})

POSITIVE_SARCASM = frozenset(
    {"1", "true", "yes", "y", "sarcasm", "sarcastic", "irony", "ironic"}
)
NEGATIVE_SARCASM = frozenset(
    {
        "0",
        "false",
        "no",
        "n",
        "non-sarcastic",
        "nonsarcastic",
        "non_sarcastic",
        "not_sarcastic",
        "regular",
    }
)
