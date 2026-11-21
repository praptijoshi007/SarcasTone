"""Label parsing and documented model interfaces.

The functions in the member packages implement these contracts. This module
is the agreement those functions have to keep.
"""

from __future__ import annotations

from shared.constants import (
    AUDIO_HIDDEN,
    CONTEXT_HIDDEN,
    FUSION_LOGITS,
    NEGATIVE_SARCASM,
    POSITIVE_SARCASM,
    PROSODY_DIM,
    QUALITY_DIM,
    SARCASM_LABELS,
    TARGET_SAMPLE_RATE,
    TEXT_HIDDEN,
)
from shared.errors import DatasetSchemaError


def parse_sarcasm_label(value: object) -> int:
    """Map a dataset cell onto ``0`` (non-sarcastic) or ``1`` (sarcastic).

    Irony datasets use the same integers. Callers must keep the dataset name
    so irony is not silently treated as the same task as MUStARD sarcasm.
    """

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, float) and value in (0.0, 1.0):
        return int(value)
    text = str(value).strip().lower()
    if text in POSITIVE_SARCASM:
        return 1
    if text in NEGATIVE_SARCASM:
        return 0
    raise DatasetSchemaError(f"Unrecognised sarcasm/irony label: {value!r}")


def sarcasm_name(label: int) -> str:
    if label not in (0, 1):
        raise DatasetSchemaError(f"Sarcasm class index must be 0 or 1, got {label}")
    return SARCASM_LABELS[label]


# Interface contract for member modules. Shapes are batch-first.
#
# encode_text(text, context=None) ->
#   utterance_embeddings: float32[B, L, TEXT_HIDDEN]
#   utterance_mask: bool[B, L]          True where a token is real
#   context_embeddings: float32[B, C, CONTEXT_HIDDEN] | None
#   context_mask: bool[B, C] | None
#   pooled_utterance: float32[B, TEXT_HIDDEN]
#
# get_audio_embedding(audio_path) ->
#   frame_embeddings: float32[1, F, AUDIO_HIDDEN]
#   frame_mask: bool[1, F]
#   utterance_embedding: float32[1, AUDIO_HIDDEN]
#   sample_rate_hz: TARGET_SAMPLE_RATE
#
# extract_audio_features(audio_path) ->
#   prosody: float32[PROSODY_DIM]
#   quality: float32[QUALITY_DIM]
#   feature_names / quality_names aligned with shared.constants
#
# predict_multimodal(audio_path, text=None, context=None) ->
#   label: "sarcastic" | "non_sarcastic"
#   label_index: 0 | 1
#   probabilities: {class_name: float}
#   entropy: float
#   uncertain: bool
#   text_weight: float          mean of the text gate, in [0, 1]
#   disagreement_norm: float
#   cosine_agreement: float | None
#   transcript: str
#   transcript_is_ground_truth: bool
#   asr_confidence: float | None
INTERFACE_SUMMARY = {
    "text_embeddings": f"[B, L, {TEXT_HIDDEN}]",
    "audio_embeddings": f"[B, F, {AUDIO_HIDDEN}]",
    "context_embeddings": f"[B, C, {CONTEXT_HIDDEN}]",
    "prosody": f"[B, {PROSODY_DIM}]",
    "quality": f"[B, {QUALITY_DIM}]",
    "fusion_logits": f"[B, {FUSION_LOGITS}]",
    "sample_rate_hz": str(TARGET_SAMPLE_RATE),
}
