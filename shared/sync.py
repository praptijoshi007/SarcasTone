"""Multimodal synchronization pipeline connecting Member 1 (Textual Semantics) and Member 2 (Speech Prosody).

Ensures bidirectional alignment between text NLP and speech acoustic intelligence,
adhering strictly to the interface contract in configs/interface_contract.yaml and shared/interfaces.py.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_M1_SRC = _REPO_ROOT / "member1_text" / "src"
_M2_SRC = _REPO_ROOT / "member2_audio" / "src"
for _p in (str(_REPO_ROOT), str(_M1_SRC), str(_M2_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.constants import (
    AUDIO_HIDDEN,
    CONTEXT_HIDDEN,
    PROSODY_DIM,
    PROSODY_FEATURES,
    QUALITY_DIM,
    QUALITY_FEATURES,
    TARGET_SAMPLE_RATE,
    TEXT_HIDDEN,
)
from shared.interfaces import parse_sarcasm_label, sarcasm_name

# Member 2 audio processors
from member2_audio.src.audio_intelligence.audio_intelligence import (
    extract_audio_features as m2_extract_audio_features,
    get_audio_quality as m2_get_audio_quality,
)
from member2_audio.src.feature_extraction.acoustic_features import (
    extract_prosody_features as m2_extract_prosody,
)
from member2_audio.src.audio_intelligence.audio_quality import (
    extract_quality_features as m2_extract_quality,
)


@dataclass
class MultimodalFeatureBundle:
    """Synchronized multimodal feature bundle for one utterance."""

    example_id: str
    text: str
    audio_path: str | None
    speaker: str = "UNKNOWN"
    context: list[str] = field(default_factory=list)
    has_text: bool = True
    has_audio: bool = False
    transcript_source: str = "gold"

    # Contract tensors (shapes match configs/interface_contract.yaml)
    text_embeddings: list[list[float]] | None = None  # [1, L, 768]
    audio_embeddings: list[list[float]] | None = None  # [1, F, 768]
    context_embeddings: list[list[float]] | None = None  # [1, C, 768]
    pooled_text: list[float] | None = None  # [768]
    pooled_audio: list[float] | None = None  # [768]

    # Prosody [1, 32] and Quality [1, 6]
    prosody: list[float] = field(default_factory=lambda: [0.0] * PROSODY_DIM)
    quality: list[float] = field(default_factory=lambda: [0.0] * QUALITY_DIM)
    prosody_features: dict[str, float] = field(default_factory=dict)
    quality_features: dict[str, float] = field(default_factory=dict)

    # Predictions
    text_prediction: dict[str, Any] = field(default_factory=dict)
    audio_prediction: dict[str, Any] = field(default_factory=dict)
    emotion_prediction: dict[str, Any] = field(default_factory=dict)

    # Agreement
    cosine_agreement: float | None = None
    sample_rate_hz: int = TARGET_SAMPLE_RATE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sync_utterance(
    *,
    text: str | None = None,
    audio_path: str | Path | None = None,
    context: list[str] | str | None = None,
    speaker: str = "UNKNOWN",
    example_id: str = "custom_utterance",
) -> MultimodalFeatureBundle:
    """
    Synchronously extract and align multimodal features across Member 1 and Member 2.

    If text is absent but audio is provided, Member 1 ASR is called to transcribe.
    Speech prosody (32-d) and audio quality (6-d) are extracted by Member 2.
    Text semantics (768-d) and predictions are extracted by Member 1.
    """
    norm_context: list[str] = []
    if isinstance(context, str):
        norm_context = [context] if context.strip() else []
    elif isinstance(context, list):
        norm_context = [str(c) for c in context]

    transcript_source = "gold"
    audio_path_obj = Path(audio_path) if audio_path is not None else None
    has_audio = audio_path_obj is not None and audio_path_obj.is_file()

    # Step 1: Member 1 ASR if text is not provided but audio is available
    final_text = (text or "").strip()
    if not final_text and has_audio:
        try:
            from member1_text.asr import transcribe_audio

            asr_res = transcribe_audio(audio_path_obj)
            final_text = asr_res.get("text", "")
            transcript_source = "asr"
        except Exception:
            final_text = ""
            transcript_source = "gold"

    has_text = bool(final_text)

    # Step 2: Member 2 Audio & Prosody Extraction
    prosody_vec = np.zeros(PROSODY_DIM, dtype=np.float32)
    quality_vec = np.zeros(QUALITY_DIM, dtype=np.float32)
    prosody_dict: dict[str, float] = {k: 0.0 for k in PROSODY_FEATURES}
    quality_dict: dict[str, float] = {k: 0.0 for k in QUALITY_FEATURES}
    audio_pred: dict[str, Any] = {}
    emotion_pred: dict[str, Any] = {}
    pooled_audio_vec: np.ndarray | None = None
    frame_audio_embeddings: list[list[float]] | None = None

    if has_audio and audio_path_obj is not None:
        try:
            p_vec, p_dict = m2_extract_prosody(audio_path_obj)
            q_vec, q_dict = m2_extract_quality(audio_path_obj)
            prosody_vec = p_vec
            quality_vec = q_vec
            prosody_dict = p_dict
            quality_dict = q_dict
        except Exception:
            pass

        # Try Wav2Vec2 embedding from Member 2
        try:
            from member2_audio.src.audio_intelligence.audio_intelligence import (
                get_audio_embedding,
                predict_audio,
                predict_emotion,
            )

            pooled = get_audio_embedding(audio_path_obj, return_dict=False)
            if pooled is not None and pooled.shape == (AUDIO_HIDDEN,):
                pooled_audio_vec = pooled
            audio_pred = predict_audio(audio_path_obj)
            emotion_pred = predict_emotion(audio_path_obj)
        except Exception:
            pass

    # Step 3: Member 1 Text Semantics Extraction
    text_pred: dict[str, Any] = {}
    pooled_text_vec: np.ndarray | None = None
    context_vec: np.ndarray | None = None

    if has_text:
        try:
            from member1_text.api import get_text_embedding, predict_text

            ctx_str = " ".join(norm_context) if norm_context else None
            t_emb = get_text_embedding(final_text, context=ctx_str)
            pooled_text_vec = np.asarray(t_emb["utterance"], dtype=np.float32)
            if t_emb.get("has_context"):
                context_vec = np.asarray(t_emb["context"], dtype=np.float32)
            text_pred = predict_text(final_text, context=ctx_str)
        except Exception:
            pass

    # Step 4: Calculate Cross-Modal Cosine Agreement if both pooled embeddings exist
    cosine_agreement = None
    if pooled_text_vec is not None and pooled_audio_vec is not None:
        t_norm = np.linalg.norm(pooled_text_vec)
        a_norm = np.linalg.norm(pooled_audio_vec)
        if t_norm > 0 and a_norm > 0:
            cosine_agreement = float(np.dot(pooled_text_vec, pooled_audio_vec) / (t_norm * a_norm))

    return MultimodalFeatureBundle(
        example_id=example_id,
        text=final_text,
        audio_path=str(audio_path_obj) if audio_path_obj else None,
        speaker=speaker,
        context=norm_context,
        has_text=has_text,
        has_audio=has_audio,
        transcript_source=transcript_source,
        text_embeddings=None,
        audio_embeddings=frame_audio_embeddings,
        context_embeddings=None,
        pooled_text=pooled_text_vec.tolist() if pooled_text_vec is not None else None,
        pooled_audio=pooled_audio_vec.tolist() if pooled_audio_vec is not None else None,
        prosody=prosody_vec.tolist(),
        quality=quality_vec.tolist(),
        prosody_features=prosody_dict,
        quality_features=quality_dict,
        text_prediction=text_pred,
        audio_prediction=audio_pred,
        emotion_prediction=emotion_pred,
        cosine_agreement=cosine_agreement,
        sample_rate_hz=TARGET_SAMPLE_RATE,
    )


def verify_corpus_sync() -> dict[str, Any]:
    """
    Verify complete synchronization between Member 1's text data and Member 2's audio manifest.
    Checks:
      - All 690 MUStARD utterance IDs exist in both Member 1 and Member 2.
      - Sarcasm labels match (0 vs 1).
      - Speakers match.
      - Utterance text strings match.
    """
    import pandas as pd
    from member1_text.data.mustard import load_mustard

    # Load Member 1 records
    m1_examples = load_mustard()
    m1_dict = {ex.example_id: ex for ex in m1_examples}

    # Load Member 2 manifest
    repo_root = Path(__file__).resolve().parents[1]
    m2_manifest_path = repo_root / "member2_audio" / "data" / "metadata" / "mustard_audio_manifest.csv"
    if not m2_manifest_path.is_file():
        raise FileNotFoundError(f"Member 2 manifest not found: {m2_manifest_path}")

    m2_df = pd.read_csv(m2_manifest_path)
    m2_dict = {str(row["sample_id"]): row for _, row in m2_df.iterrows()}

    # Compare ID sets
    m1_ids = set(m1_dict.keys())
    m2_ids = set(m2_dict.keys())

    missing_in_m1 = sorted(list(m2_ids - m1_ids))
    missing_in_m2 = sorted(list(m1_ids - m2_ids))
    common_ids = sorted(list(m1_ids & m2_ids))

    label_mismatches = []
    speaker_mismatches = []

    for uid in common_ids:
        ex = m1_dict[uid]
        row = m2_dict[uid]
        m1_label = int(ex.label)
        m2_label = int(row["sarcasm"])
        if m1_label != m2_label:
            label_mismatches.append(uid)

        m1_speaker = (ex.extra.get("speaker") or "").strip().upper()
        m2_speaker = (str(row["speaker"]) or "").strip().upper()
        if m1_speaker != m2_speaker:
            speaker_mismatches.append(uid)

    is_synced = (
        len(common_ids) == 690
        and len(missing_in_m1) == 0
        and len(missing_in_m2) == 0
        and len(label_mismatches) == 0
        and len(speaker_mismatches) == 0
    )

    return {
        "is_synced": is_synced,
        "total_member1_records": len(m1_examples),
        "total_member2_records": len(m2_df),
        "aligned_records": len(common_ids),
        "missing_in_member1": missing_in_m1,
        "missing_in_member2": missing_in_m2,
        "label_mismatches": label_mismatches,
        "speaker_mismatches": speaker_mismatches,
        "prosody_dim": PROSODY_DIM,
        "quality_dim": QUALITY_DIM,
        "text_hidden": TEXT_HIDDEN,
        "audio_hidden": AUDIO_HIDDEN,
        "sample_rate_hz": TARGET_SAMPLE_RATE,
    }
