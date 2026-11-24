"""src/inference/predict_multimodal.py

The required predict_multimodal() interface for SarcasTone Member 3.

Signature
---------
predict_multimodal(audio_path, text=None, context=None) -> dict

The function:
  1. Transcribes audio if text is not provided (via member1 ASR or whisper).
  2. Extracts acoustic features and wav2vec2 embedding (via member2).
  3. Assesses audio quality (via member2).
  4. Builds a synthetic batch and runs the fusion model.
  5. Returns predicted class, probabilities, uncertainty, modality weights,
     disagreement score, and transcript provenance.

No result is fabricated. If the model checkpoint is missing, the function
raises FileNotFoundError with a helpful message.

Modality handling
-----------------
If member2 audio extraction fails, audio_available is set to False and the
model receives zero audio embeddings — it does NOT silently produce garbage.

If member1 text encoding fails, text embeddings are set to zeros and
text_available is reported as False in the result metadata.

An explicit warning is logged any time a modality falls back to zeros.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
_M3   = _REPO / "member3_fusion"
for _p in [str(_REPO), str(_M3), str(_M3 / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.constants import (
    TEXT_HIDDEN, AUDIO_HIDDEN, PROSODY_DIM, QUALITY_DIM,
    SARCASM_LABELS, PROSODY_FEATURES, QUALITY_FEATURES,
)
from src.models.fusion_model import SarcasToneFusionModel
from src.uncertainty.calibration import predict_with_uncertainty

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cache so repeated calls don't reload from disk
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, SarcasToneFusionModel] = {}


def _load_model(checkpoint_dir: Path, device: torch.device) -> SarcasToneFusionModel:
    """Load (or return cached) fusion model from checkpoint directory."""
    key = str(checkpoint_dir)
    if key not in _MODEL_CACHE:
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"No fusion model checkpoint at '{checkpoint_dir}'.\n"
                "Run  python member3_fusion/train_fusion.py --experiment full_model  first."
            )
        # Verify config.json and model.pt exist
        if not (checkpoint_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"Checkpoint directory '{checkpoint_dir}' exists but config.json is missing."
            )
        if not (checkpoint_dir / "model.pt").is_file():
            raise FileNotFoundError(
                f"Checkpoint directory '{checkpoint_dir}' exists but model.pt is missing."
            )
        model = SarcasToneFusionModel.load(checkpoint_dir, device=device)
        model.eval()
        _MODEL_CACHE[key] = model
        log.info("Loaded fusion model from %s", checkpoint_dir)
    return _MODEL_CACHE[key]


# ---------------------------------------------------------------------------
# ASR — transcribe audio when text is not provided
# ---------------------------------------------------------------------------

def _transcribe(audio_path: Path) -> tuple[str, float | None]:
    """Try member1 ASR, then whisper, then return empty string."""
    # Try member1 ASR (whisper-based)
    try:
        m1_src = _REPO / "member1_text" / "src"
        if str(m1_src) not in sys.path:
            sys.path.insert(0, str(m1_src))
        from member1_text.asr import transcribe_audio  # type: ignore
        result     = transcribe_audio(str(audio_path))
        text       = result.get("text", "").strip()
        confidence = result.get("confidence")
        if text:
            log.debug("member1 ASR succeeded: %r (conf=%s)", text[:60], confidence)
            return text, confidence
    except Exception as e:
        log.debug("member1 ASR failed: %s", e)

    # Try openai-whisper directly
    try:
        import whisper  # type: ignore
        model  = whisper.load_model("base")
        result = model.transcribe(str(audio_path))
        text   = result.get("text", "").strip()
        if text:
            log.debug("whisper ASR succeeded: %r", text[:60])
            return text, None
    except Exception as e:
        log.debug("whisper ASR failed: %s", e)

    log.warning(
        "All ASR methods failed for %s; using empty transcript. "
        "Text modality will be unavailable.",
        audio_path,
    )
    return "", None


# ---------------------------------------------------------------------------
# Audio feature extraction — calls member2 functions
# ---------------------------------------------------------------------------

def _extract_audio_features(audio_path: Path) -> dict:
    """Extract wav2vec2 embedding, prosody, and quality via member2.

    Returns dict with keys:
        embedding:       np.ndarray [768] or None
        prosody_vector:  np.ndarray [PROSODY_DIM]
        quality_vector:  np.ndarray [QUALITY_DIM]
        acoustic_full:   np.ndarray [116] or zeros
        quality_ok:      bool
        audio_available: bool
        error:           str | None  (populated on failure for transparency)
    """
    result = {
        "embedding":       None,
        "prosody_vector":  np.zeros(PROSODY_DIM,  dtype=np.float32),
        "quality_vector":  np.zeros(QUALITY_DIM,  dtype=np.float32),
        "acoustic_full":   np.zeros(116,           dtype=np.float32),
        "quality_ok":      False,
        "audio_available": False,
        "error":           None,
    }

    if not audio_path.is_file():
        msg = f"Audio file not found: {audio_path}"
        log.warning(msg)
        result["error"] = msg
        return result

    # Try member2 audio intelligence
    try:
        m2_src = _REPO / "member2_audio" / "src"
        if str(m2_src) not in sys.path:
            sys.path.insert(0, str(m2_src))

        from audio_intelligence.audio_intelligence import (  # type: ignore
            get_audio_embedding,
            get_audio_quality,
        )
        from feature_extraction.acoustic_features import extract_features  # type: ignore

        # Wav2Vec2 pooled embedding [768]
        emb = get_audio_embedding(audio_path)
        if emb is not None and len(emb) == 768:
            result["embedding"]       = np.asarray(emb, dtype=np.float32)
            result["audio_available"] = True
        else:
            log.warning(
                "member2 returned unexpected embedding shape %s for %s; "
                "audio will be unavailable.",
                getattr(emb, "shape", None), audio_path,
            )

        # Audio quality
        q  = get_audio_quality(audio_path)
        qv = np.zeros(QUALITY_DIM, dtype=np.float32)
        qv[1] = float(q.get("clipping_ratio", 0.0) or 0.0)
        qv[2] = float(q.get("silence_ratio",  0.0) or 0.0)
        qv[3] = 1.0 if q.get("is_valid", False) else 0.0
        qv[4] = float(q.get("rms_energy",     0.0) or 0.0)
        result["quality_vector"] = qv
        result["quality_ok"]     = bool(q.get("is_valid", False))

        # Handcrafted prosody features
        raw_feats = extract_features(audio_path)
        prosody_v = np.zeros(PROSODY_DIM, dtype=np.float32)
        _prosody_map = {
            "f0_mean":                "pitch_f0_mean",
            "f0_std":                 "pitch_f0_std",
            "rms_mean":               "rms_energy_mean",
            "rms_std":                "rms_energy_std",
            "spectral_centroid_mean": "spectral_centroid_mean",
            "spectral_centroid_std":  "spectral_centroid_std",
            "zcr_mean":               "zero_crossing_rate_mean",
            "zcr_std":                "zero_crossing_rate_std",
            "duration_sec":           "duration",
        }
        for i, feat_name in enumerate(PROSODY_FEATURES):
            raw_key = _prosody_map.get(feat_name, feat_name)
            if raw_key in raw_feats:
                v = raw_feats[raw_key]
                if v is not None:
                    try:
                        fv = float(v)
                        if np.isfinite(fv):
                            prosody_v[i] = fv
                    except (TypeError, ValueError):
                        pass
        result["prosody_vector"] = prosody_v

        # Full acoustic vector (first 116 entries from raw_feats dict)
        feat_vals = []
        for k, v in raw_feats.items():
            try:
                fv = float(v)
                feat_vals.append(fv if np.isfinite(fv) else 0.0)
            except (TypeError, ValueError):
                feat_vals.append(0.0)
        full = np.array(feat_vals[:116], dtype=np.float32)
        if len(full) < 116:
            full = np.pad(full, (0, 116 - len(full)))
        result["acoustic_full"] = full

    except ImportError as e:
        msg = (
            f"member2_audio modules not importable ({e}). "
            "Audio features will be zeros and audio_available=False."
        )
        log.warning(msg)
        result["error"] = msg
    except Exception as e:
        msg = f"member2 audio extraction failed for {audio_path}: {e}"
        log.warning(msg)
        result["error"] = msg

    return result


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------

def _encode_text(
    text: str,
    context: list[str] | None,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor | None], bool]:
    """Get text and context embeddings via member1 or return zeros.

    Returns:
        (encodings_dict, text_was_available)
    """
    _zero_out = {
        "utterance":   torch.zeros(1, TEXT_HIDDEN, device=device),
        "context":     torch.zeros(1, TEXT_HIDDEN, device=device),
        "has_context": torch.zeros(1, dtype=torch.bool, device=device),
    }

    if not text.strip():
        log.warning(
            "Empty transcript — text modality will be zeros (text_available=False)."
        )
        return _zero_out, False

    try:
        m1_src = _REPO / "member1_text" / "src"
        if str(m1_src) not in sys.path:
            sys.path.insert(0, str(m1_src))
        from member1_text.api import get_text_embedding  # type: ignore
        ctx_str = " ".join(context) if context else None
        out     = get_text_embedding(text, ctx_str)
        return {
            "utterance":   out["utterance"].unsqueeze(0).to(device),
            "context":     out["context"].unsqueeze(0).to(device),
            "has_context": torch.tensor(
                [[out["has_context"]]], dtype=torch.bool, device=device
            ),
        }, True
    except ImportError:
        log.warning(
            "member1_text not importable; text modality will be zeros."
        )
        return _zero_out, False
    except Exception as e:
        log.warning("Text encoding failed (%s); text will be zeros.", e)
        return _zero_out, False


# ---------------------------------------------------------------------------
# Main predict_multimodal function
# ---------------------------------------------------------------------------

def predict_multimodal(
    audio_path: str | Path,
    text: Optional[str] = None,
    context: Optional[list[str]] = None,
    *,
    checkpoint_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    temperature: float | None = None,
    uncertainty_threshold: float = 0.347,
) -> dict:
    """Predict sarcasm from audio (and optional text/context).

    Args:
        audio_path:          Path to wav/mp3/mp4 audio file.
        text:                Ground-truth or manually-provided transcript.
                             If None, ASR is used.
        context:             List of preceding utterance strings (dialogue context).
        checkpoint_dir:      Path to the trained fusion model checkpoint directory.
                             Defaults to member3_fusion/checkpoints/full_model/checkpoint/
        device:              Torch device string or object. Auto-detected if None.
        temperature:         Calibration temperature (overrides model's fitted T).
        uncertainty_threshold: Entropy above which prediction is flagged uncertain.

    Returns:
        dict with keys:
            label:                      "sarcastic" or "non_sarcastic"
            label_index:                0 or 1
            probabilities:              {"non_sarcastic": float, "sarcastic": float}
            entropy:                    float  (predictive entropy in nats)
            uncertain:                  bool
            text_weight:                float | None   (mean modality gate weight)
            audio_weight:               float | None
            disagreement_norm:          float | None   (PSID disagreement magnitude)
            cosine_agreement:           float | None   (1=aligned, -1=opposite)
            transcript:                 str            (what was used as text input)
            transcript_is_ground_truth: bool
            asr_confidence:             float | None
            audio_available:            bool
            text_available:             bool
            audio_extraction_error:     str | None
    """
    audio_path = Path(audio_path)

    # Device
    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        _device = torch.device(device)
    else:
        _device = device

    # Checkpoint — look for the checkpoint sub-dir (from model.save())
    if checkpoint_dir is None:
        checkpoint_dir = _M3 / "checkpoints" / "full_model" / "checkpoint"
    checkpoint_dir = Path(checkpoint_dir)
    # Also accept the experiment dir directly (contains best_model.pt)
    if not checkpoint_dir.is_dir():
        # Try sibling "checkpoint" directory
        alt = checkpoint_dir.parent / "checkpoint"
        if alt.is_dir():
            checkpoint_dir = alt

    # Load model
    model = _load_model(checkpoint_dir, _device)

    # 1. Transcription
    transcript_is_gt  = text is not None
    asr_confidence: float | None = None
    if text is None:
        text, asr_confidence = _transcribe(audio_path)
        log.info("ASR transcript: %r (confidence=%s)", text[:80] if text else "", asr_confidence)
    transcript = text or ""

    # 2. Audio feature extraction
    audio_feats = _extract_audio_features(audio_path)
    if not audio_feats["audio_available"]:
        log.warning(
            "Audio unavailable for '%s'. Proceeding with zero audio embeddings.",
            audio_path,
        )

    # 3. Text encoding
    text_enc, text_was_available = _encode_text(transcript, context, _device)
    if not text_was_available:
        log.warning(
            "Text unavailable. Proceeding with zero text embeddings."
        )

    # 4. Build a batch of size 1
    def _t(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(_device)

    if audio_feats["embedding"] is not None:
        audio_emb = _t(audio_feats["embedding"])
    else:
        audio_emb = torch.zeros(1, AUDIO_HIDDEN, device=_device)

    batch = {
        "audio_embedding":  audio_emb,
        "audio_available":  torch.tensor([audio_feats["audio_available"]], dtype=torch.bool, device=_device),
        "prosody":          _t(audio_feats["prosody_vector"]),
        "acoustic_full":    _t(audio_feats["acoustic_full"]),
        "quality":          _t(audio_feats["quality_vector"]),
        "quality_ok":       torch.tensor([audio_feats["quality_ok"]], dtype=torch.bool, device=_device),
        "text_embedding":   text_enc["utterance"],
        "text_available":   torch.tensor([text_was_available], dtype=torch.bool, device=_device),
        "label":            torch.zeros(1, dtype=torch.long, device=_device),  # placeholder
        "texts":            [transcript],
        "contexts":         [context or []],
        "example_ids":      ["inference"],
        "speakers":         ["UNKNOWN"],
    }

    # 5. Run model
    model.eval()
    with torch.no_grad():
        # Flatten has_context from [1, 1] → [1] if needed
        hc = text_enc["has_context"]
        if hc.dim() > 1:
            hc = hc.squeeze(-1)
        out = model(
            batch,
            text_embeddings=text_enc["utterance"],
            context_embeddings=text_enc["context"],
            has_context=hc,
        )

    # 6. Apply calibration temperature
    T = (
        temperature
        if temperature is not None
        else float(model.temperature_scaler.temperature.item())
    )
    cal_out = predict_with_uncertainty(
        out["logits"],
        temperature=T,
        uncertainty_threshold=uncertainty_threshold,
    )

    probs     = cal_out["probabilities"][0].cpu().tolist()
    label_idx = int(cal_out["label_index"][0].item())
    entropy   = float(cal_out["entropy"][0].item())
    uncertain = bool(cal_out["uncertain"][0].item())

    # 7. Modality weights
    text_w  = float(out["text_weight"][0].item())         if out["text_weight"]         is not None else None
    audio_w = float(out["audio_weight"][0].item())        if out["audio_weight"]         is not None else None
    disag   = float(out["disagreement_norm"][0].item())   if out["disagreement_norm"]    is not None else None
    cosine  = float(out["cosine_agree"][0].item())        if out["cosine_agree"]         is not None else None

    return {
        "label":                      SARCASM_LABELS[label_idx],
        "label_index":                label_idx,
        "probabilities":              {
            SARCASM_LABELS[0]: probs[0],
            SARCASM_LABELS[1]: probs[1],
        },
        "entropy":                    entropy,
        "uncertain":                  uncertain,
        "text_weight":                text_w,
        "audio_weight":               audio_w,
        "disagreement_norm":          disag,
        "cosine_agreement":           cosine,
        "transcript":                 transcript,
        "transcript_is_ground_truth": transcript_is_gt,
        "asr_confidence":             asr_confidence,
        "audio_available":            audio_feats["audio_available"],
        "text_available":             text_was_available,
        "audio_extraction_error":     audio_feats.get("error"),
    }
