"""Pydantic request and response schemas for SarcasTone Member 4 REST API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Minimal dataclass fallback if pydantic is not installed
    def _to_jsonable(obj: Any) -> Any:
        if hasattr(obj, "__dict__"):
            return {k: _to_jsonable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [_to_jsonable(x) for x in obj]
        elif isinstance(obj, tuple):
            return [_to_jsonable(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        return obj

    class BaseModel:  # type: ignore
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

        def dict(self) -> Dict[str, Any]:
            return _to_jsonable(self)

        def model_dump(self) -> Dict[str, Any]:
            return _to_jsonable(self)

    def Field(*args: Any, **kwargs: Any) -> Any:  # type: ignore
        return kwargs.get("default", None)


class TextPredictRequest(BaseModel):
    """Request payload for text-only sarcasm prediction."""
    text: str = Field(..., description="The utterance text to classify")
    context: Optional[List[str]] = Field(default=None, description="Preceding dialogue turns")
    speaker: Optional[str] = Field(default=None, description="Speaker identifier")


class BatchItem(BaseModel):
    id: str = Field(..., description="Unique utterance ID")
    text: Optional[str] = Field(default=None, description="Transcript text")
    context: Optional[List[str]] = Field(default=None, description="Dialogue context")
    audio_path: Optional[str] = Field(default=None, description="Server-side audio path")
    speaker: Optional[str] = Field(default=None, description="Speaker name")


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""
    items: List[BatchItem] = Field(..., description="List of items to predict")


class TokenAttribution(BaseModel):
    token: str
    score: float


class AcousticAttribution(BaseModel):
    feature_name: str
    feature_value: float
    importance_score: float
    direction: str  # "promotes_sarcasm" or "promotes_literal"


class MultimodalExplanation(BaseModel):
    summary: str
    top_text_tokens: List[TokenAttribution] = Field(default_factory=list)
    top_acoustic_features: List[AcousticAttribution] = Field(default_factory=list)
    dissonance_explanation: str
    gating_explanation: str
    attention_faithfulness_note: str = (
        "Note: Attention weights and heuristic saliency indicate model focus "
        "and feature divergence, but should not be interpreted as absolute causal proofs."
    )


class PredictResponse(BaseModel):
    """Comprehensive multimodal sarcasm prediction response."""
    label: str = Field(..., description="'sarcastic' or 'non_sarcastic'")
    label_index: int = Field(..., description="1 for sarcastic, 0 for non-sarcastic")
    probabilities: Dict[str, float] = Field(..., description="Calibrated class probabilities")
    confidence: float = Field(..., description="Prediction confidence (max probability)")
    entropy: float = Field(..., description="Predictive entropy in nats")
    uncertain: bool = Field(..., description="Whether prediction exceeds uncertainty threshold")
    uncertainty_reason: Optional[str] = None

    transcript: str = Field(..., description="Utterance transcript")
    transcript_source: str = Field(..., description="'user_provided', 'asr_transcribed', or 'none'")
    text_available: bool = True
    audio_available: bool = False
    modalities_used: List[str] = Field(default_factory=list)

    modality_weights: Dict[str, float] = Field(
        default_factory=lambda: {"text": 0.5, "audio": 0.5},
        description="Adaptive confidence gating weights",
    )
    psid_dissonance_score: float = Field(
        default=0.0, description="Prosody-Semantic Incongruence score"
    )
    cosine_agreement: Optional[float] = Field(
        default=None, description="Cross-modal latent cosine agreement"
    )

    audio_quality: Dict[str, Any] = Field(default_factory=dict)
    prosody_summary: Dict[str, Any] = Field(default_factory=dict)
    emotion_prediction: Dict[str, Any] = Field(default_factory=dict)

    unimodal_comparisons: Dict[str, Any] = Field(
        default_factory=dict,
        description="Individual predictions for text-only vs audio-only vs multimodal",
    )
    explanation: Optional[MultimodalExplanation] = None

    latency_ms: float = 0.0
    timestamp: str = ""


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    service: str = "SarcasTone Multimodal Prediction Server"
    active_device: str = "cpu"
    modules: Dict[str, bool] = Field(default_factory=dict)
    uptime_seconds: float = 0.0


class ModelInfoResponse(BaseModel):
    architecture: str
    version: str
    text_encoder: str
    audio_encoder: str
    fusion_mechanism: str
    dimensions: Dict[str, int]
    target_sample_rate_hz: int
    checkpoints_loaded: Dict[str, bool]
