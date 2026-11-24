"""FastAPI REST API Server for SarcasTone Multimodal Sarcasm Detection.

Endpoints:
  POST /predict       : Full multimodal sarcasm prediction (audio upload + text + context)
  POST /predict/text  : Text-only sarcasm prediction
  POST /predict/audio : Audio-only sarcasm prediction (with ASR)
  POST /batch-predict : Batch inference across multiple utterances
  GET  /health        : Health check, system status, uptime
  GET  /model-info    : Model architecture metadata, contract dimensions
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from member4.backend.config import CONFIG
from member4.backend.pipeline import MultimodalPipeline
from member4.backend.schemas import (
    BatchPredictRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictResponse,
    TextPredictRequest,
)
from shared.constants import (
    AUDIO_HIDDEN,
    PROSODY_DIM,
    QUALITY_DIM,
    TARGET_SAMPLE_RATE,
    TEXT_HIDDEN,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("sarcas_api")

# Server start timestamp
SERVER_START_TIME = time.time()

# Allowed audio file extensions & max size
ALLOWED_EXTENSIONS = set(CONFIG.get("backend", {}).get("allowed_audio_formats", [".wav", ".mp3", ".m4a", ".flac", ".ogg"]))
MAX_UPLOAD_SIZE = int(CONFIG.get("backend", {}).get("max_upload_size_bytes", 26214400))  # 25 MB

# Initialize pipeline
pipeline = MultimodalPipeline()

# Try FastAPI setup
try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    app = FastAPI(
        title="SarcasTone Multimodal Sarcasm Detection API",
        description="Production API integrating Textual Semantics (RoBERTa) and Speech Prosody (Wav2Vec2 / GeMAPS) with Cross-Modal Attention and PSID.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check() -> HealthResponse:
        """System health and hardware status."""
        uptime = round(time.time() - SERVER_START_TIME, 2)
        device = "cuda" if "torch" in sys.modules and getattr(sys.modules.get("torch"), "cuda", None) and sys.modules["torch"].cuda.is_available() else "cpu"
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            service="SarcasTone Multimodal Prediction Server",
            active_device=device,
            modules={
                "member1_text": (REPO_ROOT / "member1_text").is_dir(),
                "member2_audio": (REPO_ROOT / "member2_audio").is_dir(),
                "member3_fusion": (REPO_ROOT / "member3_fusion").is_dir(),
                "member4_system": True,
            },
            uptime_seconds=uptime,
        )

    @app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
    async def model_info() -> ModelInfoResponse:
        """Model architecture metadata, feature dimensions, and contract specs."""
        return ModelInfoResponse(
            architecture="SarcasTone Cross-Modal Attention with PSID & Adaptive Confidence Gating",
            version="1.0.0",
            text_encoder="RoBERTa-base (768-d SpanPooling)",
            audio_encoder="Wav2Vec2 (768-d) + Handcrafted Prosody (32-d)",
            fusion_mechanism="Dual Multi-Head Cross-Attention + Dynamic Confidence Gating",
            dimensions={
                "text_hidden": TEXT_HIDDEN,
                "audio_hidden": AUDIO_HIDDEN,
                "prosody_dim": PROSODY_DIM,
                "quality_dim": QUALITY_DIM,
                "fusion_logits": 2,
            },
            target_sample_rate_hz=TARGET_SAMPLE_RATE,
            checkpoints_loaded={
                "fusion_checkpoint": pipeline.fusion_model is not None,
            },
        )

    @app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
    async def predict_multimodal_endpoint(
        audio: Optional[UploadFile] = File(default=None),
        text: Optional[str] = Form(default=None),
        context: Optional[str] = Form(default=None),
        speaker: Optional[str] = Form(default=None),
    ) -> PredictResponse:
        """Full multimodal sarcasm detection from uploaded audio, text, and dialogue context."""
        tmp_audio_path: Optional[Path] = None

        if audio is not None and audio.filename:
            # Validate extension
            ext = Path(audio.filename).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid audio format '{ext}'. Allowed: {sorted(list(ALLOWED_EXTENSIONS))}",
                )

            # Save uploaded audio to temp file
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                shutil.copyfileobj(audio.file, tmp)
                tmp_audio_path = Path(tmp.name)

            # Validate size
            if tmp_audio_path.stat().st_size > MAX_UPLOAD_SIZE:
                tmp_audio_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Audio file exceeds maximum size of {MAX_UPLOAD_SIZE / (1024*1024):.1f} MB",
                )

        # Parse context list if passed as JSON string or newline-separated
        ctx_list = []
        if context:
            try:
                parsed = json.loads(context)
                if isinstance(parsed, list):
                    ctx_list = [str(x) for x in parsed]
                else:
                    ctx_list = [str(parsed)]
            except Exception:
                ctx_list = [c.strip() for c in context.split("\n") if c.strip()]

        try:
            result = pipeline.predict(
                audio_path=tmp_audio_path,
                text=text,
                context=ctx_list,
                speaker=speaker,
            )
            return result
        finally:
            if tmp_audio_path and tmp_audio_path.is_file():
                tmp_audio_path.unlink(missing_ok=True)

    @app.post("/predict/text", response_model=PredictResponse, tags=["Prediction"])
    async def predict_text_endpoint(payload: TextPredictRequest) -> PredictResponse:
        """Text-only sarcasm prediction."""
        if not payload.text or not payload.text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text cannot be empty.",
            )
        return pipeline.predict(
            audio_path=None,
            text=payload.text,
            context=payload.context,
            speaker=payload.speaker,
        )

    @app.post("/predict/audio", response_model=PredictResponse, tags=["Prediction"])
    async def predict_audio_endpoint(
        audio: UploadFile = File(...),
        speaker: Optional[str] = Form(default=None),
    ) -> PredictResponse:
        """Audio-only sarcasm prediction with automatic speech-to-text transcription."""
        if not audio.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No audio file provided.",
            )

        ext = Path(audio.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid audio format '{ext}'. Allowed: {sorted(list(ALLOWED_EXTENSIONS))}",
            )

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            shutil.copyfileobj(audio.file, tmp)
            tmp_path = Path(tmp.name)

        try:
            return pipeline.predict(
                audio_path=tmp_path,
                text=None,
                context=None,
                speaker=speaker,
            )
        finally:
            if tmp_path.is_file():
                tmp_path.unlink(missing_ok=True)

    @app.post("/batch-predict", tags=["Prediction"])
    async def batch_predict_endpoint(payload: BatchPredictRequest) -> Dict[str, Any]:
        """Batch evaluation across multiple utterances."""
        results = []
        for item in payload.items:
            res = pipeline.predict(
                audio_path=item.audio_path,
                text=item.text,
                context=item.context,
                speaker=item.speaker,
            )
            results.append({
                "id": item.id,
                "label": res.label,
                "probabilities": res.probabilities,
                "confidence": res.confidence,
                "uncertain": res.uncertain,
                "psid_dissonance_score": res.psid_dissonance_score,
            })
        return {
            "total_items": len(results),
            "results": results,
        }

except ImportError:
    # Starlette fallback if fastapi package is not yet installed
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def starlette_health(request: Any) -> JSONResponse:
        uptime = round(time.time() - SERVER_START_TIME, 2)
        return JSONResponse({
            "status": "healthy",
            "version": "1.0.0",
            "service": "SarcasTone Multimodal Prediction Server (Starlette/Uvicorn)",
            "uptime_seconds": uptime,
            "modules": {
                "member1_text": (REPO_ROOT / "member1_text").is_dir(),
                "member2_audio": (REPO_ROOT / "member2_audio").is_dir(),
                "member3_fusion": (REPO_ROOT / "member3_fusion").is_dir(),
                "member4_system": True,
            },
        })

    async def starlette_model_info(request: Any) -> JSONResponse:
        return JSONResponse({
            "architecture": "SarcasTone Cross-Modal Attention with PSID & Adaptive Confidence Gating",
            "version": "1.0.0",
            "text_encoder": "RoBERTa-base (768-d SpanPooling)",
            "audio_encoder": "Wav2Vec2 (768-d) + Handcrafted Prosody (32-d)",
            "fusion_mechanism": "Dual Multi-Head Cross-Attention + Dynamic Confidence Gating",
            "dimensions": {
                "text_hidden": TEXT_HIDDEN,
                "audio_hidden": AUDIO_HIDDEN,
                "prosody_dim": PROSODY_DIM,
                "quality_dim": QUALITY_DIM,
                "fusion_logits": 2,
            },
        })

    async def starlette_predict_text(request: Any) -> JSONResponse:
        try:
            data = await request.json()
        except Exception:
            return JSONResponse({"detail": "Invalid JSON body."}, status_code=400)
        text = data.get("text", "")
        if not text or not str(text).strip():
            return JSONResponse({"detail": "Text cannot be empty."}, status_code=400)
        context = data.get("context", [])
        speaker = data.get("speaker")
        res = pipeline.predict(text=text, context=context, speaker=speaker)
        resp_dict = res.model_dump() if hasattr(res, "model_dump") else (res.dict() if hasattr(res, "dict") else res.__dict__)
        return JSONResponse(resp_dict)

    async def starlette_predict_general(request: Any) -> JSONResponse:
        ct = request.headers.get("content-type", "")
        if "application/json" in ct:
            try:
                data = await request.json()
            except Exception:
                return JSONResponse({"detail": "Invalid JSON."}, status_code=400)
            text = data.get("text")
            context = data.get("context")
            speaker = data.get("speaker")
            audio_path = data.get("audio_path")
            res = pipeline.predict(audio_path=audio_path, text=text, context=context, speaker=speaker)
            resp_dict = res.model_dump() if hasattr(res, "model_dump") else (res.dict() if hasattr(res, "dict") else res.__dict__)
            return JSONResponse(resp_dict)
        elif "multipart/form-data" in ct:
            form = await request.form()
            text = form.get("text")
            speaker = form.get("speaker")
            audio_file = form.get("audio")
            tmp_path = None
            if audio_file and hasattr(audio_file, "filename") and audio_file.filename:
                ext = Path(audio_file.filename).suffix or ".wav"
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    content = await audio_file.read()
                    tmp.write(content)
                    tmp_path = Path(tmp.name)
            try:
                res = pipeline.predict(audio_path=tmp_path, text=str(text) if text else None, speaker=str(speaker) if speaker else None)
                resp_dict = res.model_dump() if hasattr(res, "model_dump") else (res.dict() if hasattr(res, "dict") else res.__dict__)
                return JSONResponse(resp_dict)
            finally:
                if tmp_path and tmp_path.is_file():
                    tmp_path.unlink(missing_ok=True)
        else:
            return JSONResponse({"detail": "Unsupported content type. Send JSON or multipart/form-data."}, status_code=400)

    async def starlette_batch_predict(request: Any) -> JSONResponse:
        try:
            data = await request.json()
        except Exception:
            return JSONResponse({"detail": "Invalid JSON."}, status_code=400)
        items = data.get("items", [])
        results = []
        for it in items:
            res = pipeline.predict(
                audio_path=it.get("audio_path"),
                text=it.get("text"),
                context=it.get("context"),
                speaker=it.get("speaker"),
            )
            results.append({
                "id": it.get("id"),
                "label": res.label,
                "confidence": res.confidence,
                "probabilities": res.probabilities,
                "psid_dissonance_score": res.psid_dissonance_score,
            })
        return JSONResponse({"total_items": len(results), "results": results})

    app = Starlette(routes=[
        Route("/health", starlette_health, methods=["GET"]),
        Route("/model-info", starlette_model_info, methods=["GET"]),
        Route("/predict", starlette_predict_general, methods=["POST"]),
        Route("/predict/text", starlette_predict_text, methods=["POST"]),
        Route("/predict/audio", starlette_predict_general, methods=["POST"]),
        Route("/batch-predict", starlette_batch_predict, methods=["POST"]),
    ])


if __name__ == "__main__":
    import uvicorn
    host = CONFIG.get("backend", {}).get("host", "0.0.0.0")
    port = int(CONFIG.get("backend", {}).get("port", 8000))
    print(f"Starting SarcasTone API on http://{host}:{port} ...")
    uvicorn.run("member4.backend.main:app", host=host, port=port, reload=True)
