"""faster-whisper transcription.

Gold text is never overwritten. CPU uses int8 and CUDA uses float16.
Non-wav audio is converted with ffmpeg to 16 kHz mono. An empty transcript
is an error.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from member1_text import require_python_311
from member1_text.config import load_config

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".mp4"}


def default_asr_device() -> str:
    override = os.environ.get("MEMBER1_DEVICE")
    if override:
        return override
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def require_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if not found:
        raise RuntimeError("ffmpeg is required to convert audio and was not found on PATH.")
    return found


def convert_to_wav(source: Path, destination: Path) -> None:
    ffmpeg = require_ffmpeg()
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or "ffmpeg exited with a non-zero status."
        raise RuntimeError(detail)


def transcribe_audio(
    audio_path: str | Path,
    *,
    model_name: str | None = None,
    backend: Any | None = None,
    device: str | None = None,
    beam_size: int | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    """Transcribe one audio file. ``backend`` is for tests and is not a silent fallback."""
    path = Path(audio_path)
    suffix = path.suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        raise ValueError(
            f"Unsupported audio suffix {path.suffix!r}. "
            f"Accepted suffixes: {', '.join(sorted(AUDIO_SUFFIXES))}."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    settings = load_config()["asr"]
    model_name = model_name or settings["model"]
    language = language or settings["language"]
    beam_size = int(settings["beam_size"] if beam_size is None else beam_size)
    resolved_device = device or default_asr_device()
    compute_type = settings["cuda_compute_type"] if resolved_device.startswith("cuda") else settings["cpu_compute_type"]
    temporary: Path | None = None
    wav_path = path
    if suffix != ".wav":
        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        handle.close()
        temporary = Path(handle.name)
        convert_to_wav(path, temporary)
        wav_path = temporary
    try:
        if backend is None:
            raw = _transcribe_with_whisper(
                wav_path,
                model_name=model_name,
                device=resolved_device,
                compute_type=compute_type,
                language=language,
                beam_size=beam_size,
            )
        else:
            raw = backend.transcribe(wav_path, language=language, beam_size=beam_size)
        text = str(raw.get("text", "")).strip()
        if not text:
            raise RuntimeError("ASR returned an empty transcript. No empty string was stored.")
        segments = raw.get("segments") or []
        return {
            "text": text,
            "language": raw.get("language", language),
            "language_probability": raw.get("language_probability"),
            "duration_sec": raw.get("duration_sec"),
            "segments": segments,
            "model_name": model_name,
            "device": resolved_device,
            "compute_type": compute_type,
            "text_source": "asr",
        }
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def transcribe_mustard_utterance(audio_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Transcribe a MUStARD utterance clip. Context mp4s are rejected."""
    path = Path(audio_path)
    if "utterances_final" not in path.parts:
        raise ValueError(
            "Only MUStARD utterance clips inside utterances_final/ are transcribed here. "
            "Context clips are not transcribed."
        )
    return transcribe_audio(path, **kwargs)


def _transcribe_with_whisper(
    wav_path: Path,
    *,
    model_name: str,
    device: str,
    compute_type: str,
    language: str,
    beam_size: int,
) -> dict[str, Any]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. No transcript was invented."
        ) from exc
    whisper_device = "cuda" if device.startswith("cuda") else "cpu"
    try:
        model = WhisperModel(model_name, device=whisper_device, compute_type=compute_type)
        segments, info = model.transcribe(str(wav_path), language=language, beam_size=beam_size)
    except Exception as exc:
        raise RuntimeError(f"faster-whisper failed on {wav_path.name}. No transcript was invented.") from exc
    collected = []
    pieces = []
    for segment in segments:
        piece = segment.text or ""
        pieces.append(piece)
        collected.append({"start": float(segment.start), "end": float(segment.end), "text": piece})
    return {
        "text": "".join(pieces).strip(),
        "language": getattr(info, "language", language),
        "language_probability": getattr(info, "language_probability", None),
        "duration_sec": getattr(info, "duration", None),
        "segments": collected,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe an audio file with faster-whisper.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    require_python_311()
    args = parse_args(argv)
    result = transcribe_audio(args.audio, model_name=args.model, device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
