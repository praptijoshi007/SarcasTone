"""Configuration management for Member 4 backend and services."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "member4" / "configs" / "system_config.yaml"


def load_system_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Load and parse the YAML system configuration for Member 4."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        # Fallback to defaults
        return {
            "project": {"name": "SarcasTone-Member4", "version": "1.0.0"},
            "backend": {
                "host": "0.0.0.0",
                "port": 8000,
                "max_upload_size_bytes": 26214400,
                "allowed_audio_formats": [".wav", ".mp3", ".m4a", ".flac"],
                "prediction_log_path": str(REPO_ROOT / "member4" / "backend" / "logs" / "predictions.jsonl"),
            },
            "model": {
                "temperature": 1.25,
                "uncertainty_threshold": 0.55,
                "decision_threshold": 0.50,
                "psid_enabled": True,
                "adaptive_gating_enabled": True,
            },
        }

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# Global singleton configuration
CONFIG = load_system_config()
