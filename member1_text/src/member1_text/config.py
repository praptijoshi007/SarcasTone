"""Load member1_text/config.yaml from the package root, not the process cwd."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Return the parsed YAML configuration."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return loaded


def resolve_repo_path(path_str: str) -> Path:
    """Resolve a config path that is written relative to the repository root."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def checkpoint_name(dataset: str, task: str, model_slug: str) -> str:
    return f"{dataset}__{task}__{model_slug}"


def model_slug(model_name: str) -> str:
    """Stable directory name for a Hugging Face model id or CLI alias."""
    aliases = {
        "roberta-base": "roberta-base",
        "deberta-v3-base": "deberta-v3-base",
        "microsoft/deberta-v3-base": "deberta-v3-base",
    }
    if model_name in aliases:
        return aliases[model_name]
    return model_name.split("/")[-1]


def huggingface_model_id(model_name: str) -> str:
    """Map a CLI model name to the Hugging Face repository id."""
    if model_name in {"deberta-v3-base", "microsoft/deberta-v3-base"}:
        return "microsoft/deberta-v3-base"
    if model_name == "roberta-base":
        return "roberta-base"
    return model_name
