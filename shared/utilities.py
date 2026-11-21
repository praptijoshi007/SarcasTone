"""File, seeding, and device helpers used by every module."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def write_json(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto"):
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
