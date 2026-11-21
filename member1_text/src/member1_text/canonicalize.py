"""Canonical records produced by a dataset loader.

A ``CanonicalExample`` is an output of a loader. It is not a claim that the
source files share a schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
import unicodedata
from typing import Any


class DatasetNotAvailableError(FileNotFoundError):
    """Raised when a legal local corpus file is absent.

    Callers must stop the command and must not write a metric file.
    """

    def __init__(self, dataset: str, path: str, source_url: str) -> None:
        self.dataset = dataset
        self.path = path
        self.source_url = source_url
        super().__init__(
            f"{dataset} is not available at {path}. "
            f"Place the legal local file there. Source: {source_url}. "
            "No metrics were written."
        )


@dataclass
class CanonicalExample:
    example_id: str
    dataset: str
    split: str
    text: str
    context_turns: list[dict[str, str]]
    text_source: str
    label: int | float | str
    label_meaning: str
    extra: dict[str, Any] = field(default_factory=dict)
    overlap_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_overlap_text(text: str) -> str:
    """Normalize target text used only to detect MUStARD / MUStARD++ leakage."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def require_local_file(dataset: str, path: Any, source_url: str):
    """Return a Path when the file exists, otherwise raise with the source URL."""
    from pathlib import Path

    candidate = Path(path)
    if not candidate.is_file():
        raise DatasetNotAvailableError(dataset, str(candidate), source_url)
    return candidate
