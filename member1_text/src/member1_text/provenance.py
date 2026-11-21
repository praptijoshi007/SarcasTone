"""SHA-256 provenance records. Corpus text is never stored in these files."""

from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_provenance(
    local_file: Path,
    *,
    source_url: str,
    license_note: str,
    citation: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source_url": source_url,
        "access_date": date.today().isoformat(),
        "sha256": sha256_file(local_file),
        "license_note": license_note,
        "citation": citation,
        "local_file_name": Path(local_file).name,
        "bytes": Path(local_file).stat().st_size,
    }
    if extra:
        payload["extra"] = extra
    return payload


def write_provenance(output_path: Path, payload: dict[str, Any]) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
