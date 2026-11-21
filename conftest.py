"""Root conftest.py — adds shared/ and member3_fusion/ to sys.path for all tests."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent

for p in [
    str(_REPO),
    str(_REPO / "member3_fusion"),
    str(_REPO / "member3_fusion" / "src"),
    str(_REPO / "member1_text" / "src"),
    str(_REPO / "member2_audio" / "src"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)
