"""conftest.py for member3_fusion tests.

Inserts the correct sys.path entries so tests can import from:
  - member3_fusion/src      (as a package root, allowing `from src.xxx import`)
  - shared/                 (repo-level shared utilities)
  - member3_fusion/         (so `from member3_fusion.src.xxx import` works)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repository root
_REPO = Path(__file__).resolve().parents[2]
# member3_fusion package root
_M3   = _REPO / "member3_fusion"
# member3_fusion/src — so `from src.xxx import` works inside tests
_M3_SRC = _M3 / "src"

for p in [str(_REPO), str(_M3), str(_M3_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)
