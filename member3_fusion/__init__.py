"""member3_fusion top-level package.

Cross-modal fusion module for SarcasTone (Textual Semantics + Speech Prosody).
"""

from __future__ import annotations

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _PKG_ROOT / "src"
_REPO_ROOT = _PKG_ROOT.parent

for _p in [str(_REPO_ROOT), str(_PKG_ROOT), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

__version__ = "1.0.0"
