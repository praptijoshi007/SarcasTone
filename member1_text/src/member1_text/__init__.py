"""Text sarcasm and irony models, plus ASR, for SarcasTone Member 1.

Importing this package does not check the interpreter. Call
``require_python_311`` or run ``python -m member1_text`` to validate
Python 3.11.
"""

from __future__ import annotations

import sys

REQUIRED_PYTHON = (3, 11)
__version__ = "0.1.0"


def python_requirement_message() -> str | None:
    """Return the Python 3.11 error sentence, or None when the interpreter matches."""
    major, minor, micro = sys.version_info[:3]
    if (major, minor) == REQUIRED_PYTHON:
        return None
    return (
        "member1_text requires Python 3.11. "
        f"This interpreter is Python {major}.{minor}.{micro}. "
        "Install Python 3.11 and run this package with that interpreter."
    )


def require_python_311() -> None:
    """Raise RuntimeError when this interpreter is not Python 3.11."""
    message = python_requirement_message()
    if message is not None:
        raise RuntimeError(message)
