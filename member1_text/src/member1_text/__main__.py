"""CLI check that this interpreter is Python 3.11.

Usage, from the repository root with PYTHONPATH=member1_text/src:

    python -m member1_text
"""

from __future__ import annotations

import sys

from member1_text import __version__, python_requirement_message


def main() -> None:
    message = python_requirement_message()
    if message is not None:
        print(message, file=sys.stderr)
        sys.exit(1)
    print(__version__)


if __name__ == "__main__":
    main()
