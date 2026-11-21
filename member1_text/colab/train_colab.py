"""Standalone Colab runner script for Member 1 Text Sarcasm Training."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    print("=" * 60)
    print(" Running SarcasTone Member 1 Colab Fine-Tuning")
    print("=" * 60)

    cmd = [
        sys.executable,
        "member1_text/train.py",
        "--dataset", "mustard",
        "--model-name", "roberta-base",
        "--epochs", "5",
        "--batch-size", "16",
        "--split", "deployment_refit",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
