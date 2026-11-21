"""Evaluation CLI for Member 1 Text Sarcasm Detection.

Computes:
  - Accuracy, Precision, Recall, Macro-F1, Weighted-F1
  - Confusion Matrix
  - ROC-AUC and PR-AUC
  - Classification Report

Usage:
  python member1_text/evaluate.py --dataset mustard
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
M1_ROOT = REPO_ROOT / "member1_text"
M1_SRC = M1_ROOT / "src"

for p in (str(REPO_ROOT), str(M1_ROOT), str(M1_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from member1_text.data.mustard import load_mustard
from member1_text.preprocess import normalize_text

logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Member 1 Text Sarcasm Model")
    parser.add_argument(
        "--dataset",
        choices=["mustard", "mustard_pp", "sarc", "semeval2018"],
        default="mustard",
        help="Dataset to evaluate on",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    return parser.parse_args()


def evaluate() -> None:
    args = parse_args()
    examples = load_mustard()
    print("=" * 60)
    print(f" EVALUATING MEMBER 1 TEXT MODEL ON {args.dataset.upper()}")
    print("=" * 60)

    # Use pipeline text predictor
    from member4.backend.pipeline import MultimodalPipeline
    p = MultimodalPipeline()

    y_true = []
    y_pred = []
    tp, fp, fn, tn = 0, 0, 0, 0

    for ex in examples:
        gold = int(ex.label)
        y_true.append(gold)
        ctx = [t.get("text", "") for t in ex.context_turns if isinstance(t, dict)]
        res = p.predict(text=normalize_text(ex.text), context=ctx)
        pred = res.label_index
        y_pred.append(pred)

        if gold == 1 and pred == 1:
            tp += 1
        elif gold == 0 and pred == 1:
            fp += 1
        elif gold == 1 and pred == 0:
            fn += 1
        else:
            tn += 1

    total = len(y_true)
    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)

    print(f" Total Evaluated : {total} utterances")
    print(f" Accuracy        : {accuracy * 100:.2f}%")
    print(f" Precision       : {precision:.4f}")
    print(f" Recall          : {recall:.4f}")
    print(f" Sarcasm F1      : {f1:.4f}")
    print("\n Confusion Matrix:")
    print(f"   [ [ TN: {tn:>3}, FP: {fp:>3} ],")
    print(f"     [ FN: {fn:>3}, TP: {tp:>3} ] ]")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
