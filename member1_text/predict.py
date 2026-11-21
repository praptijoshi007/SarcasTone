"""CLI prediction entry point for Member 1 Text Sarcasm Detection.

Usage:
  python member1_text/predict.py --text "It's just a privilege to watch your mind at work."
  python member1_text/predict.py --text "Oh brilliant idea" --context "The build failed."
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

from member1_text.preprocess import normalize_text

logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict sarcasm using Member 1 Text Semantics")
    parser.add_argument("--text", type=str, required=True, help="Utterance text to classify")
    parser.add_argument("--context", type=str, default=None, help="Preceding dialogue context")
    parser.add_argument("--speaker", type=str, default="UNKNOWN", help="Speaker name")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    return parser.parse_args()


def predict() -> None:
    args = parse_args()
    norm_text = normalize_text(args.text)
    norm_ctx = normalize_text(args.context) if args.context else None

    # Try loading through member1 api if PyTorch/checkpoint exists
    result = None
    try:
        from member1_text.api import predict_text
        ckpt_path = Path(args.checkpoint) if args.checkpoint else None
        res = predict_text(norm_text, norm_ctx, checkpoint=ckpt_path)
        p_sarcastic = float(res["probability"])
        label_val = res["label"]
        confidence = p_sarcastic if label_val in (1, "sarcastic") else (1.0 - p_sarcastic)
        result = {
            "label": label_val,
            "confidence": round(confidence, 4),
            "probabilities": {
                "non_sarcastic": round(1.0 - p_sarcastic, 4),
                "sarcastic": round(p_sarcastic, 4),
            },
            "source": "roberta_span_model",
        }
    except Exception:
        # Calibrated NLP text classifier fallback
        from member4.backend.pipeline import MultimodalPipeline
        p = MultimodalPipeline()
        ctx_list = [norm_ctx] if norm_ctx else []
        pred = p.predict(text=norm_text, context=ctx_list, speaker=args.speaker)
        result = {
            "label": pred.label,
            "confidence": pred.confidence,
            "probabilities": pred.probabilities,
            "source": "calibrated_text_semantics",
            "top_tokens": [t.token for t in pred.explanation.top_text_tokens[:4]],
        }

    print("=" * 60)
    print(" SARCAS TONE: MEMBER 1 TEXT SARCASM PREDICTION")
    print("=" * 60)
    print(f" Utterance  : {args.text}")
    if args.context:
        print(f" Context    : {args.context}")
    print(f" Speaker    : {args.speaker}")
    label_val = result["label"]
    if isinstance(label_val, int):
        label_str = "SARCASTIC" if label_val == 1 else "NON-SARCASTIC"
    else:
        label_str = str(label_val).replace("_", "-").upper()
    print(f" Prediction : {label_str}")
    print(f" Confidence : {result['confidence'] * 100:.1f}%")
    print(f" Probabilities: {result['probabilities']}")
    if "top_tokens" in result:
        print(f" Key Tokens : {result['top_tokens']}")
    print("=" * 60)


if __name__ == "__main__":
    predict()
