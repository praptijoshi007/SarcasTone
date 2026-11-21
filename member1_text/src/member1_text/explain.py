"""Captum Integrated Gradients for a saved RoBERTa checkpoint.

Attributions use the pad embedding as the baseline. They are not causal,
subword pieces are not words, and a highlight is not a sarcasm reason.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any

import torch

from member1_text import require_python_311
from member1_text.model import load_span_model
from member1_text.preprocess import normalize_text
from member1_text.tokenize_encode import assert_tokenizer_matches_model, encode_pair

LIMITS = (
    "Limits: attributions use the pad embedding as the baseline; they are not causal; "
    "subword pieces are not words; context tokens and target tokens can both receive mass; "
    "a highlight is not a sarcasm reason."
)


def explain_text(
    checkpoint: str | Path,
    text: str,
    context: str | None = None,
    *,
    output_path: str | Path | None = None,
    n_steps: int = 32,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Write an HTML highlight and return token and word attributions."""
    from transformers import AutoTokenizer

    checkpoint_path = Path(checkpoint)
    payload_name = _model_name(checkpoint_path)
    if "deberta" in payload_name or "roberta" not in payload_name:
        raise RuntimeError(
            "Captum explanations run on a saved RoBERTa checkpoint. "
            f"{checkpoint_path} is {payload_name or 'not a RoBERTa model'}."
        )
    device = _device(device_name)
    model = load_span_model(checkpoint_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    assert_tokenizer_matches_model(model.model_name or payload_name, tokenizer)
    model.eval()
    normalized_text = normalize_text(text)
    normalized_context = normalize_text(context) if context else None
    encoded = encode_pair(
        tokenizer,
        normalized_text,
        normalized_context,
        max_length=model.max_length,
        model_name=model.model_name or payload_name,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    embeddings = model.encoder.get_input_embeddings()
    inputs_embeds = embeddings(encoded["input_ids"]).detach()
    pad_id = int(tokenizer.pad_token_id)
    baseline = embeddings.weight[pad_id].detach().view(1, 1, -1).expand_as(inputs_embeds)

    def forward(
        inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = model(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            context_mask=context_mask,
            target_mask=target_mask,
        )
        logits = output["logits"]
        if logits.size(-1) == 1:
            return logits.view(-1)
        return logits[:, 1]

    from captum.attr import IntegratedGradients

    integrated = IntegratedGradients(forward)
    attributions = integrated.attribute(
        inputs_embeds,
        baselines=baseline,
        additional_forward_args=(
            encoded["attention_mask"],
            encoded["context_mask"],
            encoded["target_mask"],
        ),
        n_steps=n_steps,
    )
    token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu()
    input_ids = encoded["input_ids"].squeeze(0).detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    context_flags = encoded["context_mask"].squeeze(0).detach().cpu().tolist()
    target_flags = encoded["target_mask"].squeeze(0).detach().cpu().tolist()
    token_rows = []
    for token, score, in_context, in_target in zip(tokens, token_scores.tolist(), context_flags, target_flags):
        if not in_context and not in_target:
            continue
        token_rows.append(
            {
                "token": token,
                "score": float(score),
                "span": "context" if in_context else "target",
            }
        )
    words = _group_words(token_rows)
    document = _render_html(words, text=normalized_text, context=normalized_context)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(document, encoding="utf-8")
    return {
        "limits": LIMITS,
        "tokens": token_rows,
        "words": words,
        "html_path": None if output_path is None else str(output_path),
        "baseline": "pad_embedding",
    }


def _group_words(token_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in token_rows:
        piece = row["token"]
        starts_word = piece.startswith("Ġ") or piece.startswith("▁") or current is None or current["span"] != row["span"]
        surface = piece[1:] if piece.startswith(("Ġ", "▁")) else piece
        if starts_word:
            if current is not None:
                words.append(current)
            current = {"word": surface, "score": row["score"], "span": row["span"]}
        else:
            assert current is not None
            current["word"] += surface
            current["score"] += row["score"]
    if current is not None:
        words.append(current)
    return words


def _render_html(words: list[dict[str, Any]], *, text: str, context: str | None) -> str:
    limit = max((abs(word["score"]) for word in words), default=1.0) or 1.0
    spans = []
    last_span = None
    for word in words:
        if word["span"] != last_span:
            spans.append(f"<p><strong>{html.escape(word['span'])}</strong></p>")
            last_span = word["span"]
        ratio = max(-1.0, min(1.0, word["score"] / limit))
        if ratio >= 0:
            red, green = 255, int(255 * (1.0 - ratio))
        else:
            red, green = int(255 * (1.0 + ratio)), 255
        color = f"rgb({red},{green},160)"
        spans.append(
            f'<span style="background:{color}" title="{word["score"]:.4f}">'
            f'{html.escape(word["word"])}</span> '
        )
    body = "".join(spans)
    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Integrated Gradients</title></head>"
        f"<body><p>{html.escape(LIMITS)}</p>"
        f"<p>Target: {html.escape(text)}</p>"
        f"<p>Context: {html.escape(context or '')}</p>{body}</body></html>"
    )


def _model_name(checkpoint: Path) -> str:
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No checkpoint config at {checkpoint}. Captum was not run and no explanation was invented."
        )
    import json

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("architectures") == ["TfidfLogReg"]:
        raise RuntimeError("Captum is not run on the TF-IDF baseline.")
    return str(payload.get("model_name", ""))


def _device(device_name: str | None) -> torch.device:
    import os

    choice = device_name or os.environ.get("MEMBER1_DEVICE")
    if choice:
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrated Gradients for a RoBERTa checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--context", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    require_python_311()
    args = parse_args(argv)
    output = args.output
    if output is None:
        output = str(Path(args.checkpoint) / "explanation.html")
    result = explain_text(
        args.checkpoint,
        args.text,
        args.context,
        output_path=output,
        device_name=args.device,
    )
    print(LIMITS)
    print(result["html_path"])


if __name__ == "__main__":
    main()
