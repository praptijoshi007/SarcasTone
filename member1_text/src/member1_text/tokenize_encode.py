"""Target and context encoding with non-overlapping span masks.

RoBERTa pairs are separated by the ``</s></s>`` boundary. DeBERTa-v3 pairs
fall back to token-type ids because that tokenizer uses a single separator.
Special tokens and padding are never part of either span.
"""

from __future__ import annotations

from typing import Any


def tokenizer_family(tokenizer: Any) -> str:
    blob = f"{type(tokenizer).__name__} {type(tokenizer).__module__}".lower()
    if "deberta" in blob:
        return "deberta"
    if "roberta" in blob:
        return "roberta"
    return "other"


def assert_tokenizer_matches_model(model_name: str, tokenizer: Any) -> None:
    """Raise when a RoBERTa model is paired with a non-RoBERTa tokenizer, or the reverse."""
    slug = model_name.lower()
    family = tokenizer_family(tokenizer)
    if "deberta" in slug:
        expected = "deberta"
        required = "DeBERTa SentencePiece tokenizer"
    elif "roberta" in slug:
        expected = "roberta"
        required = "RoBERTa byte-level BPE tokenizer"
    else:
        raise RuntimeError(f"Unsupported model {model_name}.")
    if family != expected:
        raise RuntimeError(
            f"Model {model_name} requires the {required}, got {type(tokenizer).__name__}."
        )


def _token_id(tokenizer: Any, *names: str) -> int:
    for name in names:
        value = getattr(tokenizer, name, None)
        if value is not None:
            return int(value)
    raise RuntimeError(f"Tokenizer {type(tokenizer).__name__} is missing {names[0]}.")


def _rows(value: Any) -> list[list[int]]:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if value and isinstance(value[0], int):
        return [list(value)]
    return [list(row) for row in value]


def build_span_masks(
    input_ids: list[list[int]] | Any,
    attention_mask: list[list[int]] | Any,
    *,
    cls_id: int,
    sep_id: int,
    pad_id: int,
    token_type_ids: list[list[int]] | Any | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    """Build context and target masks that exclude specials, padding, and each other."""
    ids_batch = _rows(input_ids)
    attention_batch = _rows(attention_mask)
    types_batch = _rows(token_type_ids) if token_type_ids is not None else None
    context_batch: list[list[int]] = []
    target_batch: list[list[int]] = []
    specials = {cls_id, sep_id, pad_id}
    for row_index, ids in enumerate(ids_batch):
        attention = attention_batch[row_index]
        boundary = None
        for index in range(len(ids) - 1):
            if ids[index] == sep_id and ids[index + 1] == sep_id and attention[index] and attention[index + 1]:
                boundary = index
                break
        context = [0] * len(ids)
        target = [0] * len(ids)
        if boundary is not None:
            for index, token in enumerate(ids[:boundary]):
                if attention[index] and token not in specials:
                    context[index] = 1
            for index in range(boundary + 2, len(ids)):
                token = ids[index]
                if attention[index] and token not in specials:
                    target[index] = 1
        elif types_batch is not None and any(types_batch[row_index]):
            for index, token in enumerate(ids):
                if not attention[index] or token in specials:
                    continue
                if types_batch[row_index][index] == 0:
                    context[index] = 1
                else:
                    target[index] = 1
        else:
            for index, token in enumerate(ids):
                if attention[index] and token not in specials:
                    target[index] = 1
        if any(context[index] and target[index] for index in range(len(ids))):
            raise RuntimeError("Context and target masks overlap.")
        context_batch.append(context)
        target_batch.append(target)
    return context_batch, target_batch


def encode_pair(
    tokenizer: Any,
    target: str,
    context: str | None = None,
    *,
    max_length: int = 256,
    model_name: str | None = None,
) -> dict[str, torch.Tensor]:
    """Encode ``<s> context </s></s> target </s>`` when context is present."""
    if model_name is not None:
        assert_tokenizer_matches_model(model_name, tokenizer)
    if context:
        encoded = tokenizer(
            context,
            target,
            truncation="only_first",
            max_length=max_length,
            padding="max_length",
            return_token_type_ids=True,
        )
    else:
        encoded = tokenizer(
            target,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_token_type_ids=True,
        )
    cls_id = _token_id(tokenizer, "cls_token_id", "bos_token_id")
    sep_id = _token_id(tokenizer, "sep_token_id", "eos_token_id")
    pad_id = _token_id(tokenizer, "pad_token_id")
    type_ids = encoded["token_type_ids"] if "token_type_ids" in encoded else None
    context_mask, target_mask = build_span_masks(
        encoded["input_ids"],
        encoded["attention_mask"],
        cls_id=cls_id,
        sep_id=sep_id,
        pad_id=pad_id,
        token_type_ids=type_ids,
    )
    import torch

    return {
        "input_ids": torch.tensor(_rows(encoded["input_ids"]), dtype=torch.long),
        "attention_mask": torch.tensor(_rows(encoded["attention_mask"]), dtype=torch.long),
        "context_mask": torch.tensor(context_mask, dtype=torch.long),
        "target_mask": torch.tensor(target_mask, dtype=torch.long),
    }
