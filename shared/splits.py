"""Speaker-independent splits and cross-dataset duplicate checks."""

from __future__ import annotations

import random
import unicodedata
from collections import defaultdict

from shared.constants import GENERIC_SPEAKERS
from shared.examples import UtteranceExample


def normalize_for_dedup(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def speaker_group(example: UtteranceExample) -> str:
    speaker = (example.speaker or "").strip().upper()
    show = (example.show or "").strip().upper()
    if speaker in GENERIC_SPEAKERS:
        return f"ID::{example.dataset}::{example.example_id}"
    if show:
        return f"{show}::{speaker}"
    return speaker


def assign_speaker_splits(
    examples: list[UtteranceExample],
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[UtteranceExample]]:
    """Put every speaker entirely in train, validation, or test.

    Speakers are shuffled with ``seed``. The shuffle order then fills the
    test quota, then the validation quota. Remaining speakers form the
    training split. An example never shares its speaker group with another split.
    """

    if not 0 < val_ratio < 1 or not 0 < test_ratio < 1 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be positive and sum to less than 1")
    groups: dict[str, list[UtteranceExample]] = defaultdict(list)
    for example in examples:
        groups[speaker_group(example)].append(example)
    speakers = list(groups)
    random.Random(seed).shuffle(speakers)
    total = len(examples)
    target_test = max(1, int(round(total * test_ratio))) if total > 2 else 1
    target_val = max(1, int(round(total * val_ratio))) if total > 2 else 0
    buckets: dict[str, list[UtteranceExample]] = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}
    for speaker in speakers:
        size = len(groups[speaker])
        if counts["test"] < target_test:
            choice = "test"
        elif counts["val"] < target_val:
            choice = "val"
        else:
            choice = "train"
        for example in groups[speaker]:
            example.split = choice
            buckets[choice].append(example)
        counts[choice] += size
    return buckets


def deduplicate_mustard_family(
    examples: list[UtteranceExample],
) -> tuple[list[UtteranceExample], list[str]]:
    """Drop MUStARD rows whose utterance text also appears in MUStARD++.

    MUStARD++ extends and re-annotates MUStARD. Keeping both copies of the
    same line would let a test utterance appear in training. MUStARD++ wins
    because it carries the later emotion and sarcasm-type labels.
    """

    later = {
        normalize_for_dedup(example.text)
        for example in examples
        if example.dataset == "mustard++" and example.text.strip()
    }
    kept: list[UtteranceExample] = []
    notes: list[str] = []
    for example in examples:
        key = normalize_for_dedup(example.text)
        if example.dataset == "mustard" and key in later:
            notes.append(example.example_id)
            continue
        kept.append(example)
    return kept, notes


def context_leakage(train: list[UtteranceExample], test: list[UtteranceExample]) -> list[dict]:
    """Test utterances whose text appears inside a training context.

    Speaker-independent splits do not remove this conversational overlap.
    The list is evidence for the paper's limitations section, not a filter
    applied silently.
    """

    train_context = set()
    for example in train:
        for turn in example.context:
            train_context.add(normalize_for_dedup(turn))
    leaks = []
    for example in test:
        key = normalize_for_dedup(example.text)
        if key and key in train_context:
            leaks.append(
                {
                    "example_id": example.example_id,
                    "dataset": example.dataset,
                    "speaker": example.speaker,
                }
            )
    return leaks
