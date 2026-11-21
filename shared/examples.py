"""Records that move between dataset loaders, manifests, and trainers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class UtteranceExample:
    """One trainable or evaluable utterance.

    ``label`` is the sarcasm/irony label and is ``None`` for emotion-only
    auxiliary audio. ``emotion`` is set only for emotion datasets. An emotion
    example is never given a sarcasm label by the loaders.
    """

    example_id: str
    dataset: str
    text: str
    speaker: str
    task: str
    context: list[str] = field(default_factory=list)
    context_speakers: list[str] = field(default_factory=list)
    label: int | None = None
    audio_path: str | None = None
    show: str | None = None
    sarcasm_type: str | None = None
    implicit_emotion: str | None = None
    explicit_emotion: str | None = None
    valence: float | None = None
    arousal: float | None = None
    emotion: str | None = None
    actor_id: str | None = None
    split: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LoadReport:
    """What a loader actually did, including repairs and dropped rows."""

    dataset: str
    source_path: str
    kept: int
    dropped: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
