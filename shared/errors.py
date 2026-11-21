"""Exceptions raised by more than one module."""


class SarcasToneError(Exception):
    """Base error for expected, user-facing failures."""


# Keep the original misspelling importable for existing callers.
ArcasToneError = SarcasToneError


class DatasetSchemaError(SarcasToneError):
    """A dataset file does not match the schema this loader understands."""


class CheckpointNotFoundError(SarcasToneError, FileNotFoundError):
    """A required trained checkpoint is not on disk."""


class UntrainedModelError(SarcasToneError):
    """Inference was asked to treat an untrained head as a sarcasm model."""


class AudioLoadError(SarcasToneError):
    """Audio could not be decoded, was empty, or failed a duration check."""
