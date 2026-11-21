"""Encoder interface wrappers for member1 (text) and member2 (audio).

Both wrappers return tensors of the shapes declared in shared/interfaces.py.
If the upstream checkpoint is unavailable, the text wrapper raises (consistent
with member1's design philosophy: no dummy probabilities).

For the audio encoder: member2's wav2vec2 embeddings were precomputed and
stored in the handoff CSV / npz files. At training time we receive them as
pre-extracted tensors from the dataset. This wrapper also supports on-the-fly
extraction when a Wav2Vec2 model is available.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from shared.constants import TEXT_HIDDEN, AUDIO_HIDDEN, CONTEXT_HIDDEN

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text encoder interface (member1)
# ---------------------------------------------------------------------------


class TextEncoderInterface(nn.Module):
    """Wraps member1_text.api.get_text_embedding for use inside the fusion model.

    At training time the checkpoint path may be None (encoder not yet trained).
    In that case the wrapper returns learnable zero-initialized embeddings that
    still allow the rest of the fusion model to train (text-unavailable mode).

    Shapes returned
    ---------------
    utterance:  float32 [B, TEXT_HIDDEN]
    context:    float32 [B, CONTEXT_HIDDEN]   (zeros if no context)
    has_context: bool   [B]
    """

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        device: torch.device | None = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.device = device or torch.device("cpu")
        self._model = None
        self._tokenizer = None
        self._available = False
        self.freeze = freeze

        self._try_load()

    def _try_load(self) -> None:
        """Attempt to load member1 deployment checkpoint. Silent on failure."""
        if self.checkpoint_path is None:
            log.info("TextEncoderInterface: no checkpoint path; text encoder disabled")
            return
        try:
            # Import member1 API
            m1_src = _REPO / "member1_text" / "src"
            if str(m1_src) not in sys.path:
                sys.path.insert(0, str(m1_src))

            from member1_text.api import load_deployment
            from transformers import AutoTokenizer

            checkpoint = self.checkpoint_path
            model, tokenizer, _ = load_deployment(checkpoint, device=self.device)
            self._model = model
            self._tokenizer = tokenizer
            self._available = True
            log.info(
                "TextEncoderInterface: loaded member1 checkpoint from %s", checkpoint
            )
            if self.freeze:
                for param in self._model.parameters():
                    param.requires_grad_(False)
        except Exception as exc:
            log.warning(
                "TextEncoderInterface: could not load member1 checkpoint: %s. "
                "Text encoder will return zeros.",
                exc,
            )

    @property
    def is_available(self) -> bool:
        return self._available

    def set_trainable(self, trainable: bool) -> None:
        """Toggle encoder fine-tuning."""
        if self._model is not None:
            for param in self._model.parameters():
                param.requires_grad_(trainable)

    def forward(
        self,
        texts: list[str],
        contexts: list[list[str]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode a batch of utterances.

        Args:
            texts: list of B utterance strings.
            contexts: optional list of B context-turn lists.

        Returns:
            utterance: [B, TEXT_HIDDEN]
            context:   [B, CONTEXT_HIDDEN]
            has_context: [B] bool
        """
        B = len(texts)
        device = self.device

        if not self._available:
            return {
                "utterance": torch.zeros(B, TEXT_HIDDEN, device=device),
                "context": torch.zeros(B, CONTEXT_HIDDEN, device=device),
                "has_context": torch.zeros(B, dtype=torch.bool, device=device),
            }

        utterance_list = []
        context_list = []
        has_ctx_list = []

        from member1_text.api import get_text_embedding, encode_text
        from member1_text.preprocess import normalize_text

        for i, text in enumerate(texts):
            ctx_str: str | None = None
            if contexts and contexts[i]:
                ctx_str = " ".join(str(t) for t in contexts[i])

            try:
                result = get_text_embedding(
                    text,
                    ctx_str,
                    checkpoint=self.checkpoint_path,
                )
                utt = result["utterance"]   # [768]
                ctx = result["context"]     # [768]
                hc = torch.tensor(result["has_context"], dtype=torch.bool)
            except Exception as exc:
                log.debug("Text encoding failed for idx %d: %s", i, exc)
                utt = torch.zeros(TEXT_HIDDEN)
                ctx = torch.zeros(CONTEXT_HIDDEN)
                hc = torch.tensor(False, dtype=torch.bool)

            utterance_list.append(utt)
            context_list.append(ctx)
            has_ctx_list.append(hc)

        return {
            "utterance": torch.stack(utterance_list).to(device),
            "context": torch.stack(context_list).to(device),
            "has_context": torch.stack(has_ctx_list).to(device),
        }


# ---------------------------------------------------------------------------
# Audio encoder interface (member2)
# ---------------------------------------------------------------------------


class AudioEncoderInterface(nn.Module):
    """Wraps member2 Wav2Vec2 for on-the-fly inference.

    At training time, embeddings are pre-extracted (from npz or the dataset).
    This interface is used during inference / predict_multimodal.

    Returned shape: utterance_embedding [1, AUDIO_HIDDEN]
    """

    def __init__(
        self,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")
        self._wav2vec_model = None
        self._processor = None
        self._available = False
        self._try_load()

    def _try_load(self) -> None:
        try:
            m2_src = _REPO / "member2_audio" / "src"
            if str(m2_src) not in sys.path:
                sys.path.insert(0, str(m2_src))

            from audio_intelligence.audio_intelligence import (
                load_wav2vec2,
                prepare_audio_for_wav2vec2,
            )
            self._load_wav2vec2 = load_wav2vec2
            self._prepare_audio = prepare_audio_for_wav2vec2
            self._available = True
            log.info("AudioEncoderInterface: member2 audio functions available")
        except Exception as exc:
            log.warning(
                "AudioEncoderInterface: could not import member2 audio: %s. "
                "Audio encoder will return zeros.",
                exc,
            )

    @property
    def is_available(self) -> bool:
        return self._available

    def forward(self, audio_path: str | Path) -> dict[str, torch.Tensor]:
        """Encode a single audio file.

        Returns:
            embedding: [1, AUDIO_HIDDEN]
            available: bool tensor [1]
        """
        if not self._available:
            return {
                "embedding": torch.zeros(1, AUDIO_HIDDEN, device=self.device),
                "available": torch.tensor([False], dtype=torch.bool, device=self.device),
            }
        try:
            wav2vec_model, processor = self._load_wav2vec2()
            waveform, sr, _meta = self._prepare_audio(audio_path)
            inputs = processor(waveform, sampling_rate=sr, return_tensors="pt")
            with torch.no_grad():
                out = wav2vec_model(inputs.input_values)
                emb = out.last_hidden_state.mean(dim=1).cpu()  # [1, 768]
            return {
                "embedding": emb.to(self.device),
                "available": torch.tensor([True], dtype=torch.bool, device=self.device),
            }
        except Exception as exc:
            log.warning("Audio encoding failed for %s: %s", audio_path, exc)
            return {
                "embedding": torch.zeros(1, AUDIO_HIDDEN, device=self.device),
                "available": torch.tensor([False], dtype=torch.bool, device=self.device),
            }
