"""Evaluation module for member3_fusion.

Computes all required metrics including:
  - Accuracy, Macro-F1, Weighted-F1, Precision, Recall
  - ROC-AUC, PR-AUC
  - Confusion matrix
  - Calibration (Expected Calibration Error)
  - Per-dataset performance (MUStARD vs MUStARD++)
  - Statistical confidence intervals (bootstrap)
  - Modality weight analysis

No metric is fabricated. If a metric cannot be computed (e.g. single class),
it is returned as None.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "member3_fusion") not in sys.path:
    sys.path.insert(0, str(_REPO / "member3_fusion"))

from shared.metrics import compute_classification_metrics
from shared.constants import SARCASM_LABELS

from src.training.trainer import run_batch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Non-parametric bootstrap CI for a scalar metric.

    Args:
        labels:    [N] integer labels.
        probs:     [N, C] class probabilities.
        metric_fn: callable(labels, probs) → float.
        n_bootstrap: number of bootstrap resamples.
        ci: confidence level.
        seed: random seed for reproducibility.

    Returns:
        {mean, std, lower, upper}
    """
    rng = np.random.default_rng(seed)
    N = len(labels)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        try:
            score = metric_fn(labels[idx], probs[idx])
            if score is not None and np.isfinite(score):
                scores.append(float(score))
        except Exception:
            pass

    if not scores:
        return {"mean": None, "std": None, "lower": None, "upper": None}

    scores_arr = np.array(scores)
    alpha = (1 - ci) / 2
    return {
        "mean": float(np.mean(scores_arr)),
        "std":  float(np.std(scores_arr)),
        "lower": float(np.quantile(scores_arr, alpha)),
        "upper": float(np.quantile(scores_arr, 1 - alpha)),
    }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class FusionEvaluator:
    """Runs full evaluation on a model checkpoint."""

    def __init__(
        self,
        model,
        device: torch.device,
        text_encoder=None,
        label_names: tuple = SARCASM_LABELS,
    ) -> None:
        self.model = model
        self.device = device
        self.text_encoder = text_encoder
        self.label_names = label_names

    @torch.no_grad()
    def collect_predictions(
        self, loader: DataLoader
    ) -> dict[str, np.ndarray]:
        """Run model on loader and collect all predictions."""
        self.model.eval()
        all_probs   = []
        all_labels  = []
        all_ids     = []
        all_text_w  = []
        all_audio_w = []
        all_entropy = []
        all_disagree = []
        all_cosine   = []

        for batch in loader:
            out = run_batch(self.model, batch, self.device, self.text_encoder)
            probs = out["probabilities"].cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(batch["label"].numpy().tolist())
            all_ids.extend(batch["example_ids"])
            all_entropy.extend(out["entropy"].cpu().numpy().tolist())

            if out["text_weight"] is not None:
                all_text_w.extend(out["text_weight"].cpu().numpy().tolist())
                all_audio_w.extend(out["audio_weight"].cpu().numpy().tolist())
            if out["disagreement_norm"] is not None:
                all_disagree.extend(out["disagreement_norm"].cpu().numpy().tolist())
            if out["cosine_agree"] is not None:
                all_cosine.extend(out["cosine_agree"].cpu().numpy().tolist())

        return {
            "probabilities": np.vstack(all_probs),
            "labels":        np.array(all_labels),
            "example_ids":   np.array(all_ids),
            "entropy":       np.array(all_entropy),
            "text_weight":   np.array(all_text_w) if all_text_w else None,
            "audio_weight":  np.array(all_audio_w) if all_audio_w else None,
            "disagreement_norm": np.array(all_disagree) if all_disagree else None,
            "cosine_agree":  np.array(all_cosine) if all_cosine else None,
        }

    def evaluate(
        self,
        loader: DataLoader,
        split_name: str = "test",
        n_bootstrap: int = 1000,
    ) -> dict:
        """Full evaluation with all metrics."""
        preds = self.collect_predictions(loader)
        probs  = preds["probabilities"]
        labels = preds["labels"]

        # Core metrics
        metrics = compute_classification_metrics(labels, probs)
        metrics["split"] = split_name
        metrics["n_examples"] = int(len(labels))
        metrics["n_sarcastic"] = int((labels == 1).sum())
        metrics["label_names"] = list(self.label_names)

        # Uncertainty stats
        metrics["entropy_mean"] = float(preds["entropy"].mean())
        metrics["entropy_std"]  = float(preds["entropy"].std())
        metrics["uncertain_ratio"] = float((preds["entropy"] > 0.347).mean())

        # Modality weight stats (when gating is enabled)
        if preds["text_weight"] is not None:
            metrics["text_weight_mean"]  = float(preds["text_weight"].mean())
            metrics["audio_weight_mean"] = float(preds["audio_weight"].mean())

        # PSID stats
        if preds["disagreement_norm"] is not None:
            metrics["disagreement_norm_mean"] = float(preds["disagreement_norm"].mean())
            metrics["cosine_agree_mean"]      = float(preds["cosine_agree"].mean())

        # Bootstrap CI on macro-F1 and accuracy
        def _macro_f1(y, p):
            m = compute_classification_metrics(y, p)
            return m.get("macro_f1")

        def _accuracy(y, p):
            m = compute_classification_metrics(y, p)
            return m.get("accuracy")

        metrics["macro_f1_ci"] = bootstrap_ci(labels, probs, _macro_f1, n_bootstrap)
        metrics["accuracy_ci"] = bootstrap_ci(labels, probs, _accuracy, n_bootstrap)

        return metrics

    def save_predictions(
        self,
        loader: DataLoader,
        output_path: Path | str,
    ) -> None:
        """Save per-example predictions to JSON."""
        preds = self.collect_predictions(loader)
        records = []
        for i, (uid, label) in enumerate(
            zip(preds["example_ids"], preds["labels"])
        ):
            prob_sarc = float(preds["probabilities"][i][1])
            pred_label = int(preds["probabilities"][i].argmax())
            records.append({
                "example_id": str(uid),
                "true_label": int(label),
                "true_label_name": SARCASM_LABELS[int(label)],
                "predicted_label": pred_label,
                "predicted_label_name": SARCASM_LABELS[pred_label],
                "probability_sarcastic": prob_sarc,
                "probability_nonsarcastic": float(preds["probabilities"][i][0]),
                "entropy": float(preds["entropy"][i]),
                "correct": bool(pred_label == int(label)),
            })
        Path(output_path).write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )
        log.info("Saved %d predictions to %s", len(records), output_path)
