"""Classification metrics.

A metric that cannot be computed (for example ROC-AUC with one class) is
returned as ``None``. It is never filled in with a placeholder score.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _as_1d_int(values: np.ndarray | list) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1-D label vector, got shape {array.shape}")
    return array.astype(int)


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """Expected calibration error of the maximum predicted probability."""

    labels = _as_1d_int(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != labels.shape[0]:
        raise ValueError("probabilities must have shape [N, C] aligned with labels")
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(labels)
    error = 0.0
    for index in range(n_bins):
        if index == 0:
            selected = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            selected = (confidence > edges[index]) & (confidence <= edges[index + 1])
        count = int(selected.sum())
        if count == 0:
            continue
        bin_accuracy = float(correct[selected].mean())
        bin_confidence = float(confidence[selected].mean())
        error += (count / total) * abs(bin_accuracy - bin_confidence)
    return float(error)


def compute_classification_metrics(
    labels: np.ndarray | list,
    probabilities: np.ndarray,
    *,
    positive_index: int = 1,
) -> dict:
    """Binary or multiclass metrics. ``probabilities`` is ``[N, C]``."""

    y_true = _as_1d_int(labels)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    if y_prob.shape[0] != y_true.shape[0]:
        raise ValueError("Label and probability counts do not match")
    y_pred = y_prob.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    per_precision, per_recall, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(y_prob.shape[1])), zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(y_prob.shape[1])))
    roc_auc = None
    pr_auc = None
    if y_prob.shape[1] == 2 and len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, y_prob[:, positive_index]))
        pr_auc = float(average_precision_score(y_true, y_prob[:, positive_index]))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "macro_f1": float(f1),
        "precision_weighted": float(weighted_p),
        "recall_weighted": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "expected_calibration_error": expected_calibration_error(y_prob, y_true),
        "support": [int(value) for value in support],
        "per_class_precision": [float(value) for value in per_precision],
        "per_class_recall": [float(value) for value in per_recall],
        "per_class_f1": [float(value) for value in per_f1],
        "confusion_matrix": matrix.astype(int).tolist(),
        "count": int(y_true.shape[0]),
    }


def bootstrap_macro_f1_interval(
    labels: np.ndarray | list,
    probabilities: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Percentile interval of macro-F1. ``n_boot`` resamples of the given rows."""

    y_true = _as_1d_int(labels)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    generator = np.random.default_rng(seed)
    scores = np.empty(n_boot, dtype=np.float64)
    count = y_true.shape[0]
    for draw in range(n_boot):
        chosen = generator.integers(0, count, size=count)
        scores[draw] = compute_classification_metrics(y_true[chosen], y_prob[chosen])["macro_f1"]
    low = float(np.quantile(scores, alpha / 2))
    high = float(np.quantile(scores, 1 - alpha / 2))
    return {
        "metric": "macro_f1",
        "n_boot": n_boot,
        "seed": seed,
        "mean": float(scores.mean()),
        "low": low,
        "high": high,
    }


def per_group_macro_f1(
    labels: np.ndarray | list, predictions: np.ndarray | list, groups: list[str]
) -> dict[str, dict]:
    """Macro-F1 inside each group that contains at least one example."""

    y_true = _as_1d_int(labels)
    y_pred = _as_1d_int(predictions)
    if len(groups) != len(y_true):
        raise ValueError("groups must align with labels")
    unique = sorted(set(groups))
    report: dict[str, dict] = {}
    for name in unique:
        chosen = [index for index, group in enumerate(groups) if group == name]
        group_true = y_true[chosen]
        group_pred = y_pred[chosen]
        # One-hot probabilities from hard predictions so the shared metric
        # function can be reused. This scores the predictions, not confidence.
        probabilities = np.zeros((len(chosen), 2), dtype=np.float64)
        probabilities[np.arange(len(chosen)), np.clip(group_pred, 0, 1)] = 1.0
        metrics = compute_classification_metrics(group_true, probabilities)
        report[name] = {"count": len(chosen), "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"]}
    return report
