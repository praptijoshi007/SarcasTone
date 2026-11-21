"""Metrics for one dataset and one task.

Files written here are computed from predictions. This module does not
contain a stored score for any corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from member1_text import require_python_311
from member1_text.canonicalize import CanonicalExample
from member1_text.config import PACKAGE_ROOT, checkpoint_name, model_slug


def classification_metrics(
    labels: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
    probabilities: list[float] | np.ndarray | None = None,
    *,
    binary: bool = False,
) -> dict[str, Any]:
    label_array = np.asarray(labels)
    prediction_array = np.asarray(predictions)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        label_array,
        prediction_array,
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        label_array,
        prediction_array,
        average="weighted",
        zero_division=0,
    )
    report = classification_report(label_array, prediction_array, zero_division=0, output_dict=True)
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(label_array, prediction_array)),
        "precision": float(precision),
        "recall": float(recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": confusion_matrix(label_array, prediction_array).tolist(),
        "classification_report": report,
        "classification_report_text": classification_report(
            label_array, prediction_array, zero_division=0
        ),
    }
    if binary:
        result["roc_auc"] = None
        result["pr_auc"] = None
        if probabilities is not None and len(set(label_array.tolist())) > 1:
            scores = np.asarray(probabilities, dtype=float)
            result["roc_auc"] = float(roc_auc_score(label_array, scores))
            result["pr_auc"] = float(average_precision_score(label_array, scores))
        else:
            result["ranking_note"] = (
                "ROC-AUC and PR-AUC were not computed because the reference labels "
                "contain a single class or no positive-class probabilities were passed."
            )
    return result


def regression_metrics(labels: list[float] | np.ndarray, predictions: list[float] | np.ndarray) -> dict[str, Any]:
    gold = np.asarray(labels, dtype=float)
    pred = np.asarray(predictions, dtype=float)
    return {
        "mse": float(mean_squared_error(gold, pred)),
        "mae": float(mean_absolute_error(gold, pred)),
    }


def binary_curves(
    labels: list[int] | np.ndarray,
    probabilities: list[float] | np.ndarray,
) -> tuple[dict[str, list[float]], dict[str, list[float]]] | None:
    label_array = np.asarray(labels)
    if len(set(label_array.tolist())) < 2:
        return None
    scores = np.asarray(probabilities, dtype=float)
    fpr, tpr, _ = roc_curve(label_array, scores)
    precision, recall, _ = precision_recall_curve(label_array, scores)
    return (
        {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        {"precision": precision.tolist(), "recall": recall.tolist()},
    )


def error_list(
    examples: list[CanonicalExample],
    predictions: list[int],
    probabilities: list[float] | None,
    *,
    binary: bool,
) -> list[dict[str, Any]]:
    """Mismatches. SemEval rows store ids only and never the tweet text."""
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        predicted = int(predictions[index])
        gold = int(example.label)
        if predicted == gold:
            continue
        probability = None if probabilities is None else float(probabilities[index])
        row: dict[str, Any] = {
            "id": example.example_id,
            "gold_label": gold,
            "predicted_label": predicted,
            "probability": probability,
        }
        if binary:
            if predicted == 1 and gold == 0:
                row["kind"] = "false_positive"
            elif predicted == 0 and gold == 1:
                row["kind"] = "false_negative"
        if example.dataset != "semeval2018":
            if example.extra.get("show"):
                row["show"] = example.extra["show"]
            if example.extra.get("subreddit"):
                row["subreddit"] = example.extra["subreddit"]
        rows.append(row)
    return rows


def word_error_rate(reference: str, hypothesis: str) -> float:
    import jiwer

    if not reference.strip() or not hypothesis.strip():
        raise ValueError("WER needs both a gold string and an ASR string.")
    return float(jiwer.wer(reference, hypothesis))


def text_source_rows(
    gold_metrics: dict[str, Any],
    asr_metrics: dict[str, Any] | None = None,
    wer: float | None = None,
) -> list[dict[str, Any]]:
    rows = [{"text_source": "gold", **gold_metrics}]
    if asr_metrics is not None:
        asr_row = {"text_source": "asr", **asr_metrics}
        if wer is not None:
            asr_row["wer"] = wer
        rows.append(asr_row)
    return rows


def eval_directory(dataset: str, task: str, model_name: str = "roberta-base") -> Path:
    """RoBERTa reports use dataset__task. Other models get their own directory."""
    slug = model_slug(model_name)
    name = f"{dataset}__{task}" if slug == "roberta-base" else f"{dataset}__{task}__{slug}"
    directory = PACKAGE_ROOT / "outputs" / "eval" / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_classification_artifacts(
    directory: Path,
    stem: str,
    metrics: dict[str, Any],
    *,
    history: list[dict[str, Any]] | None = None,
    labels: list[int] | None = None,
    probabilities: list[float] | None = None,
    title: str,
) -> None:
    from member1_text.plots import plot_confusion_matrix, plot_loss, plot_precision_recall, plot_roc

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        directory / f"{stem}_confusion.png",
        title=title,
    )
    if history:
        plot_loss(history, directory / f"{stem}_loss.png", title=f"{title} loss")
    if labels is not None and probabilities is not None:
        curves = binary_curves(labels, probabilities)
        if curves is not None:
            roc, pr = curves
            plot_roc(roc["fpr"], roc["tpr"], directory / f"{stem}_roc.png", title=f"{title} ROC")
            plot_precision_recall(
                pr["recall"],
                pr["precision"],
                directory / f"{stem}_pr.png",
                title=f"{title} precision-recall",
            )


def deployment_checkpoint(dataset: str, task: str, model_name: str) -> Path:
    slug = model_slug(model_name)
    return PACKAGE_ROOT / "outputs" / "checkpoints" / checkpoint_name(dataset, task, slug)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved Member 1 checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="speaker_grouped")
    parser.add_argument("--device", default=None)
    parser.add_argument("--text-source", choices=("gold", "asr"), default="gold")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    require_python_311()
    args = parse_args(argv)
    from member1_text.train import evaluate_checkpoint

    evaluate_checkpoint(
        checkpoint=Path(args.checkpoint),
        dataset=args.dataset,
        task=args.task,
        split_name=args.split,
        device_name=args.device,
        text_source=args.text_source,
    )


if __name__ == "__main__":
    main()
