"""Matplotlib figures for one dataset and task. No scores are embedded here."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(matrix: list[list[int]], path: Path, *, title: str) -> None:
    array = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(array, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Gold")
    axis.set_title(title)
    for row in range(array.shape[0]):
        for column in range(array.shape[1]):
            axis.text(column, row, str(int(array[row, column])), ha="center", va="center")
    figure.tight_layout()
    _save(figure, path)


def plot_roc(fpr: list[float], tpr: list[float], path: Path, *, title: str) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(fpr, tpr)
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title(title)
    figure.tight_layout()
    _save(figure, path)


def plot_precision_recall(recall: list[float], precision: list[float], path: Path, *, title: str) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(recall, precision)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(title)
    figure.tight_layout()
    _save(figure, path)


def plot_loss(history: list[dict], path: Path, *, title: str) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row.get("train_loss") for row in history]
    valid_loss = [row.get("val_loss") for row in history]
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(epochs, train_loss, label="train")
    if any(value is not None for value in valid_loss):
        axis.plot(epochs, valid_loss, label="validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    _save(figure, path)


def _save(figure: plt.Figure, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
