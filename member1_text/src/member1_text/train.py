"""Fine-tune the span-pooling transformer and evaluate a saved checkpoint.

Cross-validation scores stay in ``cv_validation.json`` and are not written
as test scores. The MUStARD deployment checkpoint is a full-data refit.
A missing corpus raises ``DatasetNotAvailableError`` and writes no metrics.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from member1_text import require_python_311
from member1_text.api import resolve_device
from member1_text.canonicalize import CanonicalExample, DatasetNotAvailableError
from member1_text.config import (
    PACKAGE_ROOT,
    checkpoint_name,
    huggingface_model_id,
    load_config,
    model_slug,
)
from member1_text.data.mustard import load_mustard, write_text_provenance
from member1_text.data.mustard_pp import load_mustard_pp
from member1_text.data.sarc import load_sarc
from member1_text.data.semeval2018 import default_train_path, load_official_test, load_semeval
from member1_text.data.splits import (
    drop_mustard_overlap,
    median_epoch,
    speaker_grouped_split,
    stratified_holdout,
    stratified_kfold,
    subset_by_ids,
)
from member1_text.evaluate import (
    classification_metrics,
    error_list,
    eval_directory,
    regression_metrics,
    write_classification_artifacts,
    write_json,
)
from member1_text.model import build_model, load_span_model
from member1_text.preprocess import join_context, preprocess_example
from member1_text.tokenize_encode import assert_tokenizer_matches_model, encode_pair

CLOSED_LABELS = {
    ("mustard", "sarcasm"): {"0": 0, "1": 1},
    ("mustard_pp", "sarcasm"): {"0": 0, "1": 1},
    ("sarc", "sarcasm"): {"0": 0, "1": 1},
    ("semeval2018", "taskA"): {"0": 0, "1": 1},
    ("semeval2018", "taskB"): {"0": 0, "1": 1, "2": 2, "3": 3},
}


def normalize_dataset(name: str) -> str:
    key = name.strip().lower().replace("+", "p").replace("-", "")
    aliases = {
        "mustard": "mustard",
        "mustardpp": "mustard_pp",
        "mustard_pp": "mustard_pp",
        "sarc": "sarc",
        "semeval": "semeval2018",
        "semeval2018": "semeval2018",
        "semeval2018task3": "semeval2018",
    }
    if key not in aliases:
        raise ValueError(f"Unknown dataset {name!r}.")
    return aliases[key]


def normalize_task(dataset: str, task: str) -> str:
    key = task.strip().lower()
    if dataset == "mustard":
        if key != "sarcasm":
            raise ValueError("MUStARD training task must be sarcasm.")
        return "sarcasm"
    if dataset == "mustard_pp":
        aliases = {
            "sarcasm": "sarcasm",
            "explicit_emotion": "explicit_emotion",
            "emotion": "explicit_emotion",
            "valence": "valence",
            "arousal": "arousal",
        }
        if key not in aliases:
            raise ValueError(
                "MUStARD++ task must be sarcasm, explicit_emotion, valence, or arousal."
            )
        return aliases[key]
    if dataset == "sarc":
        if key != "sarcasm":
            raise ValueError("SARC training task must be sarcasm.")
        return "sarcasm"
    if key in {"taska", "a", "irony"}:
        return "taskA"
    if key in {"taskb", "b", "irony_type"}:
        return "taskB"
    raise ValueError("SemEval task must be taskA or taskB.")


def task_type_for(dataset: str, task: str) -> str:
    if dataset == "mustard_pp" and task in {"valence", "arousal"}:
        return "regression"
    return "classification"


def default_split(dataset: str, config: dict[str, Any]) -> str:
    if dataset == "mustard":
        return str(config["training"]["primary"]["split"])
    if dataset == "mustard_pp":
        return "stratified_5fold"
    if dataset == "sarc":
        return "official_balanced"
    return "train_holdout"


def load_examples(dataset: str, task: str) -> list[CanonicalExample]:
    """Load the local corpus for export. A missing file raises and invents no rows."""
    dataset = normalize_dataset(dataset)
    task = normalize_task(dataset, task)
    if dataset == "mustard":
        return load_mustard()
    if dataset == "mustard_pp":
        return load_mustard_pp(task=task)
    if dataset == "sarc":
        return load_sarc(split="train") + load_sarc(split="test")
    return load_semeval(default_train_path(task), task=task, split="train")


def apply_text_source(examples: list[CanonicalExample], text_source: str) -> list[CanonicalExample]:
    if text_source == "gold":
        return list(examples)
    if text_source != "asr":
        raise ValueError("text_source must be gold or asr.")
    converted: list[CanonicalExample] = []
    for example in examples:
        asr_text = str(example.extra.get("asr_text") or "").strip()
        if not asr_text:
            raise RuntimeError(
                f"Example {example.example_id} has no asr_text. "
                "Gold text was not substituted, and no metric file was written."
            )
        extra = dict(example.extra)
        extra["gold_text"] = example.text
        converted.append(
            CanonicalExample(
                example_id=example.example_id,
                dataset=example.dataset,
                split=example.split,
                text=asr_text,
                context_turns=list(example.context_turns),
                text_source="asr",
                label=example.label,
                label_meaning=example.label_meaning,
                extra=extra,
                overlap_key=example.overlap_key,
            )
        )
    return converted


def maybe_drop_overlap(
    dataset: str,
    examples: list[CanonicalExample],
    enabled: bool,
) -> tuple[list[CanonicalExample], int]:
    if not enabled:
        return examples, 0
    if dataset == "mustard":
        raise ValueError("Overlap removal applies to an auxiliary corpus, not to MUStARD itself.")
    kept, removed = drop_mustard_overlap(load_mustard(), examples)
    print(f"Removed {removed} {dataset} examples that overlap MUStARD.")
    return kept, removed


class ExampleDataset(Dataset):
    """Tokenize canonical examples with the same preprocessor the API uses."""

    def __init__(
        self,
        examples: list[CanonicalExample],
        tokenizer: Any,
        max_length: int,
        model_name: str,
        label_map: dict[str, int] | None = None,
        task_type: str = "classification",
        expand_contractions: bool = False,
    ) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_name = model_name
        self.label_map = dict(label_map or {})
        self.task_type = task_type
        self.expand_contractions = expand_contractions

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = preprocess_example(
            self.examples[index],
            expand_contractions=self.expand_contractions,
        )
        encoded = encode_pair(
            self.tokenizer,
            example.text,
            join_context(example.context_turns),
            max_length=self.max_length,
            model_name=self.model_name,
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "context_mask": encoded["context_mask"].squeeze(0),
            "target_mask": encoded["target_mask"].squeeze(0),
            "label": encode_label(example, self.label_map, self.task_type),
            "example_id": example.example_id,
        }


def collate_examples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [item["label"] for item in batch]
    label_dtype = torch.float if isinstance(labels[0], float) else torch.long
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "context_mask": torch.stack([item["context_mask"] for item in batch]),
        "target_mask": torch.stack([item["target_mask"] for item in batch]),
        "labels": torch.tensor(labels, dtype=label_dtype),
        "example_id": [item["example_id"] for item in batch],
    }


def encode_label(
    example: CanonicalExample,
    label_map: dict[str, int],
    task_type: str,
) -> int | float:
    if task_type == "regression":
        return float(example.label)
    if isinstance(example.label, str):
        if example.label not in label_map:
            raise KeyError(
                f"Label {example.label!r} on {example.example_id} is not in the training label map."
            )
        return int(label_map[example.label])
    return int(example.label)


def label_map_for(
    dataset: str,
    task: str,
    train_examples: list[CanonicalExample],
    valid_examples: list[CanonicalExample] | None = None,
) -> dict[str, int]:
    """Closed tasks use their fixed label set. Open string tasks sort the observed names.

    Class weights are still computed from the training split only. A string that
    appears only in validation is added so the head has an index for it; its
    training weight is zero because its training count is zero.
    """
    closed = CLOSED_LABELS.get((dataset, task))
    if closed is not None:
        return dict(closed)
    names = {str(example.label) for example in train_examples}
    for example in valid_examples or []:
        if isinstance(example.label, str):
            names.add(example.label)
    if not names:
        raise ValueError("Cannot build a label map from an empty training split.")
    return {name: index for index, name in enumerate(sorted(names))}


def inverse_frequency_weights(labels: list[int], num_labels: int) -> list[float]:
    """Inverse-frequency weights from the training split. Missing classes get weight 0."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_labels).astype(np.float64)
    present = counts > 0
    if not present.any():
        raise ValueError("Training split has no labels to weight.")
    weights = np.zeros(num_labels, dtype=np.float64)
    weights[present] = counts.sum() / (int(present.sum()) * counts[present])
    return [float(value) for value in weights]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def training_device(requested: str | None) -> torch.device:
    if requested and requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is available. No checkpoint was written.")
    return resolve_device(requested)


def batch_size_for(device: torch.device, config: dict[str, Any]) -> int:
    if device.type == "cuda":
        return int(config["training"]["batch_size_gpu"])
    return int(config["training"]["batch_size_cpu"])


def make_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_for(device: torch.device, dtype: torch.dtype, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def checkpoint_root(output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    return PACKAGE_ROOT / "outputs" / "checkpoints"


def run_dir_for(dataset: str, task: str, model_name: str, split: str, output_dir: str | None) -> Path:
    name = checkpoint_name(dataset, task, model_slug(model_name))
    root = checkpoint_root(output_dir)
    if split == "deployment_refit":
        return root / name
    return root / f"{name}__{split}"


def report_dir(dataset: str, task: str, model_name: str, output_dir: str | None) -> Path:
    if output_dir:
        slug = model_slug(model_name)
        folder = f"{dataset}__{task}" if slug == "roberta-base" else f"{dataset}__{task}__{slug}"
        directory = Path(output_dir) / "eval" / folder
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    return eval_directory(dataset, task, model_name)


def refuse_existing_run(path: Path) -> None:
    if not path.exists():
        return
    has_weights = any(path.glob("checkpoint-*/trainer_state.pt")) or (path / "trainer_state.pt").is_file()
    has_result = (path / "fold_result.json").is_file() or any(path.glob("fold-*/fold_result.json"))
    if has_weights or has_result:
        raise RuntimeError(
            f"Run directory already exists: {path}. "
            "Resume with --resume <checkpoint-epoch> or remove the directory before a new run. "
            "No new metric file was written."
        )


def latest_checkpoint(directory: Path) -> Path | None:
    found: list[tuple[int, Path]] = []
    if not directory.is_dir():
        return None
    for child in directory.glob("checkpoint-*"):
        if (child / "trainer_state.pt").is_file() and (child / "config.json").is_file():
            found.append((int(child.name.rsplit("-", 1)[-1]), child))
    if not found:
        return None
    return max(found)[1]


def load_trainer_state(path: Path) -> dict[str, Any]:
    state_path = path / "trainer_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"Checkpoint {path} has no trainer_state.pt.")
    return torch.load(state_path, map_location="cpu", weights_only=False)


def save_epoch_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    directory: Path,
    state: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    torch.save(state, directory / "trainer_state.pt")


def optimizer_for(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(".bias") or "LayerNorm.weight" in name or "layer_norm.weight" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    weight_decay = float(config["training"]["weight_decay"])
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(config["training"]["learning_rate"]),
    )


def make_loader(
    examples: list[CanonicalExample],
    tokenizer: Any,
    model_name: str,
    max_length: int,
    label_map: dict[str, int],
    task_type: str,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    expand_contractions: bool,
) -> DataLoader:
    dataset = ExampleDataset(
        examples,
        tokenizer,
        max_length,
        model_name,
        label_map=label_map,
        task_type=task_type,
        expand_contractions=expand_contractions,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=collate_examples,
        num_workers=0,
    )


def forward_batch(model: torch.nn.Module, batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        context_mask=batch["context_mask"].to(device),
        target_mask=batch["target_mask"].to(device),
    )


def batch_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    task_type: str,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    if task_type == "regression":
        return torch.nn.functional.mse_loss(logits.view(-1), labels.float().view(-1))
    return torch.nn.functional.cross_entropy(logits, labels.long(), weight=class_weights)


def score_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    task_type: str,
    class_weights: torch.Tensor | None,
    use_amp: bool,
    amp_dtype: torch.dtype,
    binary: bool,
    examples: list[CanonicalExample],
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    counts: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    regression_predictions: list[float] = []
    with torch.no_grad():
        for batch in loader:
            with autocast_for(device, amp_dtype, use_amp):
                output = forward_batch(model, batch, device)
                labels = batch["labels"].to(device)
                loss = batch_loss(
                    output["logits"],
                    labels,
                    task_type=task_type,
                    class_weights=class_weights,
                )
            losses.append(float(loss.detach().cpu()))
            counts.append(int(labels.shape[0]))
            logits = output["logits"].detach().float().cpu()
            if task_type == "regression":
                regression_predictions.extend(float(value) for value in logits.view(-1).tolist())
            else:
                probs = torch.softmax(logits, dim=-1)
                predictions.extend(int(value) for value in probs.argmax(dim=-1).tolist())
                if binary:
                    probabilities.extend(float(value) for value in probs[:, 1].tolist())
    total = sum(counts) or 1
    mean_loss = float(sum(loss * count for loss, count in zip(losses, counts)) / total)
    if task_type == "regression":
        gold = [float(example.label) for example in examples]
        metrics = regression_metrics(gold, regression_predictions)
        return {
            "loss": mean_loss,
            "metrics": metrics,
            "score": metrics["mse"],
            "errors": [],
            "labels": gold,
            "probabilities": None,
        }
    gold_labels = [encode_label(example, {}, "classification") if not isinstance(example.label, str) else predictions and 0 for example in examples]
    # String labels were encoded by the dataset map. Re-encode with the loader's label ints when present.
    encoded_gold: list[int] = []
    for example, predicted in zip(examples, predictions):
        if isinstance(example.label, str):
            # The prediction index space matches label_map. Recover gold via the example order
            # stored on the dataset. Callers pass examples in loader order, which is dataset order
            # because shuffle is off. The gold index is attached below by the caller when needed.
            encoded_gold.append(-1)
        else:
            encoded_gold.append(int(example.label))
    if any(label < 0 for label in encoded_gold):
        raise RuntimeError("String-label scoring must pass encoded gold labels.")
    metrics = classification_metrics(
        encoded_gold,
        predictions,
        probabilities if binary else None,
        binary=binary,
    )
    errors = error_list(
        examples,
        predictions,
        probabilities if binary else None,
        binary=binary,
    )
    return {
        "loss": mean_loss,
        "metrics": metrics,
        "score": metrics["macro_f1"],
        "errors": errors,
        "labels": encoded_gold,
        "probabilities": probabilities if binary else None,
    }


def _gold_indices(
    examples: list[CanonicalExample],
    label_map: dict[str, int],
    task_type: str,
) -> list[int | float]:
    return [encode_label(example, label_map, task_type) for example in examples]


def score_split(
    model: torch.nn.Module,
    examples: list[CanonicalExample],
    tokenizer: Any,
    *,
    model_name: str,
    max_length: int,
    label_map: dict[str, int],
    task_type: str,
    batch_size: int,
    device: torch.device,
    class_weights: torch.Tensor | None,
    use_amp: bool,
    amp_dtype: torch.dtype,
    binary: bool,
    expand_contractions: bool,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("Cannot score an empty split.")
    loader = make_loader(
        examples,
        tokenizer,
        model_name,
        max_length,
        label_map,
        task_type,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        expand_contractions=expand_contractions,
    )
    model.eval()
    losses: list[float] = []
    counts: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    regression_values: list[float] = []
    with torch.no_grad():
        for batch in loader:
            with autocast_for(device, amp_dtype, use_amp):
                output = forward_batch(model, batch, device)
                labels = batch["labels"].to(device)
                loss = batch_loss(output["logits"], labels, task_type=task_type, class_weights=class_weights)
            losses.append(float(loss.detach().cpu()))
            counts.append(int(labels.shape[0]))
            logits = output["logits"].detach().float().cpu()
            if task_type == "regression":
                regression_values.extend(float(value) for value in logits.view(-1).tolist())
            else:
                probs = torch.softmax(logits, dim=-1)
                predictions.extend(int(value) for value in torch.argmax(probs, dim=-1).tolist())
                if binary:
                    probabilities.extend(float(value) for value in probs[:, 1].tolist())
    total = sum(counts)
    mean_loss = float(sum(loss * count for loss, count in zip(losses, counts)) / total)
    if task_type == "regression":
        gold = [float(value) for value in _gold_indices(examples, label_map, task_type)]
        metrics = regression_metrics(gold, regression_values)
        return {"loss": mean_loss, "metrics": metrics, "score": float(metrics["mse"]), "errors": []}
    gold = [int(value) for value in _gold_indices(examples, label_map, task_type)]
    metrics = classification_metrics(gold, predictions, probabilities if binary else None, binary=binary)
    errors = error_list(examples, predictions, probabilities if binary else None, binary=binary)
    return {
        "loss": mean_loss,
        "metrics": metrics,
        "score": float(metrics["macro_f1"]),
        "errors": errors,
        "labels": gold,
        "probabilities": probabilities if binary else None,
    }


def mlflow_logger(experiment: str, run_name: str, params: dict[str, Any], tags: dict[str, str]):
    import mlflow

    tracking = PACKAGE_ROOT / "outputs" / "mlruns"
    tracking.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(str(tracking))
    mlflow.set_experiment(experiment)
    run = mlflow.start_run(run_name=run_name)
    safe_params = {key: value for key, value in params.items() if not _looks_like_text(key)}
    mlflow.log_params(safe_params)
    mlflow.set_tags(tags)
    return mlflow, run


def _looks_like_text(key: str) -> bool:
    return key in {"text", "tweet", "utterance", "context"}


def fit(
    train_examples: list[CanonicalExample],
    valid_examples: list[CanonicalExample],
    *,
    dataset: str,
    task: str,
    model_name: str,
    config: dict[str, Any],
    device: torch.device,
    run_directory: Path,
    text_source: str,
    seed: int,
    expand_contractions: bool,
    early_stopping: bool,
    fixed_epochs: int | None = None,
    resume_path: Path | None = None,
    fold_index: int | None = None,
    removed_overlap: int = 0,
) -> dict[str, Any]:
    """Train one split. Checkpoints are written only after an epoch finishes."""
    if not train_examples:
        raise ValueError("Training split is empty. No checkpoint was written.")
    task_type = task_type_for(dataset, task)
    binary = task_type == "classification" and (dataset, task) in CLOSED_LABELS and task != "taskB"
    if task == "explicit_emotion":
        binary = False
    label_map = label_map_for(dataset, task, train_examples, valid_examples)
    for example in [*train_examples, *valid_examples]:
        encode_label(example, label_map, task_type)
    num_labels = 1 if task_type == "regression" else max(label_map.values()) + 1
    higher_is_better = task_type == "classification"
    set_seed(seed if fold_index is None else seed + fold_index)
    max_length = int(config["model"]["max_length"])
    hf_name = huggingface_model_id(model_name)
    resume_state = load_trainer_state(resume_path) if resume_path is not None else None
    if resume_path is not None:
        model = load_span_model(resume_path, device=device)
        tokenizer = AutoTokenizer.from_pretrained(resume_path)
    else:
        model = build_model(
            hf_name,
            num_labels=num_labels,
            task_type=task_type,
            dropout=float(config["model"]["head_dropout"]),
            max_length=max_length,
            label_map=label_map,
        )
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
    assert_tokenizer_matches_model(model.model_name or hf_name, tokenizer)
    batch_size = batch_size_for(device, config)
    train_loader = make_loader(
        train_examples,
        tokenizer,
        model.model_name or hf_name,
        max_length,
        label_map,
        task_type,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        expand_contractions=expand_contractions,
    )
    train_labels = [int(encode_label(example, label_map, "classification")) for example in train_examples] if task_type == "classification" else []
    weights = inverse_frequency_weights(train_labels, num_labels) if task_type == "classification" else []
    class_weights = torch.tensor(weights, dtype=torch.float, device=device) if weights else None
    max_epochs = int(fixed_epochs if fixed_epochs is not None else config["training"]["max_epochs"])
    optimizer = optimizer_for(model, config)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = int(round(float(config["training"]["warmup_ratio"]) * total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_score = None
    best_epoch = None
    best_metrics: dict[str, Any] | None = None
    best_errors: list[dict[str, Any]] = []
    epochs_without_improvement = 0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        history = list(resume_state.get("history") or [])
        start_epoch = int(resume_state["epoch"]) + 1
        best_score = resume_state.get("best_score")
        best_epoch = resume_state.get("best_epoch")
        best_metrics = resume_state.get("best_metrics")
        best_errors = list(resume_state.get("best_errors") or [])
        epochs_without_improvement = int(resume_state.get("epochs_without_improvement") or 0)
        if resume_state.get("stopped_early") and early_stopping:
            return _fit_result(
                run_directory,
                best_epoch,
                best_score,
                best_metrics,
                best_errors,
                history,
                weights,
                label_map,
                stopped_early=True,
            )
    use_amp = device.type == "cuda"
    amp_dtype = torch.float32
    scaler = None
    if use_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        scaler = make_scaler(amp_dtype == torch.float16)
    patience = int(config["training"]["early_stopping_patience"])
    clip_norm = float(config["training"]["gradient_clip_norm"])
    experiment = f"{dataset}_{task}_{model_slug(model_name)}_{text_source}"
    run_name = run_directory.name if fold_index is None else f"{run_directory.name}-fold-{fold_index}"
    mlflow = None
    try:
        mlflow, _run = mlflow_logger(
            experiment,
            run_name,
            {
                "dataset": dataset,
                "task": task,
                "model": hf_name,
                "epochs": max_epochs,
                "batch_size": batch_size,
                "learning_rate": config["training"]["learning_rate"],
                "seed": seed,
                "text_source": text_source,
                "n_train": len(train_examples),
                "n_valid": len(valid_examples),
                "class_weights": ",".join(f"{value:.6f}" for value in weights),
                "removed_overlap": removed_overlap,
                "early_stopping": early_stopping,
                "fold": -1 if fold_index is None else fold_index,
            },
            {"text_source": text_source, "dataset": dataset, "task": task},
        )
    except Exception as exc:
        raise RuntimeError(f"MLflow could not start. No checkpoint was written. {exc}") from exc
    stopped_early = False
    try:
        for epoch in range(start_epoch, max_epochs + 1):
            model.train()
            running = 0.0
            seen = 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                with autocast_for(device, amp_dtype, use_amp):
                    output = forward_batch(model, batch, device)
                    loss = batch_loss(
                        output["logits"],
                        batch["labels"].to(device),
                        task_type=task_type,
                        class_weights=class_weights,
                    )
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    optimizer.step()
                scheduler.step()
                running += float(loss.detach().cpu()) * int(batch["labels"].shape[0])
                seen += int(batch["labels"].shape[0])
            train_loss = running / max(seen, 1)
            row: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss, "val_loss": None}
            improved = False
            if valid_examples:
                scored = score_split(
                    model,
                    valid_examples,
                    tokenizer,
                    model_name=model.model_name or hf_name,
                    max_length=max_length,
                    label_map=label_map,
                    task_type=task_type,
                    batch_size=batch_size,
                    device=device,
                    class_weights=class_weights,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    binary=binary,
                    expand_contractions=expand_contractions,
                )
                row["val_loss"] = scored["loss"]
                row["val_score"] = scored["score"]
                score = float(scored["score"])
                if best_score is None or (score > best_score if higher_is_better else score < best_score):
                    best_score = score
                    best_epoch = epoch
                    best_metrics = scored["metrics"]
                    best_errors = scored["errors"]
                    epochs_without_improvement = 0
                    improved = True
                else:
                    epochs_without_improvement += 1
            history.append(row)
            state = {
                "epoch": epoch,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
                "best_errors": best_errors,
                "epochs_without_improvement": epochs_without_improvement,
                "class_weights": weights,
                "label_map": label_map,
                "history": history,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": {
                    "dataset": dataset,
                    "task": task,
                    "model": model_name,
                    "text_source": text_source,
                    "seed": seed,
                    "max_epochs": max_epochs,
                    "expand_contractions": expand_contractions,
                    "early_stopping": early_stopping,
                    "fold_index": fold_index,
                    "train_ids": [example.example_id for example in train_examples],
                    "valid_ids": [example.example_id for example in valid_examples],
                    "higher_is_better": higher_is_better,
                    "binary": binary,
                    "num_labels": num_labels,
                    "removed_overlap": removed_overlap,
                },
                "stopped_early": False,
            }
            epoch_dir = run_directory / f"checkpoint-{epoch}"
            save_epoch_checkpoint(model, tokenizer, epoch_dir, state)
            write_json(run_directory / "history.json", {"history": history, "text_source": text_source})
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            if row["val_loss"] is not None:
                mlflow.log_metric("val_loss", row["val_loss"], step=epoch)
                mlflow.log_metric("val_score", row["val_score"], step=epoch)
            print(
                f"epoch {epoch} train_loss={train_loss:.4f} "
                f"val_loss={row['val_loss']} val_score={row.get('val_score')}"
            )
            if early_stopping and valid_examples and not improved and epochs_without_improvement >= patience:
                stopped_early = True
                state["stopped_early"] = True
                torch.save(state, epoch_dir / "trainer_state.pt")
                break
        if best_epoch is None:
            best_epoch = max_epochs
            best_score = history[-1]["train_loss"] if history else None
    except Exception:
        mlflow.end_run(status="FAILED")
        raise
    else:
        mlflow.end_run(status="FINISHED")
    return _fit_result(
        run_directory,
        best_epoch,
        best_score,
        best_metrics,
        best_errors,
        history,
        weights,
        label_map,
        stopped_early=stopped_early,
    )


def _fit_result(
    run_directory: Path,
    best_epoch: int | None,
    best_score: float | None,
    best_metrics: dict[str, Any] | None,
    best_errors: list[dict[str, Any]],
    history: list[dict[str, Any]],
    weights: list[float],
    label_map: dict[str, int],
    *,
    stopped_early: bool,
) -> dict[str, Any]:
    best_path = None if best_epoch is None else run_directory / f"checkpoint-{best_epoch}"
    return {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "best_errors": best_errors,
        "history": history,
        "class_weights": weights,
        "label_map": label_map,
        "best_checkpoint": None if best_path is None else str(best_path),
        "stopped_early": stopped_early,
    }


def _examples_by_id(examples: list[CanonicalExample]) -> dict[str, CanonicalExample]:
    return {example.example_id: example for example in examples}


def materialize(examples: list[CanonicalExample], ids: list[str], split_name: str) -> list[CanonicalExample]:
    return subset_by_ids(examples, ids, split_name=split_name)


def write_split_report(
    directory: Path,
    stem: str,
    *,
    role: str,
    independent: bool,
    result: dict[str, Any],
    dataset: str,
    task: str,
    model_name: str,
    text_source: str,
    extra: dict[str, Any] | None = None,
) -> None:
    task_type = task_type_for(dataset, task)
    payload: dict[str, Any] = {
        "role": role,
        "is_independent_test": independent,
        "reported_as_test": independent,
        "dataset": dataset,
        "task": task,
        "model": huggingface_model_id(model_name),
        "text_source": text_source,
        "best_epoch": result.get("best_epoch"),
        "best_checkpoint": result.get("best_checkpoint"),
        "metrics": result.get("best_metrics"),
        "errors": result.get("best_errors") or [],
        "class_weights": result.get("class_weights"),
    }
    if extra:
        payload.update(extra)
    write_json(directory / f"{stem}.json", payload)
    if task_type == "classification" and result.get("best_metrics"):
        write_classification_artifacts(
            directory,
            stem,
            result["best_metrics"],
            history=result.get("history"),
            labels=None,
            probabilities=None,
            title=f"{dataset} {task} {role}",
        )
    elif result.get("history"):
        from member1_text.plots import plot_loss

        plot_loss(result["history"], directory / f"{stem}_loss.png", title=f"{dataset} {task} loss")


def run_cv(args: argparse.Namespace, config: dict[str, Any], dataset: str, task: str, model_name: str) -> None:
    if dataset == "sarc":
        raise ValueError("SARC keeps its official train and test files and is not reshuffled into five folds.")
    examples, removed = _load_training_corpus(args, dataset, task)
    folds = stratified_kfold(examples, n_splits=5, seed=int(config["project"]["seed"]))
    directory = run_dir_for(dataset, task, model_name, "stratified_5fold", args.output_dir)
    refuse_existing_run(directory)
    device = training_device(args.device)
    fold_payloads = []
    for index, (train_ids, valid_ids) in enumerate(folds):
        fold_dir = directory / f"fold-{index}"
        result = fit(
            materialize(examples, train_ids, "train"),
            materialize(examples, valid_ids, "validation"),
            dataset=dataset,
            task=task,
            model_name=model_name,
            config=config,
            device=device,
            run_directory=fold_dir,
            text_source=args.text_source,
            seed=int(config["project"]["seed"]),
            expand_contractions=args.expand_contractions,
            early_stopping=True,
            fold_index=index,
            removed_overlap=removed,
        )
        fold_payload = {
            "fold": index,
            "best_epoch": result["best_epoch"],
            "best_score": result["best_score"],
            "metrics": result["best_metrics"],
            "n_train": len(train_ids),
            "n_valid": len(valid_ids),
            "best_checkpoint": result["best_checkpoint"],
        }
        write_json(fold_dir / "fold_result.json", fold_payload)
        fold_payloads.append(fold_payload)
    _write_cv_summary(dataset, task, model_name, args, fold_payloads)


def _write_cv_summary(
    dataset: str,
    task: str,
    model_name: str,
    args: argparse.Namespace,
    fold_payloads: list[dict[str, Any]],
) -> None:
    scores = [float(item["best_score"]) for item in fold_payloads]
    best_index = max(range(len(scores)), key=lambda index: scores[index])
    if task_type_for(dataset, task) == "regression":
        best_index = min(range(len(scores)), key=lambda index: scores[index])
    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    epochs = [int(item["best_epoch"]) for item in fold_payloads]
    payload: dict[str, Any] = {
        "role": "cross_validation_validation",
        "is_independent_test": False,
        "reported_as_test": False,
        "used_for_epoch_selection_only": True,
        "copy_best_fold_weights": False,
        "dataset": dataset,
        "task": task,
        "model": huggingface_model_id(model_name),
        "text_source": args.text_source,
        "folds": fold_payloads,
        "per_fold_best_epochs": epochs,
        "best_fold": best_index,
    }
    if task_type_for(dataset, task) == "classification":
        payload["validation_macro_f1_mean"] = mean
        payload["validation_macro_f1_std"] = std
        payload["best_fold_validation_macro_f1"] = scores[best_index]
    else:
        payload["validation_mse_mean"] = mean
        payload["validation_mse_std"] = std
        payload["best_fold_validation_mse"] = scores[best_index]
    destination = report_dir(dataset, task, model_name, args.output_dir)
    write_json(destination / "cv_validation.json", payload)
    print(f"Wrote {destination / 'cv_validation.json'}")


def run_deployment_refit(args: argparse.Namespace, config: dict[str, Any], dataset: str, task: str, model_name: str) -> None:
    if dataset not in {"mustard", "mustard_pp"}:
        raise ValueError(
            f"{dataset} is not refit onto a deployment checkpoint. "
            "Use its official split or train holdout, and do not copy validation scores forward."
        )
    summary_path = report_dir(dataset, task, model_name, args.output_dir) / "cv_validation.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"No cross-validation summary at {summary_path}. "
            "Run --split stratified_5fold first. No deployment checkpoint was written."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    epochs = [int(value) for value in summary["per_fold_best_epochs"]]
    chosen = median_epoch(epochs)
    examples, removed = _load_training_corpus(args, dataset, task)
    directory = run_dir_for(dataset, task, model_name, "deployment_refit", args.output_dir)
    refuse_existing_run(directory)
    device = training_device(args.device)
    result = fit(
        examples,
        [],
        dataset=dataset,
        task=task,
        model_name=model_name,
        config=config,
        device=device,
        run_directory=directory,
        text_source=args.text_source,
        seed=int(config["project"]["seed"]),
        expand_contractions=args.expand_contractions,
        early_stopping=False,
        fixed_epochs=chosen,
        removed_overlap=removed,
    )
    final_epoch = int(result["history"][-1]["epoch"]) if result["history"] else chosen
    final_dir = directory / f"checkpoint-{final_epoch}"
    if not (final_dir / "config.json").is_file():
        raise RuntimeError(
            f"Deployment refit finished epoch accounting at {final_epoch} "
            f"but {final_dir} has no checkpoint."
        )
    _publish_directory(final_dir, directory)
    payload = {
        "role": "deployment_refit",
        "is_independent_test": False,
        "reported_as_test": False,
        "dataset": dataset,
        "task": task,
        "model": huggingface_model_id(model_name),
        "text_source": args.text_source,
        "per_fold_best_epochs": epochs,
        "deployment_epochs": chosen,
        "cv_validation": str(summary_path),
        "checkpoint": str(directory),
        "early_stopping_during_refit": False,
        "copy_best_fold_weights": False,
    }
    write_json(report_dir(dataset, task, model_name, args.output_dir) / "deployment_refit.json", payload)
    print(f"Deployment checkpoint: {directory}")


def _publish_directory(source: Path, destination: Path) -> None:
    """Copy the finished refit epoch to the directory the API loads."""
    for item in source.iterdir():
        target = destination / item.name
        if item.is_file():
            target.write_bytes(item.read_bytes())


def run_speaker_grouped(args: argparse.Namespace, config: dict[str, Any], dataset: str, task: str, model_name: str) -> None:
    if dataset not in {"mustard", "mustard_pp"}:
        raise ValueError(f"{dataset} has no speaker field for a speaker-grouped split.")
    examples, removed = _load_training_corpus(args, dataset, task)
    train_ids, holdout_ids = speaker_grouped_split(examples, seed=int(config["project"]["seed"]))
    directory = run_dir_for(dataset, task, model_name, "speaker_grouped", args.output_dir)
    refuse_existing_run(directory)
    result = fit(
        materialize(examples, train_ids, "train"),
        materialize(examples, holdout_ids, "speaker_holdout"),
        dataset=dataset,
        task=task,
        model_name=model_name,
        config=config,
        device=training_device(args.device),
        run_directory=directory,
        text_source=args.text_source,
        seed=int(config["project"]["seed"]),
        expand_contractions=args.expand_contractions,
        early_stopping=True,
        removed_overlap=removed,
    )
    write_split_report(
        report_dir(dataset, task, model_name, args.output_dir),
        "speaker_grouped",
        role="speaker_grouped",
        independent=False,
        result=result,
        dataset=dataset,
        task=task,
        model_name=model_name,
        text_source=args.text_source,
        extra={
            "note": (
                "Speaker-grouped holdout is a speaker-shift check. "
                "It is not the five-fold validation mean and not an independent test score."
            )
        },
    )


def run_sarc_official(args: argparse.Namespace, config: dict[str, Any], model_name: str) -> None:
    task = "sarcasm"
    train_examples, removed = _load_training_corpus(args, "sarc", task, sarc_part="train")
    test_examples, test_removed = _load_training_corpus(args, "sarc", task, sarc_part="test")
    removed += test_removed
    train_ids, valid_ids = stratified_holdout(train_examples, seed=int(config["project"]["seed"]), valid_size=0.1)
    directory = run_dir_for("sarc", task, model_name, "official_balanced", args.output_dir)
    refuse_existing_run(directory)
    result = fit(
        materialize(train_examples, train_ids, "train"),
        materialize(train_examples, valid_ids, "train_holdout"),
        dataset="sarc",
        task=task,
        model_name=model_name,
        config=config,
        device=training_device(args.device),
        run_directory=directory,
        text_source=args.text_source,
        seed=int(config["project"]["seed"]),
        expand_contractions=args.expand_contractions,
        early_stopping=True,
        removed_overlap=removed,
    )
    reports = report_dir("sarc", task, model_name, args.output_dir)
    write_split_report(
        reports,
        "train_holdout",
        role="train_holdout",
        independent=False,
        result=result,
        dataset="sarc",
        task=task,
        model_name=model_name,
        text_source=args.text_source,
        extra={"note": "Early-stopping slice of the official SARC train file. Not the official test."},
    )
    test_result = _score_checkpoint_on(
        Path(result["best_checkpoint"]),
        test_examples,
        dataset="sarc",
        task=task,
        device_name=args.device,
        expand_contractions=args.expand_contractions,
    )
    test_result["history"] = result["history"]
    test_result["class_weights"] = result["class_weights"]
    write_split_report(
        reports,
        "independent_test",
        role="independent_test",
        independent=True,
        result=test_result,
        dataset="sarc",
        task=task,
        model_name=model_name,
        text_source=args.text_source,
        extra={"note": "Official SARC balanced test. The epoch was chosen on a slice of official train."},
    )


def run_semeval(args: argparse.Namespace, config: dict[str, Any], task: str, model_name: str, split: str) -> None:
    train_examples, removed = _load_training_corpus(args, "semeval2018", task)
    if split == "official_test":
        load_official_test(task)
    train_ids, valid_ids = stratified_holdout(train_examples, seed=int(config["project"]["seed"]), valid_size=0.1)
    directory = run_dir_for("semeval2018", task, model_name, split, args.output_dir)
    refuse_existing_run(directory)
    result = fit(
        materialize(train_examples, train_ids, "train"),
        materialize(train_examples, valid_ids, "train_holdout"),
        dataset="semeval2018",
        task=task,
        model_name=model_name,
        config=config,
        device=training_device(args.device),
        run_directory=directory,
        text_source=args.text_source,
        seed=int(config["project"]["seed"]),
        expand_contractions=args.expand_contractions,
        early_stopping=True,
        removed_overlap=removed,
    )
    reports = report_dir("semeval2018", task, model_name, args.output_dir)
    write_split_report(
        reports,
        "train_holdout",
        role="train_holdout",
        independent=False,
        result=result,
        dataset="semeval2018",
        task=task,
        model_name=model_name,
        text_source=args.text_source,
        extra={"note": "Slice carved from SemEval train. This is not the official test."},
    )
    if split == "official_test":
        test_examples = apply_text_source(load_official_test(task), args.text_source)
        test_result = _score_checkpoint_on(
            Path(result["best_checkpoint"]),
            test_examples,
            dataset="semeval2018",
            task=task,
            device_name=args.device,
            expand_contractions=args.expand_contractions,
        )
        test_result["history"] = result["history"]
        test_result["class_weights"] = result["class_weights"]
        write_split_report(
            reports,
            "independent_test",
            role="independent_test",
            independent=True,
            result=test_result,
            dataset="semeval2018",
            task=task,
            model_name=model_name,
            text_source=args.text_source,
        )


def _score_checkpoint_on(
    checkpoint: Path,
    examples: list[CanonicalExample],
    *,
    dataset: str,
    task: str,
    device_name: str | None,
    expand_contractions: bool,
) -> dict[str, Any]:
    device = training_device(device_name)
    model = load_span_model(checkpoint, device=device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    assert_tokenizer_matches_model(model.model_name, tokenizer)
    task_type = model.task_type
    binary = task_type == "classification" and model.num_labels == 2
    use_amp = device.type == "cuda"
    amp_dtype = torch.float32
    if use_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    state = load_trainer_state(checkpoint)
    weights = state.get("class_weights") or []
    class_weights = torch.tensor(weights, dtype=torch.float, device=device) if weights else None
    scored = score_split(
        model,
        examples,
        tokenizer,
        model_name=model.model_name,
        max_length=model.max_length,
        label_map=model.label_map,
        task_type=task_type,
        batch_size=batch_size_for(device, load_config()),
        device=device,
        class_weights=class_weights,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        binary=binary,
        expand_contractions=expand_contractions,
    )
    return {
        "best_epoch": state.get("epoch"),
        "best_checkpoint": str(checkpoint),
        "best_metrics": scored["metrics"],
        "best_errors": scored["errors"],
        "history": [],
        "class_weights": weights,
    }


def _load_training_corpus(
    args: argparse.Namespace,
    dataset: str,
    task: str,
    sarc_part: str | None = None,
) -> tuple[list[CanonicalExample], int]:
    if dataset == "mustard":
        examples = load_mustard()
        try:
            write_text_provenance()
        except Exception as exc:
            print(f"MUStARD provenance was not written: {exc}", file=sys.stderr)
    elif dataset == "mustard_pp":
        examples = load_mustard_pp(task=task)
    elif dataset == "sarc":
        examples = load_sarc(split=sarc_part or "train")
    else:
        examples = load_semeval(default_train_path(task), task=task, split="train")
    examples, removed = maybe_drop_overlap(dataset, examples, args.drop_mustard_overlap)
    return apply_text_source(examples, args.text_source), removed


def resume(args: argparse.Namespace) -> None:
    checkpoint = Path(args.resume)
    state = load_trainer_state(checkpoint)
    stored = state["config"]
    dataset = normalize_dataset(stored["dataset"])
    task = normalize_task(dataset, stored["task"])
    model_name = stored["model"]
    config = load_config()
    examples, removed = _load_for_resume(dataset, task, args)
    train_examples = materialize(examples, stored["train_ids"], "train")
    valid_examples = materialize(examples, stored["valid_ids"], "validation") if stored["valid_ids"] else []
    run_directory = checkpoint.parent
    fit(
        train_examples,
        valid_examples,
        dataset=dataset,
        task=task,
        model_name=model_name,
        config=config,
        device=training_device(args.device),
        run_directory=run_directory,
        text_source=stored["text_source"],
        seed=int(stored["seed"]),
        expand_contractions=bool(stored["expand_contractions"]),
        early_stopping=bool(stored["early_stopping"]),
        fixed_epochs=int(stored["max_epochs"]) if not stored["valid_ids"] else None,
        resume_path=checkpoint,
        fold_index=stored.get("fold_index"),
        removed_overlap=removed,
    )
    if stored.get("fold_index") is not None:
        _continue_cv_after_resume(run_directory.parent, dataset, task, model_name, args, config)


def _load_for_resume(dataset: str, task: str, args: argparse.Namespace) -> tuple[list[CanonicalExample], int]:
    args.drop_mustard_overlap = getattr(args, "drop_mustard_overlap", False)
    if not getattr(args, "text_source", None):
        args.text_source = "gold"
    if dataset == "sarc":
        return _load_training_corpus(args, dataset, task, sarc_part="train")
    return _load_training_corpus(args, dataset, task)


def _continue_cv_after_resume(
    cv_dir: Path,
    dataset: str,
    task: str,
    model_name: str,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> None:
    examples, _removed = _load_training_corpus(args, dataset, task)
    folds = stratified_kfold(examples, n_splits=5, seed=int(config["project"]["seed"]))
    device = training_device(args.device)
    payloads = []
    for index, (train_ids, valid_ids) in enumerate(folds):
        fold_dir = cv_dir / f"fold-{index}"
        result_path = fold_dir / "fold_result.json"
        if result_path.is_file():
            payloads.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue
        existing = latest_checkpoint(fold_dir)
        result = fit(
            materialize(examples, train_ids, "train"),
            materialize(examples, valid_ids, "validation"),
            dataset=dataset,
            task=task,
            model_name=model_name,
            config=config,
            device=device,
            run_directory=fold_dir,
            text_source=args.text_source,
            seed=int(config["project"]["seed"]),
            expand_contractions=args.expand_contractions,
            early_stopping=True,
            resume_path=existing,
            fold_index=index,
        )
        payload = {
            "fold": index,
            "best_epoch": result["best_epoch"],
            "best_score": result["best_score"],
            "metrics": result["best_metrics"],
            "n_train": len(train_ids),
            "n_valid": len(valid_ids),
            "best_checkpoint": result["best_checkpoint"],
        }
        write_json(result_path, payload)
        payloads.append(payload)
    if len(payloads) == 5:
        _write_cv_summary(dataset, task, model_name, args, payloads)


def evaluate_checkpoint(
    checkpoint: Path,
    dataset: str,
    task: str,
    split_name: str,
    device_name: str | None,
    text_source: str,
) -> None:
    """Score a saved checkpoint. This does not invent a metric when the file or split is missing."""
    dataset = normalize_dataset(dataset)
    task = normalize_task(dataset, task)
    checkpoint = Path(checkpoint)
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint}. Evaluation wrote nothing."
        )
    examples = _evaluation_examples(dataset, task, split_name, text_source)
    result = _score_checkpoint_on(
        checkpoint,
        examples,
        dataset=dataset,
        task=task,
        device_name=device_name,
        expand_contractions=False,
    )
    role, independent, stem = _report_identity(dataset, split_name)
    model_name = json.loads((checkpoint / "config.json").read_text(encoding="utf-8")).get("model_name", "roberta-base")
    write_split_report(
        report_dir(dataset, task, model_name, None),
        stem,
        role=role,
        independent=independent,
        result=result,
        dataset=dataset,
        task=task,
        model_name=model_name,
        text_source=text_source,
    )


def _report_identity(dataset: str, split_name: str) -> tuple[str, bool, str]:
    if split_name in {"test", "official_test", "official_balanced"} and dataset in {"sarc", "semeval2018"}:
        return "independent_test", True, "independent_test"
    if split_name == "speaker_grouped":
        return "speaker_grouped", False, "speaker_grouped"
    if split_name == "train_holdout":
        return "train_holdout", False, "train_holdout"
    raise ValueError(
        f"Split {split_name!r} is not an evaluation split for {dataset}. "
        "Five-fold scores are written by training as cv_validation.json and are not relabeled here."
    )


def _evaluation_examples(dataset: str, task: str, split_name: str, text_source: str) -> list[CanonicalExample]:
    config = load_config()
    if dataset in {"mustard", "mustard_pp"} and split_name == "speaker_grouped":
        examples = load_mustard() if dataset == "mustard" else load_mustard_pp(task=task)
        _train_ids, holdout_ids = speaker_grouped_split(examples, seed=int(config["project"]["seed"]))
        return apply_text_source(materialize(examples, holdout_ids, "speaker_holdout"), text_source)
    if dataset == "sarc" and split_name in {"test", "official_test", "official_balanced"}:
        return apply_text_source(load_sarc(split="test"), text_source)
    if dataset == "sarc" and split_name == "train_holdout":
        train_examples = load_sarc(split="train")
        _train_ids, valid_ids = stratified_holdout(train_examples, seed=int(config["project"]["seed"]))
        return apply_text_source(materialize(train_examples, valid_ids, "train_holdout"), text_source)
    if dataset == "semeval2018" and split_name in {"test", "official_test"}:
        return apply_text_source(load_official_test(task), text_source)
    if dataset == "semeval2018" and split_name == "train_holdout":
        train_examples = load_semeval(default_train_path(task), task=task, split="train")
        _train_ids, valid_ids = stratified_holdout(train_examples, seed=int(config["project"]["seed"]))
        return apply_text_source(materialize(train_examples, valid_ids, "train_holdout"), text_source)
    raise ValueError(f"Cannot evaluate {dataset} with split {split_name}.")


def dispatch(args: argparse.Namespace) -> None:
    if not args.dataset or not args.task:
        raise ValueError("--dataset and --task are required unless --resume is set.")
    config = load_config()
    dataset = normalize_dataset(args.dataset)
    task = normalize_task(dataset, args.task)
    model_name = args.model or config["model"]["primary"]
    if model_slug(model_name) not in {"roberta-base", "deberta-v3-base"}:
        raise ValueError("The trainer fine-tunes roberta-base or deberta-v3-base. Baselines have their own module.")
    split = args.split or default_split(dataset, config)
    if dataset == "mustard":
        try:
            load_mustard()
        except DatasetNotAvailableError:
            raise
    if split == "stratified_5fold":
        run_cv(args, config, dataset, task, model_name)
    elif split == "deployment_refit":
        run_deployment_refit(args, config, dataset, task, model_name)
    elif split == "speaker_grouped":
        run_speaker_grouped(args, config, dataset, task, model_name)
    elif split == "official_balanced":
        if dataset != "sarc":
            raise ValueError("official_balanced is the SARC split.")
        run_sarc_official(args, config, model_name)
    elif split in {"train_holdout", "official_test"}:
        if dataset != "semeval2018":
            raise ValueError(f"{split} is the SemEval split.")
        run_semeval(args, config, task, model_name, split)
    else:
        raise ValueError(f"Unknown split {split!r}.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Member 1 span-pooling transformer.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--text-source", choices=("gold", "asr"), default="gold")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--expand-contractions", action="store_true")
    parser.add_argument("--drop-mustard-overlap", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    require_python_311()
    args = parse_args(argv)
    try:
        if args.resume:
            resume(args)
        else:
            dispatch(args)
    except (DatasetNotAvailableError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
