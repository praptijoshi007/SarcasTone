"""Uncertainty-aware prediction — temperature scaling and entropy.

Research contribution E of SarcasTone.

Temperature scaling
-------------------
A single scalar T > 0 divides the logits before softmax:

    p_calibrated = softmax( logits / T )

T is fitted post-training on a held-out validation set by minimising NLL.
During training T is fixed to 1.0.

Predictive entropy
------------------
H = -∑ p_i log p_i   (nats)

For binary classification H ∈ [0, log 2]. The uncertainty threshold
marks predictions where H > 0.5 * log(2) (probability below 60/40 split).

Usage
-----
1. At training time:  UncertaintyHead wraps the classifier — just returns
   raw logits (T=1) and entropy.
2. After training:    call fit_temperature(val_loader, model) to fit T on the
   validation split.
3. At inference:      predict_with_uncertainty(logits, temperature) returns
   calibrated probabilities + entropy + uncertain flag.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

# Entropy threshold: > this value → flagged as uncertain
# For binary classification max entropy = ln(2) ≈ 0.693
_BINARY_MAX_ENTROPY = math.log(2)
_UNCERTAINTY_THRESHOLD = 0.5 * _BINARY_MAX_ENTROPY   # ≈ 0.347


class TemperatureScaler(nn.Module):
    """Learnable temperature scalar for post-hoc calibration.

    T is kept positive via softplus: T = softplus(raw_T).
    """

    def __init__(self, init_value: float = 1.0) -> None:
        super().__init__()
        # Initialise raw such that softplus(raw) ≈ init_value
        # softplus(x) = ln(1 + e^x)  → x = ln(e^v - 1) for v >> 0
        init_raw = math.log(math.exp(init_value) - 1 + 1e-9)
        self.raw_temperature = nn.Parameter(torch.tensor([init_raw]))

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.raw_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature. Returns calibrated logits."""
        return logits / self.temperature


def predictive_entropy(probabilities: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Shannon entropy of class probability distribution.

    Args:
        probabilities: [B, C] probability tensor (must sum to 1).
    Returns:
        entropy: [B] in nats.
    """
    return -(probabilities * (probabilities + eps).log()).sum(dim=-1)


def predict_with_uncertainty(
    logits: torch.Tensor,
    temperature: float | torch.Tensor = 1.0,
    uncertainty_threshold: float = _UNCERTAINTY_THRESHOLD,
) -> dict[str, torch.Tensor]:
    """Convert raw logits to calibrated prediction + uncertainty.

    Args:
        logits:      [B, 2] raw classifier output.
        temperature: scalar or tensor T > 0.
        uncertainty_threshold: entropy above which prediction is flagged.

    Returns:
        probabilities:  [B, 2]
        label_index:    [B] int
        entropy:        [B]
        uncertain:      [B] bool
    """
    if isinstance(temperature, (int, float)):
        T = max(float(temperature), 1e-6)
        cal_logits = logits / T
    else:
        T = temperature.clamp(min=1e-6)
        cal_logits = logits / T

    probs = torch.softmax(cal_logits, dim=-1)
    labels = probs.argmax(dim=-1)
    entropy = predictive_entropy(probs)
    uncertain = entropy > uncertainty_threshold

    return {
        "probabilities": probs,
        "label_index": labels,
        "entropy": entropy,
        "uncertain": uncertain,
    }


def fit_temperature(
    model: nn.Module,
    val_loader,
    device: torch.device,
    max_iter: int = 1000,
    lr: float = 0.01,
) -> float:
    """Fit temperature on validation set by minimising NLL.

    Args:
        model:       the full SarcasToneFusionModel (must have .temperature_scaler).
        val_loader:  DataLoader yielding batches with 'label' key.
        device:      torch device.
        max_iter:    optimisation steps.
        lr:          learning rate for temperature.

    Returns:
        Fitted temperature value (float).
    """
    if not hasattr(model, "temperature_scaler"):
        raise AttributeError("model has no attribute 'temperature_scaler'")

    scaler: TemperatureScaler = model.temperature_scaler
    # Freeze everything except temperature
    for name, param in model.named_parameters():
        param.requires_grad_(name == "temperature_scaler.raw_temperature")

    optimiser = torch.optim.LBFGS(
        [scaler.raw_temperature], lr=lr, max_iter=max_iter
    )
    criterion = nn.CrossEntropyLoss()

    # Collect all logits and labels
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            labels = batch["label"].to(device)
            # We need raw logits; run forward with temperature=1
            out = model(batch, return_logits_only=True)
            all_logits.append(out["logits"].cpu())
            all_labels.append(labels.cpu())

    logits_cat = torch.cat(all_logits).to(device)
    labels_cat = torch.cat(all_labels).to(device)

    def _closure():
        optimiser.zero_grad()
        scaled = logits_cat / scaler.temperature
        loss = criterion(scaled, labels_cat)
        loss.backward()
        return loss

    optimiser.step(_closure)

    fitted_T = float(scaler.temperature.item())
    log.info("Temperature calibration: T = %.4f", fitted_T)

    # Restore all gradients
    for param in model.parameters():
        param.requires_grad_(True)

    return fitted_T
