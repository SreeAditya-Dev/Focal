"""Confidence calibration and uncertainty estimation.

A network trained with `BCEWithLogitsLoss` produces scores that rank well and
mean little. Because `pos_weight` deliberately inflates the positive term to
counter class imbalance, the resulting probabilities are systematically
overconfident — the model reports 0.9 for cases that are right about 70% of the
time. That matters here because the confidence is not an internal detail: it is
shown to the user, and it scales the penalty in the quality score, so a
miscalibrated 0.9 corrupts the headline number as well as the explanation.

Two independent things are provided:

  * **Temperature scaling** (Guo et al., 2017) fixes the systematic bias. One
    scalar per issue, fitted on the validation split, dividing the logit. Being
    monotonic it cannot change any ranking or any AUC — it only moves the
    probabilities onto a scale where 0.7 means 70%.

  * **Monte-Carlo dropout** estimates *epistemic* uncertainty — how much the
    model's answer depends on which subset of its features it happens to use.
    A wide spread across stochastic passes flags an image the model is
    internally inconsistent about, which is different from one it confidently
    judges borderline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from focal_ml.constants import ISSUE_TYPES


@dataclass
class Calibration:
    """Per-issue temperatures, keyed by issue name."""

    temperatures: dict[str, float]
    version: str = "calibration_v1"
    ece_before: dict[str, float] | None = None
    ece_after: dict[str, float] | None = None

    def vector(self) -> np.ndarray:
        return np.array([self.temperatures.get(issue, 1.0) for issue in ISSUE_TYPES], dtype=np.float32)

    def apply(self, logits: np.ndarray) -> np.ndarray:
        """Convert raw logits into calibrated probabilities."""
        return 1.0 / (1.0 + np.exp(-logits / self.vector()))

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "version": self.version,
                    "temperatures": self.temperatures,
                    "ece_before": self.ece_before,
                    "ece_after": self.ece_after,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "Calibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            temperatures=payload["temperatures"],
            version=payload.get("version", "calibration_v1"),
            ece_before=payload.get("ece_before"),
            ece_after=payload.get("ece_after"),
        )

    @classmethod
    def identity(cls) -> "Calibration":
        """A no-op calibration, for running before one has been fitted."""
        return cls(temperatures={issue: 1.0 for issue in ISSUE_TYPES}, version="uncalibrated")


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    """Mean gap between predicted confidence and observed frequency.

    Predictions are bucketed by confidence; within each bucket the average
    predicted probability is compared to the fraction actually positive. A
    perfectly calibrated model scores 0.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (probabilities > low) & (probabilities <= high)
        count = int(in_bin.sum())
        if count == 0:
            continue
        confidence = float(probabilities[in_bin].mean())
        accuracy = float(labels[in_bin].mean())
        error += (count / len(probabilities)) * abs(confidence - accuracy)
    return float(error)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    """Find the scalar T minimising NLL of ``sigmoid(logit / T)``.

    Optimised over log(T) so that T stays strictly positive without a
    constrained solver — a negative temperature would invert every prediction.
    """
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 1.0

    logit_tensor = torch.from_numpy(logits.astype(np.float32))
    label_tensor = torch.from_numpy(labels.astype(np.float32))
    log_temperature = torch.zeros(1, requires_grad=True)

    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter)
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logit_tensor / log_temperature.exp(), label_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.exp().item())

    # A pathological fit is worse than no calibration, so fall back to identity
    # rather than shipping a temperature that flattens every prediction.
    if not np.isfinite(temperature) or not (0.05 < temperature < 20.0):
        return 1.0
    return temperature


def fit_calibration(logits: np.ndarray, labels: np.ndarray) -> Calibration:
    """Fit one temperature per issue on held-out data.

    Args:
        logits: raw presence logits, shape (n_samples, n_issues).
        labels: binary presence labels, same shape.
    """
    temperatures: dict[str, float] = {}
    ece_before: dict[str, float] = {}
    ece_after: dict[str, float] = {}

    for index, issue in enumerate(ISSUE_TYPES):
        issue_logits = logits[:, index]
        issue_labels = labels[:, index]

        temperature = fit_temperature(issue_logits, issue_labels)

        before_probabilities = 1.0 / (1.0 + np.exp(-issue_logits))
        after_probabilities = 1.0 / (1.0 + np.exp(-issue_logits / temperature))
        before = expected_calibration_error(before_probabilities, issue_labels)
        after = expected_calibration_error(after_probabilities, issue_labels)

        # Temperature is fitted by minimising NLL, which is not the same
        # objective as calibration error, and on a small or unlucky validation
        # split the two can disagree. Since the entire purpose of this step is
        # to reduce ECE, a temperature that raises it has failed and is
        # discarded — leaving the issue uncalibrated is strictly better than
        # shipping a transform that makes its confidences less honest.
        if after > before:
            temperature = 1.0
            after = before

        temperatures[issue] = round(temperature, 4)
        ece_before[issue] = round(before, 4)
        ece_after[issue] = round(after, 4)

    return Calibration(temperatures=temperatures, ece_before=ece_before, ece_after=ece_after)


# --------------------------------------------------------------------------
# Monte-Carlo dropout
# --------------------------------------------------------------------------


def enable_dropout(model: nn.Module) -> list[nn.Module]:
    """Put only the dropout layers into training mode.

    Calling ``model.train()`` would also reactivate BatchNorm, which at a batch
    size of one normalises each sample by its own statistics and produces
    garbage. Only the dropout layers may be switched.
    """
    switched = []
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            module.train()
            switched.append(module)
    return switched


@dataclass
class UncertaintyEstimate:
    issue: str
    mean: float
    std: float
    flagged: bool


#: Standard deviation across stochastic passes above which a prediction is
#: called unstable. Chosen so that a prediction swinging by more than roughly a
#: tenth of the probability range is surfaced rather than silently averaged.
UNCERTAINTY_THRESHOLD = 0.10


@torch.no_grad()
def mc_dropout_uncertainty(
    model: nn.Module,
    image: torch.Tensor | None,
    features: torch.Tensor | None,
    passes: int = 20,
    calibration: Calibration | None = None,
) -> list[UncertaintyEstimate]:
    """Spread of the presence probabilities across stochastic forward passes.

    Each pass drops a different random subset of trunk units, so the variation
    measures how much the answer depends on any particular pathway. A confident
    prediction that survives every subset is trustworthy in a way that a
    confident prediction relying on one pathway is not.
    """
    was_training = model.training
    model.eval()
    enable_dropout(model)

    samples = []
    for _ in range(passes):
        output = model(
            image=image if model.config.use_image else None,
            features=features if model.config.use_features else None,
        )
        logits = output["presence_logits"].cpu().numpy()
        if calibration is not None:
            samples.append(calibration.apply(logits))
        else:
            samples.append(1.0 / (1.0 + np.exp(-logits)))

    model.train(was_training)

    stacked = np.stack(samples)[:, 0, :]  # (passes, n_issues)
    means = stacked.mean(axis=0)
    stds = stacked.std(axis=0)

    return [
        UncertaintyEstimate(
            issue=issue,
            mean=round(float(means[index]), 4),
            std=round(float(stds[index]), 4),
            flagged=bool(stds[index] > UNCERTAINTY_THRESHOLD),
        )
        for index, issue in enumerate(ISSUE_TYPES)
    ]
