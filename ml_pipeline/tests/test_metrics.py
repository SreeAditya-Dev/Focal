"""Tests for the evaluation metrics.

Metrics are the one part of the system with no independent check on it: a
subtly wrong F1 produces a number that looks entirely reasonable in a report,
and nothing downstream disagrees with it. So each is tested against a case whose
answer can be worked out by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics import (
    best_f1_threshold,
    binary_metrics,
    confusion_matrix,
    label_accuracy,
    reliability_curve,
    score_regression_metrics,
    severity_metrics,
)
from focal_ml.constants import QUALITY_LABELS


def test_perfect_detector():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    metrics = binary_metrics(labels, scores, "blur", threshold=0.5)

    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.roc_auc == 1.0
    assert (metrics.true_positive, metrics.false_positive) == (2, 0)
    assert (metrics.true_negative, metrics.false_negative) == (2, 0)


def test_counts_are_worked_out_by_hand():
    #                    TP   FN   FP   TN
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1])
    metrics = binary_metrics(labels, scores, "noise", threshold=0.5)

    assert (metrics.true_positive, metrics.false_negative) == (2, 1)
    assert (metrics.false_positive, metrics.true_negative) == (1, 2)
    assert metrics.precision == pytest.approx(2 / 3, abs=1e-4)
    assert metrics.recall == pytest.approx(2 / 3, abs=1e-4)
    assert metrics.f1 == pytest.approx(2 / 3, abs=1e-4)
    assert metrics.specificity == pytest.approx(2 / 3, abs=1e-4)


def test_single_class_split_degrades_gracefully():
    """A split with no positives has no meaningful ranking metric."""
    metrics = binary_metrics(np.zeros(10), np.random.random(10), "defect")
    assert metrics.roc_auc == 0.5
    assert metrics.support == 0
    assert metrics.f1 == 0.0


def test_pr_auc_is_stricter_than_roc_on_imbalanced_data():
    """Why both are reported.

    With 5% positives, a mediocre detector still posts a respectable ROC-AUC
    because the huge true-negative pool flatters it. PR-AUC ignores true
    negatives and exposes the weakness.
    """
    rng = np.random.default_rng(0)
    labels = np.zeros(2000)
    labels[:100] = 1
    scores = np.where(labels == 1, rng.normal(0.6, 0.25, 2000), rng.normal(0.4, 0.25, 2000))
    scores = np.clip(scores, 0, 1)

    metrics = binary_metrics(labels, scores, "blur")
    assert metrics.roc_auc > metrics.pr_auc
    assert metrics.pr_auc < 0.5


def test_best_threshold_beats_a_badly_placed_one():
    """Separates 'the ranking is bad' from 'the threshold is misplaced'."""
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.35, 0.32, 0.30, 0.10, 0.08, 0.05])

    at_default = binary_metrics(labels, scores, "blur", threshold=0.5).f1
    threshold, best = best_f1_threshold(labels, scores)

    assert at_default == 0.0, "nothing clears 0.5, so the fixed point scores zero"
    assert best == 1.0, "the ranking is perfect at the right threshold"
    assert 0.1 < threshold <= 0.30


def test_severity_is_scored_only_on_positives():
    """Including absent issues would report accuracy driven by predicting zero."""
    present = np.array([1, 0, 0, 0])
    actual = np.array([0.8, 0.0, 0.0, 0.0])
    predicted = np.array([0.5, 0.9, 0.9, 0.9])  # wildly wrong, but only where absent

    metrics = severity_metrics(actual, predicted, present, "blur")
    assert metrics.support == 1
    assert metrics.mae == pytest.approx(0.3, abs=1e-4)


def test_severity_bucket_tolerance():
    present = np.array([1, 1])
    actual = np.array([0.75, 0.75])       # high
    predicted = np.array([0.65, 0.20])    # medium, low

    metrics = severity_metrics(actual, predicted, present, "noise")
    assert metrics.bucket_accuracy == 0.0
    assert metrics.bucket_accuracy_within_one == 0.5, "medium-for-high is adjacent, low-for-high is not"


def test_no_positives_gives_zeroed_severity_metrics():
    metrics = severity_metrics(np.zeros(5), np.random.random(5), np.zeros(5), "defect")
    assert metrics.support == 0 and metrics.mae == 0.0


def test_score_regression_detects_bias():
    actual = np.array([50.0, 60.0, 70.0, 80.0])
    predicted = actual + 5.0

    metrics = score_regression_metrics(actual, predicted)
    assert metrics["bias"] == pytest.approx(5.0)
    assert metrics["mae"] == pytest.approx(5.0)
    assert metrics["pearson_r"] == pytest.approx(1.0)
    assert metrics["spearman_rho"] == pytest.approx(1.0)


def test_spearman_survives_monotonic_distortion():
    """Ranking is what the product needs; Spearman is the metric that says so."""
    actual = np.array([10.0, 30.0, 50.0, 70.0, 90.0])
    predicted = actual**1.5 / 20  # order-preserving, badly scaled

    metrics = score_regression_metrics(actual, predicted)
    assert metrics["spearman_rho"] == pytest.approx(1.0)
    assert metrics["pearson_r"] < 1.0


def test_confusion_matrix_orientation():
    """Rows are truth, columns are prediction — the transpose is a silent lie."""
    actual = np.array(["EXCELLENT", "EXCELLENT", "POOR"])
    predicted = np.array(["EXCELLENT", "POOR", "POOR"])

    matrix = confusion_matrix(actual, predicted, list(QUALITY_LABELS))
    excellent = QUALITY_LABELS.index("EXCELLENT")
    poor = QUALITY_LABELS.index("POOR")

    assert matrix[excellent][excellent] == 1
    assert matrix[excellent][poor] == 1, "a true EXCELLENT predicted POOR belongs in row EXCELLENT"
    assert matrix[poor][excellent] == 0
    assert matrix[poor][poor] == 1


def test_label_accuracy_adjacency():
    actual = ["EXCELLENT", "EXCELLENT", "EXCELLENT"]
    predicted = ["EXCELLENT", "ACCEPTABLE", "UNUSABLE"]

    # Reported metrics are rounded to 4 decimals, so compare at that precision.
    accuracy = label_accuracy(actual, predicted)
    assert accuracy["exact"] == pytest.approx(1 / 3, abs=1e-4)
    assert accuracy["within_one_band"] == pytest.approx(2 / 3, abs=1e-4)


def test_reliability_curve_recovers_a_known_frequency():
    # 100 predictions at 0.7 confidence, of which exactly 70 are positive.
    probabilities = np.full(100, 0.7)
    labels = np.zeros(100)
    labels[:70] = 1

    confidence, accuracy, counts = reliability_curve(probabilities, labels, bins=10)
    populated = counts > 0
    assert confidence[populated][0] == pytest.approx(0.7)
    assert accuracy[populated][0] == pytest.approx(0.7), "perfectly calibrated bin"
