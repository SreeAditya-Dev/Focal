"""Metric computations for the evaluation report.

Separated from the driver so the numbers can be unit-tested against cases with
known answers, rather than only ever being exercised on a real run where a
subtly wrong metric looks entirely plausible.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from focal_ml.constants import ISSUE_TYPES, QUALITY_LABELS


@dataclass
class BinaryMetrics:
    """Per-issue detection quality at one operating point."""

    issue: str
    support: int
    threshold: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def specificity(self) -> float:
        denominator = self.true_negative + self.false_positive
        return self.true_negative / denominator if denominator else 0.0

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["specificity"] = round(self.specificity, 4)
        return payload


def binary_metrics(
    labels: np.ndarray, scores: np.ndarray, issue: str, threshold: float = 0.5
) -> BinaryMetrics:
    """Detection metrics for one issue.

    Both AUCs are reported because they answer different questions on
    imbalanced data. Each issue is positive in roughly a sixth of the corpus, so
    ROC-AUC is buoyed by the large true-negative pool; PR-AUC ignores true
    negatives and is the more honest summary of how the detector behaves on the
    minority class that actually matters.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = labels.astype(int)
    predicted = (scores >= threshold).astype(int)

    true_positive = int(((predicted == 1) & (labels == 1)).sum())
    false_positive = int(((predicted == 1) & (labels == 0)).sum())
    true_negative = int(((predicted == 0) & (labels == 0)).sum())
    false_negative = int(((predicted == 0) & (labels == 1)).sum())

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # A split containing one class has no meaningful ranking metric; 0.5 and the
    # base rate are the honest degenerate values.
    if labels.min() == labels.max():
        roc_auc, pr_auc = 0.5, float(labels.mean())
    else:
        roc_auc = float(roc_auc_score(labels, scores))
        pr_auc = float(average_precision_score(labels, scores))

    return BinaryMetrics(
        issue=issue,
        support=int(labels.sum()),
        threshold=round(float(threshold), 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        roc_auc=round(roc_auc, 4),
        pr_auc=round(pr_auc, 4),
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
    )


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Threshold maximising F1, and the F1 it achieves.

    Reported alongside the fixed operating point to separate two questions: how
    well the model *ranks* images, and whether the threshold currently shipped
    is placed sensibly. A large gap between them means the ranking is fine and
    the threshold is not.
    """
    from sklearn.metrics import precision_recall_curve

    if labels.min() == labels.max():
        return 0.5, 0.0

    precision, recall, thresholds = precision_recall_curve(labels.astype(int), scores)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    # precision_recall_curve returns one more point than thresholds.
    best = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[best]), float(f1[best])


def confusion_matrix(labels: np.ndarray, predicted: np.ndarray, classes: list[str]) -> np.ndarray:
    """Rows are the true class, columns the predicted class."""
    index = {name: i for i, name in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for actual, guess in zip(labels, predicted):
        if actual in index and guess in index:
            matrix[index[actual], index[guess]] += 1
    return matrix


@dataclass
class SeverityMetrics:
    issue: str
    support: int
    mae: float
    rmse: float
    bucket_accuracy: float
    bucket_accuracy_within_one: float


def severity_metrics(
    true_severity: np.ndarray, predicted_severity: np.ndarray, present: np.ndarray, issue: str
) -> SeverityMetrics:
    """Regression error over the images where the issue is actually present.

    Scored only on positives, for the same reason the loss masks them: severity
    is undefined where an issue is absent, and including those zeros would
    report near-perfect accuracy driven entirely by correctly predicting nothing.
    """
    from focal_ml.constants import severity_bucket_from_score

    mask = present.astype(bool)
    if not mask.any():
        return SeverityMetrics(issue, 0, 0.0, 0.0, 0.0, 0.0)

    actual = true_severity[mask]
    predicted = predicted_severity[mask]
    errors = predicted - actual

    actual_buckets = np.array([severity_bucket_from_score(v) for v in actual])
    predicted_buckets = np.array([severity_bucket_from_score(v) for v in predicted])

    return SeverityMetrics(
        issue=issue,
        support=int(mask.sum()),
        mae=round(float(np.abs(errors).mean()), 4),
        rmse=round(float(np.sqrt((errors**2).mean())), 4),
        bucket_accuracy=round(float((actual_buckets == predicted_buckets).mean()), 4),
        # Ordinal tolerance: confusing "medium" with "high" is a far smaller
        # error than confusing it with "low", and exact-match accuracy hides that.
        bucket_accuracy_within_one=round(float((np.abs(actual_buckets - predicted_buckets) <= 1).mean()), 4),
    )


def score_regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Agreement between predicted and ground-truth quality scores."""
    from scipy.stats import pearsonr, spearmanr

    errors = predicted - actual
    return {
        "mae": round(float(np.abs(errors).mean()), 3),
        "rmse": round(float(np.sqrt((errors**2).mean())), 3),
        "bias": round(float(errors.mean()), 3),
        "pearson_r": round(float(pearsonr(actual, predicted)[0]), 4),
        # Spearman is the more relevant of the two here: the product's job is to
        # rank images by quality, and a monotonic miscalibration of the score
        # would hurt Pearson while leaving the ranking perfect.
        "spearman_rho": round(float(spearmanr(actual, predicted)[0]), 4),
    }


def label_accuracy(actual: list[str], predicted: list[str]) -> dict[str, float]:
    """Exact and adjacent accuracy over the four quality bands."""
    order = {name: i for i, name in enumerate(QUALITY_LABELS)}
    actual_index = np.array([order[a] for a in actual])
    predicted_index = np.array([order[p] for p in predicted])
    return {
        "exact": round(float((actual_index == predicted_index).mean()), 4),
        "within_one_band": round(float((np.abs(actual_index - predicted_index) <= 1).mean()), 4),
    }


def macro_summary(metrics: list[BinaryMetrics]) -> dict[str, float]:
    return {
        "macro_f1": round(float(np.mean([m.f1 for m in metrics])), 4),
        "macro_roc_auc": round(float(np.mean([m.roc_auc for m in metrics])), 4),
        "macro_pr_auc": round(float(np.mean([m.pr_auc for m in metrics])), 4),
        "macro_precision": round(float(np.mean([m.precision for m in metrics])), 4),
        "macro_recall": round(float(np.mean([m.recall for m in metrics])), 4),
    }


def reliability_curve(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean confidence, observed frequency and count per confidence bin."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    confidence, accuracy, counts = [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (probabilities > low) & (probabilities <= high)
        count = int(in_bin.sum())
        counts.append(count)
        confidence.append(float(probabilities[in_bin].mean()) if count else 0.0)
        accuracy.append(float(labels[in_bin].mean()) if count else 0.0)
    return np.array(confidence), np.array(accuracy), np.array(counts)
