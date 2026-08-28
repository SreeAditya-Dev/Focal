"""Shared definitions for the Focal quality model.

Everything in this module is a *contract* between three otherwise independent
parts of the system:

  * the synthetic dataset generator, which writes ground-truth labels,
  * the training / evaluation code, which consumes them,
  * the runtime fusion scorer, which must produce scores on the same scale.

Keeping them here means the ground-truth ``quality_score`` in the manifest and
the predicted ``quality_score`` returned by the API are computed by the exact
same function, so the two are directly comparable at evaluation time.
"""

from __future__ import annotations

from typing import Mapping, Sequence

# --------------------------------------------------------------------------
# Issue taxonomy
# --------------------------------------------------------------------------

#: The six quality problems the system detects. Order is fixed and is used as
#: the channel order of the model's multi-label output heads — do not reorder.
ISSUE_TYPES: tuple[str, ...] = (
    "blur",
    "underexposure",
    "overexposure",
    "noise",
    "corruption",
    "defect",
)

ISSUE_INDEX: dict[str, int] = {name: i for i, name in enumerate(ISSUE_TYPES)}

#: Human-readable descriptions, surfaced in the API response and the UI.
ISSUE_DESCRIPTIONS: dict[str, str] = {
    "blur": "Insufficient sharpness from defocus or motion",
    "underexposure": "Image is too dark; detail lost in shadows",
    "overexposure": "Image is too bright; detail lost in highlights",
    "noise": "Visible sensor or compression noise",
    "corruption": "Structural degradation such as block artifacts or data loss",
    "defect": "Localised visual defect such as scratches, dust or smudging",
}

# --------------------------------------------------------------------------
# Severity
# --------------------------------------------------------------------------

#: Severity is modelled as a continuous 0-1 score. Buckets exist only to give
#: the API a human-readable label; the continuous score drives the maths.
SEVERITY_NAMES: dict[int, str] = {0: "none", 1: "low", 2: "medium", 3: "high"}

#: Continuous-score span each bucket occupies. The generator samples within
#: these spans; ``severity_bucket_from_score`` inverts the mapping.
SEVERITY_SCORE_RANGES: dict[int, tuple[float, float]] = {
    1: (0.15, 0.40),
    2: (0.40, 0.70),
    3: (0.70, 1.00),
}

#: Below this, a degradation is considered visually irrelevant.
MIN_SEVERITY_SCORE: float = 0.15


def severity_bucket_from_score(score: float) -> int:
    """Map a continuous 0-1 severity score onto a low/medium/high bucket."""
    if score < SEVERITY_SCORE_RANGES[2][0]:
        return 1
    if score < SEVERITY_SCORE_RANGES[3][0]:
        return 2
    return 3


def severity_name(bucket: int) -> str:
    return SEVERITY_NAMES.get(bucket, "none")


# --------------------------------------------------------------------------
# Quality scoring
# --------------------------------------------------------------------------

#: How much a fully-severe instance of each issue subtracts from a perfect 100.
#:
#: Calibrated so that a single max-severity issue lands in the band a human
#: would assign it: total corruption is UNUSABLE, severe blur sits at the
#: POOR/UNUSABLE boundary, and a mild single issue still reads as ACCEPTABLE.
ISSUE_PENALTY: dict[str, float] = {
    "corruption": 70.0,
    "blur": 60.0,
    "defect": 55.0,
    "noise": 55.0,
    "underexposure": 50.0,
    "overexposure": 50.0,
}

#: Score floor for each label, checked high to low.
QUALITY_LABEL_THRESHOLDS: Sequence[tuple[float, str]] = (
    (85.0, "EXCELLENT"),
    (70.0, "ACCEPTABLE"),
    (40.0, "POOR"),
    (0.0, "UNUSABLE"),
)

QUALITY_LABELS: tuple[str, ...] = tuple(label for _, label in QUALITY_LABEL_THRESHOLDS)


def compute_quality_score(
    severities: Mapping[str, float],
    confidences: Mapping[str, float] | None = None,
) -> float:
    """Combine per-issue severities into a single 0-100 quality score.

    Args:
        severities: continuous 0-1 severity per issue. Absent issues may be
            omitted or passed as 0.
        confidences: optional 0-1 confidence per issue. At training time the
            ground truth is certain so this is omitted (implicitly 1.0); at
            inference the fusion layer passes its fused confidence, so an
            uncertain detection is penalised proportionally less.

    Returns:
        A score in [0, 100], where 100 is a flawless image.
    """
    penalty = 0.0
    for issue, severity in severities.items():
        weight = ISSUE_PENALTY.get(issue)
        if weight is None:
            continue
        sev = min(max(float(severity), 0.0), 1.0)
        conf = 1.0 if confidences is None else min(max(float(confidences.get(issue, 1.0)), 0.0), 1.0)
        penalty += weight * sev * conf
    return round(min(max(100.0 - penalty, 0.0), 100.0), 2)


def quality_label(score: float) -> str:
    """Map a 0-100 quality score onto its categorical label."""
    for threshold, label in QUALITY_LABEL_THRESHOLDS:
        if score >= threshold:
            return label
    return QUALITY_LABEL_THRESHOLDS[-1][1]


# --------------------------------------------------------------------------
# Image geometry
# --------------------------------------------------------------------------

#: Every image is resized to this longest-side length before *either* synthetic
#: degradation or classical feature extraction. Absolute-scale features such as
#: Laplacian variance and noise sigma are resolution dependent, so training
#: images and user uploads must be normalised to the same scale or the learned
#: thresholds do not transfer.
CANONICAL_LONG_SIDE: int = 768

#: Input resolution of the CNN backbone (ImageNet-pretrained MobileNetV3).
CNN_INPUT_SIZE: int = 224

#: JPEG quality used when writing generated images. High enough that the
#: re-encode does not itself introduce artifacts the model would learn to read
#: as the `corruption` or `noise` class.
GENERATED_JPEG_QUALITY: int = 95
