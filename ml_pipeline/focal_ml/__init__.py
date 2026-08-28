"""Focal — hybrid classical-CV + CNN image quality and defect analysis."""

from focal_ml.constants import (
    ISSUE_TYPES,
    QUALITY_LABELS,
    compute_quality_score,
    quality_label,
    severity_bucket_from_score,
    severity_name,
)

__version__ = "0.1.0"

__all__ = [
    "ISSUE_TYPES",
    "QUALITY_LABELS",
    "compute_quality_score",
    "quality_label",
    "severity_bucket_from_score",
    "severity_name",
    "__version__",
]
