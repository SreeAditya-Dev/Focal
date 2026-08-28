"""Decision layer: interpretable rules, and their fusion with the CNN."""

from focal_ml.fusion.rules import (
    IssueRule,
    Ramp,
    RampGroup,
    RuleConfig,
    RuleOutcome,
    Term,
    default_rules,
    evaluate_rules,
)
from focal_ml.fusion.scorer import DetectedIssue, FusionResult, fuse, summarise

__all__ = [
    "DetectedIssue",
    "FusionResult",
    "fuse",
    "summarise",
    "IssueRule",
    "Ramp",
    "RampGroup",
    "RuleConfig",
    "RuleOutcome",
    "Term",
    "default_rules",
    "evaluate_rules",
]
