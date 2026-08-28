"""Decision layer: interpretable rules now, rule+CNN fusion in Phase 4."""

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

__all__ = [
    "IssueRule",
    "Ramp",
    "RampGroup",
    "RuleConfig",
    "RuleOutcome",
    "Term",
    "default_rules",
    "evaluate_rules",
]
