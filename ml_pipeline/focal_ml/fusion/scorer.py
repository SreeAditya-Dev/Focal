"""Combine the rule layer and the CNN into one decision.

The blend is a weighted sum per issue, but two details make it more than an
average.

**The weight is per issue, not global.** Phase 2 measured where the rules are
trustworthy and where they are not: brightness is a near-definitive test for
exposure, while no global statistic identifies a localised defect. Using one
blending constant everywhere would either discard the rules' reliability on
exposure or grant them authority on defects they have not earned. The weights
below encode those measurements directly.

**Severity is blended by confidence, not by the fixed weight.** A source that
did not detect an issue has no opinion about how severe it is; its severity is
zero because it saw nothing, not because it judged the problem mild. Averaging
that zero in would systematically under-report severity whenever the two
sources disagree. Weighting each source's severity by its own confidence means
a silent source abstains instead of voting for "not severe".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from focal_ml.constants import (
    ISSUE_TYPES,
    MIN_SEVERITY_SCORE,
    compute_quality_score,
    quality_label,
    severity_bucket_from_score,
    severity_name,
)
from focal_ml.fusion.rules import RuleOutcome

#: How much of each issue's decision comes from the rules, the rest from the
#: CNN. Derived from the Phase 2 measurements rather than tuned as one constant:
#:
#:   exposure    the rules are near-definitive — mean luma and clipping
#:               fractions measure the failure directly
#:   blur/noise  the rules are strong but content-dependent; a photo of gravel
#:               reads as noisy, a photo of a blank wall reads as soft
#:   corruption  the rules catch the mechanisms they were written for and are
#:               blind to any other
#:   defect      the rules detect only severe vignetting and severe smudging;
#:               mild cases overlap the clean population outright
RULE_WEIGHT: dict[str, float] = {
    "underexposure": 0.55,
    "overexposure": 0.55,
    "blur": 0.45,
    "noise": 0.45,
    "corruption": 0.30,
    "defect": 0.15,
}

#: Fused confidence below which an issue is not reported at all.
REPORT_THRESHOLD = 0.35


@dataclass
class DetectedIssue:
    type: str
    severity: str
    severity_score: float
    confidence: float
    rule_confidence: float
    cnn_confidence: float
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "severity": self.severity,
            "severity_score": round(self.severity_score, 4),
            "confidence": round(self.confidence, 4),
            "rule_confidence": round(self.rule_confidence, 4),
            "cnn_confidence": round(self.cnn_confidence, 4),
            "evidence": self.evidence,
        }


@dataclass
class FusionResult:
    quality_score: float
    quality_label: str
    issues: list[DetectedIssue]
    #: Every issue's fused confidence, including those below the report
    #: threshold. The API omits these; evaluation needs them.
    all_confidences: dict[str, float] = field(default_factory=dict)
    all_severities: dict[str, float] = field(default_factory=dict)


def fuse(
    rule_outcomes: Mapping[str, RuleOutcome],
    cnn_presence: Mapping[str, float] | None = None,
    cnn_severity: Mapping[str, float] | None = None,
    *,
    rule_weight: Mapping[str, float] | None = None,
    report_threshold: float = REPORT_THRESHOLD,
) -> FusionResult:
    """Merge rule and CNN outputs into a score, a label and an issue list.

    ``cnn_presence`` may be ``None``, in which case the rules decide alone. That
    path is what lets the service degrade to a working — if less accurate —
    analyser when no model checkpoint is available, rather than failing.
    """
    weights = dict(RULE_WEIGHT)
    if rule_weight:
        weights.update(rule_weight)

    issues: list[DetectedIssue] = []
    confidences: dict[str, float] = {}
    severities: dict[str, float] = {}

    for issue in ISSUE_TYPES:
        outcome = rule_outcomes.get(issue)
        rule_confidence = outcome.confidence if outcome else 0.0
        rule_severity = outcome.severity if outcome else 0.0

        if cnn_presence is None:
            alpha = 1.0
            model_confidence = 0.0
            model_severity = 0.0
        else:
            alpha = weights.get(issue, 0.4)
            model_confidence = float(cnn_presence.get(issue, 0.0))
            model_severity = float((cnn_severity or {}).get(issue, 0.0))

        confidence = alpha * rule_confidence + (1.0 - alpha) * model_confidence

        # Confidence-weighted severity: a source that saw nothing abstains
        # rather than voting for "not severe".
        rule_vote = alpha * rule_confidence
        model_vote = (1.0 - alpha) * model_confidence
        total_vote = rule_vote + model_vote
        if total_vote > 1e-6:
            severity = (rule_vote * rule_severity + model_vote * model_severity) / total_vote
        else:
            severity = 0.0

        confidences[issue] = round(confidence, 4)
        severities[issue] = round(severity, 4)

        if confidence >= report_threshold:
            severity = max(severity, MIN_SEVERITY_SCORE)
            issues.append(
                DetectedIssue(
                    type=issue,
                    severity=severity_name(severity_bucket_from_score(severity)),
                    severity_score=severity,
                    confidence=confidence,
                    rule_confidence=rule_confidence,
                    cnn_confidence=model_confidence,
                    evidence=list(outcome.evidence) if outcome else [],
                )
            )

    # The score is computed only from reported issues. Letting sub-threshold
    # detections bleed into it would mean an image with six faint signals and no
    # actual problem scored worse than one with a real defect.
    reported = {issue.type: issue.severity_score for issue in issues}
    reported_confidence = {issue.type: issue.confidence for issue in issues}
    score = compute_quality_score(reported, reported_confidence)

    issues.sort(key=lambda item: (item.severity_score * item.confidence), reverse=True)

    return FusionResult(
        quality_score=score,
        quality_label=quality_label(score),
        issues=issues,
        all_confidences=confidences,
        all_severities=severities,
    )


def summarise(result: FusionResult) -> str:
    """One-line human summary, used in logs and as an API convenience field."""
    if not result.issues:
        return f"No quality issues detected (score {result.quality_score:.0f})."
    parts = [f"{issue.severity} {issue.type} ({issue.confidence:.0%} confidence)" for issue in result.issues]
    return f"Score {result.quality_score:.0f} ({result.quality_label}). Detected: " + ", ".join(parts) + "."
