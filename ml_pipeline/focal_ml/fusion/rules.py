"""Interpretable rule layer over the classical features.

This is the non-learned half of the hybrid. It converts raw measurements into a
confidence and a severity per issue, and — crucially — into a sentence a human
can check. When the system says an image is underexposed, this layer is what
lets it add "mean brightness 41, and 34% of pixels are crushed to black",
which is a claim the user can verify by looking.

Design:

  * A ``Ramp`` maps one feature onto 0-1 through a piecewise-linear response
    with an onset and a saturation point. Ramps are inverted simply by giving
    an onset above the saturation point, so "lower is worse" needs no special
    case.
  * Each issue aggregates several ramps. The aggregation differs by issue
    because the evidence structure differs: corruption takes the strongest
    single signal, since its mechanisms are mutually exclusive and one firing is
    conclusive, while blur averages agreeing signals because any one of them
    alone is unreliable.
  * Thresholds live in ``RuleConfig`` and can be refit from data — see
    ``training/fit_rules.py``. The defaults below are set from the physical
    meaning of each measurement, not from a training run.

The rule layer is deliberately weakest on `defect`, where "wrong" is defined by
location rather than by any global statistic. That is where the CNN earns its
place in the hybrid.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Mapping

from focal_ml.constants import ISSUE_TYPES, MIN_SEVERITY_SCORE

Aggregation = Literal["max", "mean", "noisy_or"]


@dataclass(frozen=True)
class Ramp:
    """Piecewise-linear map from a feature value onto [0, 1].

    ``onset`` is where the response leaves 0; ``saturate`` is where it reaches
    1. When ``saturate < onset`` the ramp is descending, which expresses
    "lower values are worse" without a separate flag.
    """

    feature: str
    onset: float
    saturate: float
    weight: float = 1.0

    def __call__(self, features: Mapping[str, float]) -> float:
        value = features.get(self.feature)
        if value is None:
            return 0.0
        if self.saturate >= self.onset:
            if value <= self.onset:
                return 0.0
            if value >= self.saturate:
                return 1.0
            return (value - self.onset) / (self.saturate - self.onset)
        if value >= self.onset:
            return 0.0
        if value <= self.saturate:
            return 1.0
        return (self.onset - value) / (self.onset - self.saturate)

    def describe(self, features: Mapping[str, float]) -> str:
        value = features.get(self.feature, 0.0)
        direction = "above" if self.saturate >= self.onset else "below"
        return f"{self.feature}={value:.4g} ({direction} {self.onset:.4g})"


@dataclass(frozen=True)
class RampGroup:
    """Several ramps that must *all* fire, combined by their weakest member.

    Some evidence is only meaningful as a conjunction. A lens smudge shows up
    as uneven sharpness across the frame — but so does uniform blur, which
    drives the same statistic down just as far. What separates them is that
    after a smudge the sharpest parts of the image are still sharp, and after
    blur nothing is. Neither condition identifies a smudge alone; together they
    do.

    Using the minimum rather than a product or mean makes this a real AND: one
    unmet condition vetoes the term no matter how strongly the others fire.
    """

    ramps: tuple[Ramp, ...]
    weight: float = 1.0
    label: str = "conjunction"

    def __call__(self, features: Mapping[str, float]) -> float:
        return min(ramp(features) for ramp in self.ramps) if self.ramps else 0.0

    def describe(self, features: Mapping[str, float]) -> str:
        return f"{self.label}: " + " and ".join(ramp.describe(features) for ramp in self.ramps)


Term = Ramp | RampGroup


@dataclass
class IssueRule:
    ramps: list[Term]
    aggregation: Aggregation = "max"
    #: Which ramp drives the severity estimate. Confidence answers "is this
    #: wrong?"; severity answers "by how much?", and the feature best suited to
    #: the second question is not always the one best suited to the first.
    severity_ramp: int = 0
    #: Confidence below which the issue is not reported at all.
    report_threshold: float = 0.25


@dataclass
class RuleOutcome:
    confidence: float
    severity: float
    evidence: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Default thresholds
#
# Every number below is grounded in the unit the feature is measured in, so
# each is arguable on its own terms rather than being a fitted constant with no
# interpretation. `training/fit_rules.py` refits them against real data.
# --------------------------------------------------------------------------


def default_rules() -> dict[str, IssueRule]:
    return {
        # Blur averages three independent views of the same physical fact —
        # attenuated high spatial frequencies. Each is individually fooled by
        # content (a photo of a blank wall, a photo of gravel); together they
        # are much harder to fool.
        "blur": IssueRule(
            ramps=[
                # Fraction of spectral energy above quarter-Nyquist. Sharp
                # photographs sit well above 0.30; heavy blur drives it under 0.10.
                Ramp("hf_energy_ratio", onset=0.34, saturate=0.12, weight=1.0),
                # Laplacian energy normalised by contrast, so it asks whether
                # the detail this scene contains is resolved.
                Ramp("sharpness_ratio", onset=0.30, saturate=0.04, weight=0.8),
                Ramp("edge_density", onset=0.045, saturate=0.006, weight=0.6),
            ],
            aggregation="mean",
            severity_ramp=0,
        ),
        # Exposure is the one class where a global statistic really is the
        # whole story, so the rules here are strong and the CNN adds little.
        "underexposure": IssueRule(
            ramps=[
                # 8-bit luma. Correct exposure centres near 110-130; below ~95
                # shadow detail starts becoming unrecoverable.
                Ramp("brightness_mean", onset=95.0, saturate=35.0, weight=1.0),
                # Pixels crushed into the bottom 16 levels are destroyed, not
                # merely dark — this is what separates underexposed from
                # legitimately low-key.
                Ramp("shadow_clip_fraction", onset=0.06, saturate=0.35, weight=0.9),
                Ramp("brightness_p95", onset=150.0, saturate=60.0, weight=0.5),
            ],
            aggregation="mean",
            severity_ramp=0,
        ),
        "overexposure": IssueRule(
            ramps=[
                Ramp("brightness_mean", onset=155.0, saturate=225.0, weight=1.0),
                Ramp("highlight_clip_fraction", onset=0.05, saturate=0.30, weight=0.9),
                Ramp("brightness_p05", onset=110.0, saturate=200.0, weight=0.5),
            ],
            aggregation="mean",
            severity_ramp=0,
        ),
        # Noise takes the max: additive grain and impulse noise are different
        # phenomena with different signatures, and an image with one need not
        # show the other. Averaging would let a silent estimator veto a loud one.
        "noise": IssueRule(
            ramps=[
                # Sigma in 8-bit levels, read from the flattest blocks. Grain
                # becomes visible around 3 and objectionable past 12.
                Ramp("noise_sigma_flat", onset=3.0, saturate=16.0, weight=1.0),
                Ramp("noise_impulse_ratio", onset=0.002, saturate=0.04, weight=1.0),
                Ramp("noise_sigma_chroma", onset=2.5, saturate=14.0, weight=0.8),
            ],
            aggregation="max",
            severity_ramp=0,
        ),
        # Corruption is disjunctive by nature: block artifacts, a torn scanline
        # and a displaced colour plane share no common trend, and an image
        # showing one will look perfectly normal to the detectors for the
        # others. Any single detector firing is the evidence.
        "corruption": IssueRule(
            ramps=[
                # Ratio of steps on the JPEG 8x8 grid to steps elsewhere; 1.0
                # means no preferential blocking.
                Ramp("blockiness", onset=1.18, saturate=2.2, weight=1.0),
                # Correlation between colour planes' edge maps; a real scene
                # keeps this above ~0.9.
                Ramp("channel_edge_correlation", onset=0.88, saturate=0.45, weight=1.0),
                Ramp("row_discontinuity", onset=6.0, saturate=25.0, weight=0.9),
                Ramp("col_discontinuity", onset=6.0, saturate=25.0, weight=0.9),
                Ramp("largest_uniform_region", onset=0.12, saturate=0.40, weight=0.7),
                Ramp("flat_block_fraction", onset=0.10, saturate=0.45, weight=0.6),
            ],
            aggregation="max",
            severity_ramp=0,
        ),
        # The weak class, by construction. A scratch in one corner moves no
        # global statistic appreciably, so these ramps are given deliberately
        # low weights and a high report threshold: better to defer to the CNN
        # than to guess loudly.
        "defect": IssueRule(
            ramps=[
                # Corner brightness relative to centre. Measured separation is
                # clean, with no overlap: clean scenes sit at 0.98-1.12 and
                # heavy vignetting at 0.61-0.70.
                Ramp("radial_falloff", onset=0.93, saturate=0.66, weight=1.0),
                # Long straight high-contrast structures. Overlaps mildly with
                # clean scenes containing architecture, hence the lower weight.
                Ramp("linear_structure", onset=0.21, saturate=0.38, weight=0.7),
                # Hot pixels and sensor dust, which move no other statistic.
                Ramp("noise_impulse_ratio", onset=0.004, saturate=0.05, weight=0.6),
                # Localised blur — a lens smudge or a fingerprint on the glass.
                #
                # Uneven sharpness alone cannot express this: uniform blur drives
                # tile uniformity down just as far as a smudge does (0.14 vs
                # 0.12, against 0.29 for a clean frame), and using it alone made
                # motion blur rank as a defect. The second condition is what
                # disambiguates — measured across scenes, the sharpest tiles
                # retain 97% of their clean sharpness under a smudge but only 1%
                # under Gaussian blur.
                #
                # Only severe smudges clear this. Mild ones overlap the clean
                # population outright, and are left to the CNN.
                RampGroup(
                    ramps=(
                        Ramp("tile_sharpness_uniformity", onset=0.16, saturate=0.07),
                        Ramp("tile_sharpness_median", onset=1.2, saturate=3.0),
                    ),
                    weight=0.9,
                    label="localised blur",
                ),
            ],
            aggregation="max",
            severity_ramp=0,
            report_threshold=0.35,
        ),
    }


@dataclass
class RuleConfig:
    rules: dict[str, IssueRule] = field(default_factory=default_rules)
    version: str = "rules_v1_defaults"

    def to_json(self, path: str | Path) -> None:
        payload = {
            "version": self.version,
            "rules": {
                issue: {
                    "aggregation": rule.aggregation,
                    "severity_ramp": rule.severity_ramp,
                    "report_threshold": rule.report_threshold,
                    "ramps": [_term_to_dict(term) for term in rule.ramps],
                }
                for issue, rule in self.rules.items()
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "RuleConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rules = {
            issue: IssueRule(
                ramps=[_term_from_dict(term) for term in spec["ramps"]],
                aggregation=spec.get("aggregation", "max"),
                severity_ramp=spec.get("severity_ramp", 0),
                report_threshold=spec.get("report_threshold", 0.25),
            )
            for issue, spec in payload["rules"].items()
        }
        return cls(rules=rules, version=payload.get("version", "rules_custom"))


def _term_to_dict(term: Term) -> dict:
    if isinstance(term, RampGroup):
        return {
            "type": "group",
            "label": term.label,
            "weight": term.weight,
            "ramps": [asdict(ramp) for ramp in term.ramps],
        }
    return {"type": "ramp", **asdict(term)}


def _term_from_dict(payload: dict) -> Term:
    # Absent "type" means a plain ramp, so configs written before groups
    # existed still load.
    if payload.get("type") == "group":
        return RampGroup(
            ramps=tuple(Ramp(**ramp) for ramp in payload["ramps"]),
            weight=payload.get("weight", 1.0),
            label=payload.get("label", "conjunction"),
        )
    return Ramp(**{key: value for key, value in payload.items() if key != "type"})


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def _aggregate(values: list[float], weights: list[float], how: Aggregation) -> float:
    if not values:
        return 0.0
    if how == "max":
        return max(value * weight for value, weight in zip(values, weights))
    if how == "mean":
        total = sum(weights)
        return sum(value * weight for value, weight in zip(values, weights)) / total if total else 0.0
    # noisy_or: independent evidence accumulates, so several weak signals can
    # together be convincing while no single one is.
    product = 1.0
    for value, weight in zip(values, weights):
        product *= 1.0 - min(1.0, value * weight)
    return 1.0 - product


def evaluate_rules(
    features: Mapping[str, float], config: RuleConfig | None = None
) -> dict[str, RuleOutcome]:
    """Score every issue from the classical features alone.

    Returns an outcome for all six issues, including those that did not fire —
    the fusion layer needs the zeros as much as the hits, since a confident
    "not blurry" from the rules should temper an uncertain "blurry" from the CNN.
    """
    config = config or RuleConfig()
    outcomes: dict[str, RuleOutcome] = {}

    for issue in ISSUE_TYPES:
        rule = config.rules.get(issue)
        if rule is None:
            outcomes[issue] = RuleOutcome(0.0, 0.0)
            continue

        responses = [term(features) for term in rule.ramps]
        weights = [term.weight for term in rule.ramps]
        confidence = min(1.0, _aggregate(responses, weights, rule.aggregation))

        index = min(rule.severity_ramp, len(responses) - 1)
        severity = responses[index] if responses else 0.0
        # A detected issue is by definition at least minimally severe; letting
        # severity fall below the floor would produce "present but harmless",
        # which the scoring function cannot represent meaningfully.
        if confidence >= rule.report_threshold:
            severity = max(severity, MIN_SEVERITY_SCORE)

        evidence = [
            rule.ramps[i].describe(features)
            for i, response in enumerate(responses)
            if response > 0.05
        ]
        outcomes[issue] = RuleOutcome(round(confidence, 4), round(severity, 4), evidence)

    return outcomes
