"""Behaviour tests for the interpretable rule layer.

The bar differs by issue on purpose. Blur, exposure and noise are global
statistical phenomena that hand-written rules genuinely can detect, so they are
held to strict standards: the right issue must win outright. Corruption and
defect are only required to fire at all — they are the classes where a rule
layer is structurally weak, and pretending otherwise in the test suite would
hide the very gap the CNN exists to fill.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataset.degradations import apply_degradation
from focal_ml.constants import ISSUE_TYPES
from focal_ml.features import extract_features
from focal_ml.fusion.rules import Ramp, RampGroup, RuleConfig, evaluate_rules

STRICT_ISSUES = ("blur", "underexposure", "overexposure", "noise")


def outcomes_for(image: np.ndarray) -> dict:
    return evaluate_rules(extract_features(image, already_canonical=True))


def degraded_outcomes(image: np.ndarray, issue: str, method: str, bucket: int = 3, seed: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    return outcomes_for(apply_degradation(image, issue, bucket, rng, method=method).image)


# --------------------------------------------------------------------------
# Ramp mechanics
# --------------------------------------------------------------------------


def test_ascending_ramp():
    ramp = Ramp("x", onset=10.0, saturate=20.0)
    assert ramp({"x": 5.0}) == 0.0
    assert ramp({"x": 10.0}) == 0.0
    assert ramp({"x": 15.0}) == pytest.approx(0.5)
    assert ramp({"x": 25.0}) == 1.0


def test_descending_ramp():
    """A ramp inverts by having its saturation below its onset."""
    ramp = Ramp("x", onset=20.0, saturate=10.0)
    assert ramp({"x": 25.0}) == 0.0
    assert ramp({"x": 15.0}) == pytest.approx(0.5)
    assert ramp({"x": 5.0}) == 1.0


def test_ramp_tolerates_missing_feature():
    assert Ramp("absent", onset=1.0, saturate=2.0)({}) == 0.0


def test_ramp_group_is_a_true_conjunction():
    """One unmet condition must veto the term regardless of the others."""
    group = RampGroup(
        ramps=(Ramp("a", onset=0.0, saturate=1.0), Ramp("b", onset=0.0, saturate=1.0))
    )
    assert group({"a": 1.0, "b": 1.0}) == 1.0
    assert group({"a": 1.0, "b": 0.0}) == 0.0, "an unmet condition must veto"
    assert group({"a": 0.6, "b": 0.9}) == pytest.approx(0.6), "weakest member governs"


def test_ramp_group_survives_json_round_trip(tmp_path):
    path = tmp_path / "rules.json"
    RuleConfig().to_json(path)
    restored = RuleConfig.from_json(path)

    groups = [term for term in restored.rules["defect"].ramps if isinstance(term, RampGroup)]
    assert groups, "the defect rule's conjunction must survive serialisation"
    assert len(groups[0].ramps) == 2


def test_config_json_round_trip(tmp_path):
    path = tmp_path / "rules.json"
    RuleConfig().to_json(path)
    restored = RuleConfig.from_json(path)

    original = RuleConfig()
    for issue in ISSUE_TYPES:
        assert restored.rules[issue].ramps == original.rules[issue].ramps
        assert restored.rules[issue].aggregation == original.rules[issue].aggregation


# --------------------------------------------------------------------------
# Behaviour on real degradations
# --------------------------------------------------------------------------


def test_clean_images_stay_quiet(clean_images):
    """The property that matters most in production.

    A quality checker that flags good photographs is worse than useless, so no
    issue may fire on clean input.
    """
    for image in clean_images:
        outcomes = outcomes_for(image)
        fired = {
            issue: outcome.confidence
            for issue, outcome in outcomes.items()
            if outcome.confidence >= RuleConfig().rules[issue].report_threshold
        }
        assert not fired, f"false positives on a clean image: {fired}"


@pytest.mark.parametrize(
    "issue,method",
    [
        ("blur", "gaussian_blur"),
        ("blur", "motion_blur"),
        ("underexposure", "gamma_gain"),
        ("overexposure", "gamma_gain"),
        ("noise", "gaussian_noise"),
        ("noise", "salt_pepper_noise"),
    ],
)
def test_strict_issues_win_outright(clean_images, issue, method):
    """For the globally-measurable issues, the right one must rank first."""
    for image in clean_images:
        outcomes = degraded_outcomes(image, issue, method)
        ranked = sorted(outcomes.items(), key=lambda item: item[1].confidence, reverse=True)
        winner, outcome = ranked[0]
        assert winner == issue, (
            f"{method}: expected {issue} to rank first, got {winner} "
            f"({{k: round(v.confidence, 3) for k, v in outcomes.items()}})"
        )
        assert outcome.confidence > 0.4


@pytest.mark.parametrize(
    "issue,method",
    [
        ("corruption", "jpeg_recompression"),
        ("corruption", "channel_desync"),
        ("corruption", "row_tear"),
        ("corruption", "occlusion"),
        ("defect", "vignette"),
        ("defect", "lens_smudge"),
    ],
)
def test_weak_issues_at_least_fire(clean_images, issue, method):
    """Corruption and defect need only clear their own reporting threshold.

    These are the classes the rule layer cannot rank reliably. Asserting more
    would encode an expectation the design does not actually make.
    """
    threshold = RuleConfig().rules[issue].report_threshold
    hits = sum(1 for image in clean_images if degraded_outcomes(image, issue, method)[issue].confidence >= threshold)
    assert hits >= 2, f"{method}: {issue} fired on only {hits}/{len(clean_images)} scenes"


def test_severity_tracks_degradation_strength(clean_images):
    """Reported severity must increase as the applied degradation worsens."""
    for image in clean_images:
        severities = [
            degraded_outcomes(image, "underexposure", "gamma_gain", bucket)["underexposure"].severity
            for bucket in (1, 2, 3)
        ]
        assert severities == sorted(severities), severities


def test_evidence_is_populated_when_an_issue_fires(clean_images):
    """Explainability is a product requirement, so it gets a test.

    Every fired issue must carry the measurements that caused it, in a form the
    API can hand to a user.
    """
    outcomes = degraded_outcomes(clean_images[0], "underexposure", "gamma_gain")
    assert outcomes["underexposure"].evidence
    assert any("brightness_mean" in item for item in outcomes["underexposure"].evidence)


def test_all_issues_always_reported(clean_images):
    """Including the zeros — fusion needs a confident 'no' as much as a 'yes'."""
    outcomes = outcomes_for(clean_images[0])
    assert set(outcomes) == set(ISSUE_TYPES)
