"""Tests for the fusion layer, calibration and the predictor.

The fusion tests use synthetic rule/CNN outputs rather than real images: the
question here is whether the arithmetic combines two opinions correctly, and
feeding it real predictions would test the feature extractor again instead.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from focal_ml.constants import ISSUE_TYPES, quality_label
from focal_ml.fusion.rules import RuleOutcome
from focal_ml.fusion.scorer import RULE_WEIGHT, fuse, summarise
from focal_ml.model.calibration import (
    Calibration,
    expected_calibration_error,
    fit_calibration,
    fit_temperature,
)


def rules(**confidences) -> dict[str, RuleOutcome]:
    """Rule outcomes for every issue; unnamed ones report nothing."""
    outcomes = {issue: RuleOutcome(0.0, 0.0) for issue in ISSUE_TYPES}
    for issue, value in confidences.items():
        confidence, severity = value if isinstance(value, tuple) else (value, value)
        outcomes[issue] = RuleOutcome(confidence, severity, [f"{issue} evidence"])
    return outcomes


def cnn(**values) -> tuple[dict, dict]:
    presence = {issue: 0.0 for issue in ISSUE_TYPES}
    severity = {issue: 0.0 for issue in ISSUE_TYPES}
    for issue, value in values.items():
        confidence, sev = value if isinstance(value, tuple) else (value, value)
        presence[issue] = confidence
        severity[issue] = sev
    return presence, severity


# --------------------------------------------------------------------------
# Fusion arithmetic
# --------------------------------------------------------------------------


def test_clean_image_scores_perfectly():
    result = fuse(rules(), *cnn())
    assert result.quality_score == 100.0
    assert result.quality_label == "EXCELLENT"
    assert result.issues == []


def test_both_sources_agreeing_yields_high_confidence():
    presence, severity = cnn(blur=0.9)
    result = fuse(rules(blur=0.9), presence, severity)
    assert len(result.issues) == 1
    assert result.issues[0].type == "blur"
    assert result.issues[0].confidence == pytest.approx(0.9)


def test_blend_uses_the_per_issue_weight():
    """Exposure trusts the rules more than defect does, by design."""
    presence, severity = cnn(underexposure=0.0, defect=0.0)
    result = fuse(rules(underexposure=1.0, defect=1.0), presence, severity)

    assert result.all_confidences["underexposure"] == pytest.approx(RULE_WEIGHT["underexposure"])
    assert result.all_confidences["defect"] == pytest.approx(RULE_WEIGHT["defect"])
    assert result.all_confidences["underexposure"] > result.all_confidences["defect"]


def test_silent_source_abstains_from_the_severity_vote():
    """The central design point of the blend.

    When the rules see nothing, their severity is 0 because they detected
    nothing — not because they judged the problem mild. A plain weighted mean
    would read that 0 as a vote for "not severe" and halve the reported
    severity. Weighting each source's severity by its own confidence lets a
    silent source abstain.
    """
    presence, severity = cnn(defect=(0.9, 0.8))
    result = fuse(rules(), presence, severity)  # rules silent on everything

    assert result.all_severities["defect"] == pytest.approx(0.8), (
        "severity must come entirely from the source that actually detected it"
    )


def test_disagreement_lowers_confidence_below_either_source():
    presence, severity = cnn(blur=0.0)
    only_rules = fuse(rules(blur=1.0), presence, severity)
    assert 0.0 < only_rules.all_confidences["blur"] < 1.0


def test_sub_threshold_detections_do_not_affect_the_score():
    """Six faint signals must not outscore one real problem.

    The score is built only from reported issues; letting every weak
    below-threshold confidence contribute would make a clean image with diffuse
    noise score worse than one with a genuine defect.
    """
    faint = {issue: 0.2 for issue in ISSUE_TYPES}
    result = fuse(rules(), faint, faint)
    assert result.issues == []
    assert result.quality_score == 100.0


def test_score_and_label_agree():
    presence, severity = cnn(corruption=(0.95, 0.95))
    result = fuse(rules(corruption=(0.95, 0.95)), presence, severity)
    assert result.quality_label == quality_label(result.quality_score)
    assert result.quality_score < 50


def test_issues_are_ordered_by_impact():
    presence, severity = cnn(blur=(0.9, 0.9), noise=(0.5, 0.2))
    result = fuse(rules(blur=(0.9, 0.9), noise=(0.5, 0.2)), presence, severity)
    assert [issue.type for issue in result.issues] == ["blur", "noise"]


def test_rules_only_mode_still_produces_a_verdict():
    """The service must work before any model has been trained."""
    result = fuse(rules(underexposure=(0.9, 0.7)), cnn_presence=None)
    assert result.issues and result.issues[0].type == "underexposure"
    assert result.quality_score < 100


def test_evidence_is_carried_through_to_the_issue():
    result = fuse(rules(blur=0.9), *cnn(blur=0.9))
    assert result.issues[0].evidence == ["blur evidence"]


def test_summary_is_human_readable():
    assert "No quality issues" in summarise(fuse(rules(), *cnn()))
    assert "blur" in summarise(fuse(rules(blur=0.9), *cnn(blur=0.9)))


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def test_temperature_scaling_corrects_overconfidence():
    """An overconfident model should be assigned a temperature above 1."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 2000).astype(np.float64)
    # Logits that point the right way but far too strongly: the sign is correct
    # ~80% of the time yet the magnitude claims near-certainty.
    logits = np.where(labels > 0.5, 6.0, -6.0)
    flip = rng.random(2000) < 0.2
    logits[flip] *= -1

    temperature = fit_temperature(logits, labels)
    assert temperature > 1.0, "overconfident logits must be softened"

    before = expected_calibration_error(1 / (1 + np.exp(-logits)), labels)
    after = expected_calibration_error(1 / (1 + np.exp(-logits / temperature)), labels)
    assert after < before


def test_calibration_preserves_ranking():
    """Temperature scaling is monotonic, so AUC cannot change.

    This is what makes it safe: it moves probabilities onto an honest scale
    without touching which images the model considers worse than which.
    """
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 3, 500)
    scaled = logits / 2.5
    assert np.array_equal(logits.argsort(), scaled.argsort())


def test_degenerate_input_falls_back_to_identity():
    """A single-class split gives no calibration signal."""
    assert fit_temperature(np.random.normal(size=100), np.zeros(100)) == 1.0
    assert fit_temperature(np.random.normal(size=100), np.ones(100)) == 1.0


def test_calibration_round_trips_through_json(tmp_path):
    rng = np.random.default_rng(2)
    logits = rng.normal(0, 2, (400, len(ISSUE_TYPES)))
    labels = (rng.random((400, len(ISSUE_TYPES))) < 0.3).astype(np.float64)

    calibration = fit_calibration(logits, labels)
    path = tmp_path / "calibration.json"
    calibration.to_json(path)
    restored = Calibration.from_json(path)

    assert restored.temperatures == calibration.temperatures
    assert np.allclose(restored.apply(logits), calibration.apply(logits))


def test_calibration_never_makes_ece_worse():
    """The step exists to reduce calibration error, so it must never raise it.

    Temperature is fitted by minimising NLL, which is a different objective from
    ECE; on a small validation split the two can disagree. Any issue whose
    fitted temperature would worsen ECE is left uncalibrated instead.
    """
    rng = np.random.default_rng(7)
    # Deliberately uninformative logits, the regime where an NLL fit is most
    # likely to pick a temperature that does not help.
    logits = rng.normal(0, 0.3, (60, len(ISSUE_TYPES)))
    labels = (rng.random((60, len(ISSUE_TYPES))) < 0.5).astype(np.float64)

    calibration = fit_calibration(logits, labels)
    for issue in ISSUE_TYPES:
        assert calibration.ece_after[issue] <= calibration.ece_before[issue] + 1e-9, issue
        if calibration.ece_after[issue] == calibration.ece_before[issue]:
            assert calibration.temperatures[issue] == 1.0


def test_identity_calibration_is_a_plain_sigmoid():
    logits = np.array([[-2.0, 0.0, 2.0, 1.0, -1.0, 0.5]])
    expected = 1 / (1 + np.exp(-logits))
    assert np.allclose(Calibration.identity().apply(logits), expected)


# --------------------------------------------------------------------------
# MC dropout
# --------------------------------------------------------------------------


def test_mc_dropout_leaves_batchnorm_alone():
    """Reactivating BatchNorm at batch size 1 normalises a sample by itself.

    ``enable_dropout`` must switch dropout layers only; calling ``model.train()``
    would switch both and silently corrupt every prediction.
    """
    from focal_ml.features import FEATURE_NAMES
    from focal_ml.model.architecture import FocalNet, ModelConfig
    from focal_ml.model.calibration import enable_dropout

    model = FocalNet(ModelConfig(n_features=len(FEATURE_NAMES), pretrained=False)).eval()
    enable_dropout(model)

    batchnorms = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    assert batchnorms, "this test assumes the feature encoder uses BatchNorm"
    assert all(not m.training for m in batchnorms), "BatchNorm must stay in eval mode"
    assert any(m.training for m in model.modules() if isinstance(m, torch.nn.Dropout))


def test_mc_dropout_reports_spread_across_passes():
    from focal_ml.features import FEATURE_NAMES
    from focal_ml.model.architecture import FocalNet, ModelConfig
    from focal_ml.model.calibration import mc_dropout_uncertainty

    torch.manual_seed(0)
    model = FocalNet(ModelConfig(n_features=len(FEATURE_NAMES), pretrained=False)).eval()
    estimates = mc_dropout_uncertainty(
        model, torch.randn(1, 3, 224, 224), torch.randn(1, len(FEATURE_NAMES)), passes=8
    )

    assert len(estimates) == len(ISSUE_TYPES)
    assert all(0.0 <= e.mean <= 1.0 and e.std >= 0.0 for e in estimates)
    assert any(e.std > 0 for e in estimates), "dropout must actually vary the output"
