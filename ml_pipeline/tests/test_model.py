"""Contract tests for FocalNet.

These target the failures that do not announce themselves: a checkpoint that
loads but normalises its inputs differently than it was trained with, a freeze
schedule that silently does nothing, a severity head trained against labels that
carry no information. Each of those produces a model that runs fine and predicts
badly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from focal_ml.constants import ISSUE_TYPES
from focal_ml.features import FEATURE_NAMES
from focal_ml.model.architecture import FocalNet, ModelConfig

N_FEATURES = len(FEATURE_NAMES)


def build(**overrides) -> FocalNet:
    config = ModelConfig(n_features=N_FEATURES, pretrained=False, **overrides)
    return FocalNet(config)


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.randn(4, 3, 224, 224), torch.randn(4, N_FEATURES) * 100


# --------------------------------------------------------------------------
# Shape and range contracts
# --------------------------------------------------------------------------


@pytest.mark.parametrize("use_image,use_features", [(True, True), (True, False), (False, True)])
def test_all_ablation_variants_produce_correct_shapes(batch, use_image, use_features):
    image, features = batch
    model = build(use_image=use_image, use_features=use_features)
    output = model(
        image=image if use_image else None,
        features=features if use_features else None,
    )
    assert output["presence_logits"].shape == (4, len(ISSUE_TYPES))
    assert output["severity_logits"].shape == (4, len(ISSUE_TYPES))


def test_predict_outputs_are_bounded(batch):
    image, features = batch
    model = build().eval()
    prediction = model.predict(image, features)
    for key in ("presence", "severity"):
        assert prediction[key].min() >= 0.0 and prediction[key].max() <= 1.0, key


def test_missing_required_input_is_rejected(batch):
    image, features = batch
    with pytest.raises(ValueError, match="use_features"):
        build()(image=image, features=None)
    with pytest.raises(ValueError, match="use_image"):
        build()(image=None, features=features)


def test_model_with_no_inputs_is_rejected():
    with pytest.raises(ValueError):
        build(use_image=False, use_features=False)


# --------------------------------------------------------------------------
# Feature handling
# --------------------------------------------------------------------------


def test_log_transform_is_monotonic_and_sign_preserving():
    """Ramps and the model both rely on feature ordering being preserved."""
    values = torch.tensor([-5000.0, -1.0, 0.0, 0.5, 1.0, 5000.0])
    transformed = FocalNet.transform_features(values)
    assert torch.all(transformed[1:] > transformed[:-1]), "must preserve ordering"
    assert torch.sign(transformed).tolist() == torch.sign(values).tolist()


def test_log_transform_compresses_heavy_tails():
    """Laplacian variance spans orders of magnitude; clipping fractions do not.

    Without compression the standardised vector is dominated by whichever
    feature happens to have the widest raw range.
    """
    wide = torch.tensor([1.0, 10.0, 100.0, 1000.0, 10000.0])
    ratio_before = float(wide.max() / wide.min())
    compressed = FocalNet.transform_features(wide)
    ratio_after = float(compressed.max() / compressed.min())
    assert ratio_after < ratio_before / 100


def test_feature_statistics_survive_checkpoint_round_trip():
    """The classic serving bug: weights restored with different normalisation.

    The statistics are registered buffers precisely so they cannot be left
    behind, and this asserts that they are not.
    """
    model = build()
    mean = torch.randn(N_FEATURES)
    std = torch.rand(N_FEATURES) + 0.5
    model.set_feature_stats(mean, std)

    restored = build()
    restored.load_state_dict(model.state_dict())

    assert torch.allclose(restored.feature_mean, mean)
    assert torch.allclose(restored.feature_std, std)


def test_zero_variance_feature_does_not_produce_nan(batch):
    """A constant feature would divide by zero without the clamp."""
    _, features = batch
    model = build().eval()
    model.set_feature_stats(torch.zeros(N_FEATURES), torch.zeros(N_FEATURES))
    output = model.predict(torch.randn(4, 3, 224, 224), features)
    assert torch.isfinite(output["presence"]).all()


def test_normalisation_actually_applied(batch):
    """Changing the statistics must change the output.

    Guards against the buffers being stored but never read — which would leave
    the model working on raw features spanning six orders of magnitude.
    """
    image, features = batch
    model = build().eval()

    model.set_feature_stats(torch.zeros(N_FEATURES), torch.ones(N_FEATURES))
    first = model.predict(image, features)["presence"]
    model.set_feature_stats(torch.full((N_FEATURES,), 3.0), torch.full((N_FEATURES,), 0.5))
    second = model.predict(image, features)["presence"]

    assert not torch.allclose(first, second)


# --------------------------------------------------------------------------
# Transfer-learning schedule
# --------------------------------------------------------------------------


def test_freeze_then_unfreeze_changes_trainable_count():
    model = build()
    everything, total = model.trainable_parameter_count()

    model.freeze_backbone()
    frozen, _ = model.trainable_parameter_count()
    assert frozen < everything, "freezing must actually reduce trainable parameters"

    model.unfreeze_last_blocks(2)
    thawed, _ = model.trainable_parameter_count()
    assert frozen < thawed < total, "fine-tuning must reopen some but not all of the backbone"


def test_frozen_backbone_receives_no_gradient(batch):
    """A freeze that leaves gradients flowing destroys the pretrained filters."""
    image, features = batch
    model = build()
    model.freeze_backbone()

    output = model(image=image, features=features)
    output["presence_logits"].sum().backward()

    assert all(p.grad is None for p in model.features.parameters())
    assert any(p.grad is not None for p in model.presence_head.parameters())


def test_features_only_variant_has_no_backbone():
    """The ablation must genuinely remove the image path, not just ignore it."""
    model = build(use_image=False)
    assert not hasattr(model, "features")
    _, total = model.trainable_parameter_count()
    assert total < 100_000


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------


def test_severity_loss_is_masked_to_present_issues():
    """Severity is undefined where an issue is absent.

    Its label is 0 by convention only, and regressing against those zeros is a
    far easier objective than the real one — it would dominate the gradient and
    teach the head to predict 0 everywhere.
    """
    from training.train import FocalLoss

    criterion = FocalLoss(torch.ones(len(ISSUE_TYPES)), severity_weight=1.0)
    presence = torch.zeros(2, len(ISSUE_TYPES))
    presence[0, 0] = 1.0

    output = {
        "presence_logits": torch.zeros(2, len(ISSUE_TYPES), requires_grad=True),
        "severity_logits": torch.full((2, len(ISSUE_TYPES)), 5.0, requires_grad=True),
    }
    # Absent issues carry a wildly wrong severity label. A masked loss ignores
    # them; an unmasked one would be enormous.
    target = {"presence": presence, "severity": torch.full((2, len(ISSUE_TYPES)), 0.0)}
    target["severity"][0, 0] = 1.0

    losses = criterion(output, target)
    # Only the single present issue contributes, and it predicts ~0.993 for a
    # label of 1.0, so the severity term is small.
    assert float(losses["severity"]) < 0.01


def test_loss_with_no_positives_does_not_crash():
    from training.train import FocalLoss

    criterion = FocalLoss(torch.ones(len(ISSUE_TYPES)))
    output = {
        "presence_logits": torch.zeros(2, len(ISSUE_TYPES), requires_grad=True),
        "severity_logits": torch.zeros(2, len(ISSUE_TYPES), requires_grad=True),
    }
    target = {
        "presence": torch.zeros(2, len(ISSUE_TYPES)),
        "severity": torch.zeros(2, len(ISSUE_TYPES)),
    }
    losses = criterion(output, target)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
