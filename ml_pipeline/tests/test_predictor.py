"""End-to-end tests for FocalPredictor — the interface the backend depends on.

These run against an untrained model on purpose. A randomly initialised network
predicts nonsense, but every contract tested here is about the *shape* of the
pipeline rather than its accuracy: that bad input is rejected cleanly, that the
service still works with no checkpoint, that a heatmap never takes down an
analysis. Accuracy is Phase 5's job and needs a trained model to mean anything.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from focal_ml.constants import ISSUE_TYPES
from focal_ml.features import REPORTED_FEATURES
from focal_ml.inference import FocalPredictor, ImageDecodeError


@pytest.fixture(scope="module")
def rules_predictor() -> FocalPredictor:
    """No checkpoint — the degraded path the API boots with."""
    return FocalPredictor()


@pytest.fixture(scope="module")
def trained_predictor(tmp_path_factory) -> FocalPredictor:
    """A predictor over an untrained-but-valid checkpoint."""
    from focal_ml.features import FEATURE_NAMES
    from focal_ml.model.architecture import FocalNet, ModelConfig

    torch.manual_seed(0)
    config = ModelConfig(n_features=len(FEATURE_NAMES), pretrained=False)
    model = FocalNet(config)

    path = tmp_path_factory.mktemp("models") / "test_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": config.to_dict(),
            "issue_types": list(ISSUE_TYPES),
            "feature_names": list(FEATURE_NAMES),
            "version": "test_v1",
        },
        path,
    )
    return FocalPredictor(model_path=path)


@pytest.fixture(scope="module")
def photo(clean_images) -> np.ndarray:
    return clean_images[0]


def encode(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    return buffer.tobytes()


# --------------------------------------------------------------------------
# Decoding and rejection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        # Explicit ids: pytest derives one from the value otherwise, and a
        # multi-kilobyte binary payload produces a test id long enough to
        # overflow the environment variable pytest passes it through on Windows.
        pytest.param(b"", id="empty"),
        pytest.param(b"this is prose, not an image" * 100, id="text"),
        pytest.param(b"\x89PNG\r\n\x1a\n" + b"\x00" * 512, id="png-header-only"),
        pytest.param(bytes(range(256)) * 100, id="random-bytes"),
    ],
)
def test_undecodable_input_raises_a_typed_error(rules_predictor, payload):
    """The backend maps this to a 422, so it must be its own exception type."""
    with pytest.raises(ImageDecodeError):
        rules_predictor.decode(payload)


def test_truncated_jpeg_is_rejected(rules_predictor, photo):
    data = encode(photo)
    with pytest.raises(ImageDecodeError):
        rules_predictor.decode(data[: len(data) // 4])


def test_absurdly_small_image_is_rejected(rules_predictor):
    with pytest.raises(ImageDecodeError):
        rules_predictor.decode(encode(np.full((4, 4, 3), 128, dtype=np.uint8)))


def test_visibly_corrupted_image_is_analysed_not_rejected(rules_predictor, photo):
    """The distinction the whole corruption class rests on.

    An image with block artifacts or a torn scanline decodes perfectly well. It
    is a normal analysis that should report a `corruption` issue — not a
    rejected upload.
    """
    from dataset.degradations import apply_degradation

    damaged = apply_degradation(photo, "corruption", 3, np.random.default_rng(0), method="row_tear")
    result = rules_predictor.analyse(encode(damaged.image))
    assert result.quality_score <= 100
    assert result.quality_label in ("EXCELLENT", "ACCEPTABLE", "POOR", "UNUSABLE")


# --------------------------------------------------------------------------
# Rules-only operation
# --------------------------------------------------------------------------


def test_service_works_without_a_checkpoint(rules_predictor, photo):
    """Booting before a model exists must produce answers, not errors."""
    assert not rules_predictor.model_loaded
    result = rules_predictor.analyse(photo, include_heatmap=True)

    assert result.model_loaded is False
    assert result.heatmap_base64 is None, "no model means no Grad-CAM"
    assert 0 <= result.quality_score <= 100


def test_clean_photo_scores_well_on_rules_alone(rules_predictor, clean_images):
    for image in clean_images:
        result = rules_predictor.analyse(image, include_heatmap=False)
        assert result.quality_score >= 85, f"clean image scored {result.quality_score}"
        assert result.quality_label == "EXCELLENT"


def test_degraded_photo_scores_worse_than_clean(rules_predictor, photo):
    from dataset.degradations import apply_degradation

    clean = rules_predictor.analyse(photo, include_heatmap=False).quality_score
    dark = apply_degradation(photo, "underexposure", 3, np.random.default_rng(0), method="gamma_gain")
    assert rules_predictor.analyse(dark.image, include_heatmap=False).quality_score < clean


# --------------------------------------------------------------------------
# Result contract
# --------------------------------------------------------------------------


def test_result_serialises_to_the_api_shape(trained_predictor, photo):
    payload = trained_predictor.analyse(photo).to_dict()

    for key in ("quality_score", "quality_label", "issues", "stats", "summary",
                "model_version", "model_loaded", "width", "height", "processing_time_ms"):
        assert key in payload, key

    assert set(payload["stats"]) == set(REPORTED_FEATURES)
    for issue in payload["issues"]:
        assert set(issue) >= {"type", "severity", "confidence", "severity_score", "evidence"}
        assert issue["type"] in ISSUE_TYPES
        assert issue["severity"] in ("low", "medium", "high")
        assert 0.0 <= issue["confidence"] <= 1.0


def test_reported_dimensions_are_the_originals(trained_predictor, photo):
    """Users are shown the size they uploaded, not the internal working size."""
    large = cv2.resize(photo, (2400, 1600))
    result = trained_predictor.analyse(large, include_heatmap=False)
    assert (result.width, result.height) == (2400, 1600)


def test_timings_are_broken_down(trained_predictor, photo):
    result = trained_predictor.analyse(photo, include_heatmap=False)
    assert {"resize", "features", "rules", "model", "fusion"} <= set(result.timings_ms)
    assert result.processing_time_ms > 0


# --------------------------------------------------------------------------
# Grad-CAM and uncertainty
# --------------------------------------------------------------------------


def test_heatmap_is_produced_for_a_detected_issue(trained_predictor, photo):
    import base64

    from dataset.degradations import apply_degradation

    blurred = apply_degradation(photo, "blur", 3, np.random.default_rng(0), method="gaussian_blur")
    result = trained_predictor.analyse(blurred.image, include_heatmap=True)

    if not result.issues:
        pytest.skip("untrained model detected nothing; heatmap path not exercised")

    assert result.heatmap_base64
    assert result.heatmap_issue in ISSUE_TYPES
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(result.heatmap_base64), np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded is not None, "heatmap must be a valid PNG"


def test_gradcam_hooks_are_not_leaked(trained_predictor, photo):
    """The predictor keeps one GradCAM; re-registering per request would leak.

    Hooks accumulate silently — the model keeps working, just slower each call,
    which is exactly the kind of leak that only shows up under sustained load.
    """
    from dataset.degradations import apply_degradation

    blurred = apply_degradation(photo, "blur", 3, np.random.default_rng(0), method="gaussian_blur").image
    for _ in range(3):
        trained_predictor.analyse(blurred, include_heatmap=True)

    layer = trained_predictor.model.features[-1]
    assert len(layer._forward_hooks) <= 1
    assert len(layer._backward_hooks) + len(getattr(layer, "_backward_pre_hooks", {})) <= 2


def test_uncertainty_is_optional_and_well_formed(trained_predictor, photo):
    without = trained_predictor.analyse(photo, include_heatmap=False)
    assert without.uncertainty is None

    with_uncertainty = trained_predictor.analyse(
        photo, include_heatmap=False, uncertainty=True, uncertainty_passes=5
    )
    assert with_uncertainty.uncertainty is not None
    assert len(with_uncertainty.uncertainty) == len(ISSUE_TYPES)
    for entry in with_uncertainty.uncertainty:
        assert set(entry) == {"issue", "mean", "std", "flagged"}
        assert entry["std"] >= 0.0


def test_model_stays_in_eval_mode_after_gradcam_and_dropout(trained_predictor, photo):
    """Both borrow the model's mode; leaving it switched would corrupt every
    subsequent request served by the same process."""
    trained_predictor.analyse(photo, include_heatmap=True, uncertainty=True, uncertainty_passes=3)
    assert not trained_predictor.model.training
    assert all(not m.training for m in trained_predictor.model.modules() if isinstance(m, torch.nn.BatchNorm1d))


# --------------------------------------------------------------------------
# Loading contracts
# --------------------------------------------------------------------------


def test_feature_set_mismatch_is_refused(tmp_path):
    """A checkpoint trained on different features must not load silently.

    The feature encoder indexes its input positionally, so a mismatch would run
    without error and read every value from the wrong column.
    """
    from focal_ml.model.architecture import FocalNet, ModelConfig

    config = ModelConfig(n_features=3, pretrained=False)
    model = FocalNet(config)
    path = tmp_path / "stale.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": config.to_dict(),
            "feature_names": ["only", "three", "features"],
            "version": "stale",
        },
        path,
    )

    with pytest.raises(ValueError, match="different feature set"):
        FocalPredictor(model_path=path)


def test_missing_checkpoint_path_degrades_quietly(tmp_path):
    predictor = FocalPredictor(model_path=tmp_path / "does_not_exist.pt")
    assert not predictor.model_loaded
    assert predictor.health()["model_version"] == "rules-only"


def test_warmup_runs(trained_predictor):
    assert trained_predictor.warmup() > 0
