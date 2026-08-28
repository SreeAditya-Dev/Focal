"""Directional response tests for the classical features.

These check the claim each feature is built on: that it moves the expected way
when the degradation it targets is applied, and does so consistently across
independent scenes. A feature that fails here is measuring something other than
what its name says, and would poison both the rule layer and the model input.

Absolute values are deliberately not asserted — they are content-dependent, and
pinning them would make the suite a change-detector rather than a correctness
check. Every assertion is about direction or ordering.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataset.degradations import apply_degradation
from focal_ml.constants import ISSUE_TYPES
from focal_ml.features import FEATURE_NAMES, extract_features, features_to_vector

SEVERITIES = (1, 2, 3)


def measure(image: np.ndarray) -> dict[str, float]:
    return extract_features(image, already_canonical=True)


def degrade(image: np.ndarray, issue: str, bucket: int, method: str, seed: int = 11) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    return measure(apply_degradation(image, issue, bucket, rng, method=method).image)


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------


def test_feature_vector_contract(clean_image):
    features = measure(clean_image)
    assert set(features) == set(FEATURE_NAMES)

    vector = features_to_vector(features)
    assert vector.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(vector).all(), "features must never emit NaN or infinity"


def test_extraction_is_deterministic(clean_image):
    """Repeated extraction of the same image must give the same vector.

    Several OpenCV primitives reduce with threaded, order-dependent summation,
    so raw float64 results wander in their last few digits between calls.
    ``extract_features`` quantises to float32 for exactly this reason, and this
    test guards that guarantee — cached features have to compare equal to
    recomputed ones, or every downstream cache is silently wrong.

    Repeated more than twice because the drift is intermittent: it depends on
    thread-pool state, and two consecutive calls often agree by chance.
    """
    runs = [measure(clean_image) for _ in range(8)]
    for name in FEATURE_NAMES:
        distinct = {run[name] for run in runs}
        assert len(distinct) == 1, f"{name} varied across runs: {sorted(distinct)}"


def test_resolution_normalisation(clean_image):
    """A resized copy of an image must measure almost the same.

    Absolute-scale features would otherwise report a 4000px upload and a 768px
    training image as entirely different, which is exactly the train/serve skew
    the canonical resize exists to prevent.
    """
    import cv2

    large = cv2.resize(clean_image, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    base = extract_features(clean_image)
    scaled = extract_features(large)

    for name in ("brightness_mean", "contrast_rms", "hf_energy_ratio", "saturation_mean"):
        assert base[name] == pytest.approx(scaled[name], rel=0.25), name


# --------------------------------------------------------------------------
# Blur
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["gaussian_blur", "motion_blur"])
def test_blur_monotonically_reduces_high_frequency_energy(clean_images, method):
    for image in clean_images:
        baseline = measure(image)["hf_energy_ratio"]
        responses = [degrade(image, "blur", b, method)["hf_energy_ratio"] for b in SEVERITIES]
        assert baseline > responses[0], f"{method}: mild blur must reduce HF energy"
        assert responses == sorted(responses, reverse=True), f"{method}: must fall with severity"


@pytest.mark.parametrize("method", ["gaussian_blur", "motion_blur"])
def test_blur_reduces_sharpness_and_edges(clean_images, method):
    for image in clean_images:
        baseline = measure(image)
        blurred = degrade(image, "blur", 3, method)
        assert blurred["sharpness_ratio"] < baseline["sharpness_ratio"]
        assert blurred["edge_density"] < baseline["edge_density"]
        assert blurred["sharpness_laplacian_var"] < baseline["sharpness_laplacian_var"]


# --------------------------------------------------------------------------
# Exposure
# --------------------------------------------------------------------------


def test_underexposure_darkens_and_crushes_shadows(clean_images):
    for image in clean_images:
        baseline = measure(image)
        for bucket in SEVERITIES:
            dark = degrade(image, "underexposure", bucket, "gamma_gain")
            assert dark["brightness_mean"] < baseline["brightness_mean"]
        severe = degrade(image, "underexposure", 3, "gamma_gain")
        assert severe["shadow_clip_fraction"] > baseline["shadow_clip_fraction"]


def test_overexposure_brightens_and_blows_highlights(clean_images):
    for image in clean_images:
        baseline = measure(image)
        for bucket in SEVERITIES:
            bright = degrade(image, "overexposure", bucket, "gamma_gain")
            assert bright["brightness_mean"] > baseline["brightness_mean"]
        severe = degrade(image, "overexposure", 3, "gamma_gain")
        assert severe["highlight_clip_fraction"] > baseline["highlight_clip_fraction"]


def test_exposure_severity_ordering(clean_images):
    for image in clean_images:
        means = [degrade(image, "underexposure", b, "gamma_gain")["brightness_mean"] for b in SEVERITIES]
        assert means == sorted(means, reverse=True)


# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["gaussian_noise", "speckle_noise"])
def test_additive_noise_raises_flat_region_sigma(clean_images, method):
    for image in clean_images:
        baseline = measure(image)["noise_sigma_flat"]
        responses = [degrade(image, "noise", b, method)["noise_sigma_flat"] for b in SEVERITIES]
        assert responses[0] > baseline
        assert responses == sorted(responses), f"{method}: sigma must rise with severity"


def test_impulse_noise_is_caught_by_impulse_ratio(clean_images):
    """Salt-and-pepper barely moves a variance, so it needs its own detector."""
    for image in clean_images:
        baseline = measure(image)["noise_impulse_ratio"]
        responses = [degrade(image, "noise", b, "salt_pepper_noise")["noise_impulse_ratio"] for b in SEVERITIES]
        assert responses[0] > baseline
        assert responses == sorted(responses)


# --------------------------------------------------------------------------
# Corruption
# --------------------------------------------------------------------------


def test_recompression_raises_blockiness(clean_images):
    for image in clean_images:
        baseline = measure(image)["blockiness"]
        severe = degrade(image, "corruption", 3, "jpeg_recompression")["blockiness"]
        assert severe > baseline


def test_channel_desync_breaks_channel_correlation(clean_images):
    """The signal that only exists in colour — luma conversion destroys it."""
    for image in clean_images:
        baseline = measure(image)["channel_edge_correlation"]
        severe = degrade(image, "corruption", 3, "channel_desync")["channel_edge_correlation"]
        assert severe < baseline


def test_row_tear_raises_row_discontinuity(clean_images):
    for image in clean_images:
        baseline = measure(image)["row_discontinuity"]
        severe = degrade(image, "corruption", 3, "row_tear")["row_discontinuity"]
        assert severe > baseline


def test_occlusion_creates_uniform_region(clean_images):
    for image in clean_images:
        baseline = measure(image)["largest_uniform_region"]
        severe = degrade(image, "corruption", 3, "occlusion")["largest_uniform_region"]
        assert severe > baseline


# --------------------------------------------------------------------------
# Defects
# --------------------------------------------------------------------------


def test_vignette_reduces_radial_falloff(clean_images):
    for image in clean_images:
        baseline = measure(image)["radial_falloff"]
        responses = [degrade(image, "defect", b, "vignette")["radial_falloff"] for b in SEVERITIES]
        assert responses[0] < baseline
        assert responses == sorted(responses, reverse=True)


def test_smudge_hides_from_global_sharpness_unlike_uniform_blur(clean_images):
    """The case that justifies the hybrid architecture.

    A lens smudge blurs one region and leaves the rest of the frame sharp, so
    whole-image sharpness barely moves and a global threshold cannot catch it.
    Uniform blur of comparable strength collapses the same metric.

    The comparison against blur is the point. An earlier version of this test
    only checked that a smudge moved *local* statistics more than global ones,
    which passed — but so does uniform blur, so it demonstrated nothing. Local
    sharpness spread reacts to both and identifies neither; what actually
    distinguishes a smudge is that global sharpness *survives* it.
    """
    smudge_retention, blur_retention = [], []

    for image in clean_images:
        baseline = measure(image)["sharpness_ratio"]
        smudge_retention.append(degrade(image, "defect", 3, "lens_smudge")["sharpness_ratio"] / baseline)
        blur_retention.append(degrade(image, "blur", 3, "gaussian_blur")["sharpness_ratio"] / baseline)

    assert np.mean(smudge_retention) > 3 * np.mean(blur_retention), (
        "a smudge must leave global sharpness far more intact than uniform blur does; "
        f"smudge retained {np.mean(smudge_retention):.2f}x, blur {np.mean(blur_retention):.2f}x"
    )


def test_local_sharpness_spread_cannot_separate_smudge_from_blur(clean_images):
    """Pins a known limitation, so it cannot be forgotten and misused.

    Tile sharpness uniformity falls for a localised smudge *and* for global
    blur, by similar amounts. This test exists to document that the feature is
    not a defect detector — if a future change makes it discriminative, this
    test fails and the defect rule should be revisited.
    """
    smudge_values, blur_values = [], []

    for image in clean_images:
        smudge_values.append(degrade(image, "defect", 3, "lens_smudge")["tile_sharpness_uniformity"])
        blur_values.append(degrade(image, "blur", 3, "gaussian_blur")["tile_sharpness_uniformity"])

    assert abs(np.mean(smudge_values) - np.mean(blur_values)) < 0.1, (
        "smudge and blur now separate on tile uniformity "
        f"({np.mean(smudge_values):.3f} vs {np.mean(blur_values):.3f}) — "
        "reconsider using it as a defect signal in fusion/rules.py"
    )


def test_dust_and_hot_pixels_raise_impulse_ratio(clean_images):
    for image in clean_images:
        baseline = measure(image)["noise_impulse_ratio"]
        severe = degrade(image, "defect", 3, "dust_and_hot_pixels")["noise_impulse_ratio"]
        assert severe > baseline
