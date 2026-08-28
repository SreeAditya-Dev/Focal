"""Assemble the full classical feature vector.

``FEATURE_NAMES`` is the ordered contract for everything downstream: the rule
layer indexes it by name, the learned baseline consumes it as a fixed-length
vector, and the API returns a readable subset of it as the `stats` object. Adding
a feature means appending to this tuple — never reordering it — or previously
trained models silently read the wrong columns.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from focal_ml.constants import CANONICAL_LONG_SIDE
from focal_ml.features.artifacts import artifact_features
from focal_ml.features.defects import defect_features
from focal_ml.features.exposure import exposure_features
from focal_ml.features.noise import noise_features
from focal_ml.features.saturation import saturation_features
from focal_ml.features.sharpness import sharpness_features
from focal_ml.features.texture import texture_features
from focal_ml.utils import imread_bgr, resize_long_side, to_gray

FEATURE_NAMES: tuple[str, ...] = (
    # sharpness / blur
    "sharpness_laplacian_var",
    "sharpness_ratio",
    "sharpness_tenengrad",
    "edge_density",
    "hf_energy_ratio",
    "tile_sharpness_p10",
    "tile_sharpness_median",
    "tile_sharpness_uniformity",
    "tile_sharpness_count",
    # exposure / contrast
    "brightness_mean",
    "brightness_std",
    "brightness_p01",
    "brightness_p05",
    "brightness_median",
    "brightness_p95",
    "brightness_p99",
    "shadow_clip_fraction",
    "highlight_clip_fraction",
    "contrast_rms",
    "dynamic_range",
    "histogram_entropy",
    # noise
    "noise_sigma_immerkaer",
    "noise_sigma_flat",
    "noise_sigma_chroma",
    "noise_sigma_median_residual",
    "noise_impulse_ratio",
    "noise_texture_ratio",
    # texture
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
    "lbp_uniformity",
    # colour
    "saturation_mean",
    "saturation_std",
    "colorfulness",
    "grey_pixel_fraction",
    "channel_edge_correlation",
    # structural corruption
    "blockiness",
    "flat_block_fraction",
    "block_mean_jump",
    "row_discontinuity",
    "col_discontinuity",
    "largest_uniform_region",
    "byte_entropy",
    # localised defects
    "radial_falloff",
    "linear_structure",
    "local_contrast_spread",
)

N_FEATURES = len(FEATURE_NAMES)

#: The subset surfaced in the API response — the measurements a human can read
#: and act on ("too dark", "not sharp"), rather than the full vector.
REPORTED_FEATURES: tuple[str, ...] = (
    "sharpness_laplacian_var",
    "hf_energy_ratio",
    "brightness_mean",
    "brightness_std",
    "contrast_rms",
    "shadow_clip_fraction",
    "highlight_clip_fraction",
    "noise_sigma_flat",
    "saturation_mean",
    "colorfulness",
    "edge_density",
    "blockiness",
)


def extract_features(
    image_bgr: np.ndarray,
    raw_bytes: bytes | None = None,
    *,
    already_canonical: bool = False,
) -> dict[str, float]:
    """Compute every classical feature for one image.

    The image is normalised to the canonical long side first. Absolute-scale
    measurements — Laplacian variance, noise sigma, blockiness — all depend on
    resolution, so a 4000px upload and a 768px training image must be reduced to
    the same scale before their numbers can be compared.

    Args:
        image_bgr: decoded BGR uint8 image.
        raw_bytes: the original encoded file, if available. Only ``byte_entropy``
            uses it; it is 0.0 when absent.
        already_canonical: skip the resize when the caller has done it.
    """
    if not already_canonical:
        image_bgr = resize_long_side(image_bgr, CANONICAL_LONG_SIDE)
    gray = to_gray(image_bgr)

    features: dict[str, float] = {}
    features.update(sharpness_features(gray))
    features.update(exposure_features(image_bgr, gray))
    features.update(noise_features(image_bgr, gray))
    features.update(texture_features(gray))
    features.update(saturation_features(image_bgr))
    features.update(artifact_features(gray, raw_bytes))
    features.update(defect_features(gray))

    # Two things happen at this boundary.
    #
    # NaN and infinity are collapsed to 0, because either would propagate
    # silently into the model input and poison a whole training run.
    #
    # Values are then rounded through float32. Several OpenCV primitives use
    # threaded reductions whose summation order depends on thread-pool state,
    # so a few features are reproducible only to about 1e-10 relative rather
    # than bit-exactly. float32 is the dtype every consumer uses anyway, and its
    # epsilon (1.2e-7) is three orders of magnitude coarser than that drift, so
    # quantising here makes the extractor deterministic in the precision that
    # actually reaches the model.
    return {
        name: float(np.float32(np.nan_to_num(features.get(name, 0.0), nan=0.0, posinf=0.0, neginf=0.0)))
        for name in FEATURE_NAMES
    }


def features_to_vector(features: dict[str, float]) -> np.ndarray:
    """Flatten a feature dict into the canonical fixed-length ordering."""
    return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)


def extract_from_path(path: str | Path) -> dict[str, float] | None:
    """Read and measure an image file; ``None`` if it cannot be decoded."""
    path = Path(path)
    image = imread_bgr(path)
    if image is None:
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        raw = None
    return extract_features(image, raw)


def benchmark(image_bgr: np.ndarray, runs: int = 10) -> dict[str, float]:
    """Per-group timings, for checking the API's synchronous inference budget."""
    from focal_ml.features import artifacts, defects, exposure, noise, saturation, sharpness, texture

    image_bgr = resize_long_side(image_bgr, CANONICAL_LONG_SIDE)
    gray = to_gray(image_bgr)
    groups = {
        "sharpness": lambda: sharpness.sharpness_features(gray),
        "exposure": lambda: exposure.exposure_features(image_bgr, gray),
        "noise": lambda: noise.noise_features(image_bgr, gray),
        "texture": lambda: texture.texture_features(gray),
        "saturation": lambda: saturation.saturation_features(image_bgr),
        "artifacts": lambda: artifacts.artifact_features(gray, None),
        "defects": lambda: defects.defect_features(gray),
    }

    timings: dict[str, float] = {}
    for name, function in groups.items():
        start = time.perf_counter()
        for _ in range(runs):
            function()
        timings[name] = (time.perf_counter() - start) / runs * 1000.0
    timings["total"] = sum(timings.values())
    return timings
