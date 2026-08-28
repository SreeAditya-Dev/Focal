"""Controlled, labelled image degradations.

Each function takes a clean BGR image and a requested severity bucket, applies
one physically-motivated degradation, and reports back *what it actually did*.
That last part is the point: because the pipeline owns every parameter, the
resulting labels are exact rather than annotated, which is what makes a purely
synthetic training corpus viable.

Two conventions hold throughout:

  * ``severity_score`` (continuous, 0-1) is the ground truth. ``severity_bucket``
    is always derived from it via ``severity_bucket_from_score`` so the two can
    never disagree.
  * For exposure, the score is measured from the *achieved* image brightness
    rather than the requested parameter, because the same gamma darkens a bright
    image and a dim one by different visible amounts.

Run ``python -m dataset.degradations --demo <image>`` from ``ml_pipeline/`` to
render a sample of every issue at every severity for visual inspection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from focal_ml.constants import (
    ISSUE_TYPES,
    MIN_SEVERITY_SCORE,
    SEVERITY_SCORE_RANGES,
    severity_bucket_from_score,
)
from focal_ml.utils import imread_bgr, imwrite_jpeg, jpeg_roundtrip, to_gray

Range = tuple[float, float]


@dataclass
class DegradationResult:
    """A degraded image plus the exact ground truth describing it."""

    image: np.ndarray
    issue_type: str
    severity_score: float
    method: str
    params: dict = field(default_factory=dict)

    @property
    def severity_bucket(self) -> int:
        return severity_bucket_from_score(self.severity_score)


# --------------------------------------------------------------------------
# Sampling helpers
# --------------------------------------------------------------------------


def _sample_t(rng: np.random.Generator) -> float:
    """Draw the position within a severity bucket, in [0, 1]."""
    return float(rng.random())


def _score_for(bucket: int, t: float) -> float:
    low, high = SEVERITY_SCORE_RANGES[bucket]
    return float(low + t * (high - low))


def _interp(span: Range, t: float) -> float:
    """Interpolate a parameter range at the same position ``t`` used for the
    severity score, so parameter magnitude and reported severity stay in step."""
    low, high = span
    return float(low + t * (high - low))


def _interp_int(span: Range, t: float) -> int:
    return int(round(_interp(span, t)))


def _odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


# --------------------------------------------------------------------------
# Blur
# --------------------------------------------------------------------------

_GAUSSIAN_SIGMA: dict[int, Range] = {1: (1.0, 2.0), 2: (2.0, 4.0), 3: (4.0, 7.0)}
_MOTION_LENGTH: dict[int, Range] = {1: (5, 11), 2: (11, 19), 3: (19, 29)}


def gaussian_blur(image: np.ndarray, bucket: int, rng: np.random.Generator) -> DegradationResult:
    """Defocus blur — an isotropic low-pass, the classic out-of-focus case."""
    t = _sample_t(rng)
    sigma = _interp(_GAUSSIAN_SIGMA[bucket], t)
    # ksize=(0,0) lets OpenCV derive a kernel wide enough for sigma.
    out = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return DegradationResult(out, "blur", _score_for(bucket, t), "gaussian_blur", {"sigma": round(sigma, 3)})


def motion_blur(image: np.ndarray, bucket: int, rng: np.random.Generator) -> DegradationResult:
    """Directional blur from camera shake or subject movement.

    Distinct from defocus in the frequency domain — it attenuates detail along
    one axis only — so including both teaches the model that "blurry" is not a
    single signature.
    """
    t = _sample_t(rng)
    length = max(3, _odd(_interp_int(_MOTION_LENGTH[bucket], t)))
    angle = float(rng.uniform(0.0, 180.0))

    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D((length / 2.0 - 0.5, length / 2.0 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (length, length))
    total = kernel.sum()
    kernel = kernel / total if total > 0 else kernel

    out = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)
    return DegradationResult(
        out, "blur", _score_for(bucket, t), "motion_blur",
        {"length": length, "angle": round(angle, 1)},
    )


# --------------------------------------------------------------------------
# Exposure
# --------------------------------------------------------------------------

# Target mean luma per bucket, written (least severe -> most severe) so that
# interpolating at t moves monotonically deeper into the failure.
_UNDER_TARGET: dict[int, Range] = {1: (100.0, 80.0), 2: (80.0, 50.0), 3: (50.0, 18.0)}
_OVER_TARGET: dict[int, Range] = {1: (150.0, 175.0), 2: (175.0, 205.0), 3: (205.0, 238.0)}

# Mean luma spanning "no exposure problem" -> "fully blown". Achieved brightness
# is mapped through these to produce the final severity score.
_UNDER_FULL_SPAN: Range = (120.0, 18.0)
_OVER_FULL_SPAN: Range = (135.0, 238.0)


def _solve_gamma(gray_norm: np.ndarray, target_mean: float, gain: float) -> float:
    """Bisect for the gamma that drives mean luma to ``target_mean``.

    ``mean(clip(x**g * gain))`` is monotonically decreasing in g for x in [0,1],
    so plain bisection converges quickly and needs no gradients.
    """
    target = target_mean / 255.0
    low, high = 0.05, 12.0
    for _ in range(40):
        mid = 0.5 * (low + high)
        achieved = float(np.clip(np.power(gray_norm, mid) * gain, 0.0, 1.0).mean())
        if achieved > target:
            low = mid  # still too bright, need a larger gamma
        else:
            high = mid
    return 0.5 * (low + high)


def _apply_exposure(image: np.ndarray, gamma: float, gain: float) -> np.ndarray:
    # A 256-entry LUT is exact for uint8 input and far faster than pow().
    table = np.arange(256, dtype=np.float32) / 255.0
    table = np.clip(np.power(table, gamma) * gain, 0.0, 1.0) * 255.0
    return cv2.LUT(image, table.astype(np.uint8))


def _exposure(
    image: np.ndarray, bucket: int, rng: np.random.Generator, *, over: bool
) -> DegradationResult:
    t = _sample_t(rng)
    gray = to_gray(image)
    current = float(gray.mean())
    gray_norm = gray.astype(np.float32) / 255.0

    if over:
        target = _interp(_OVER_TARGET[bucket], t)
        target = max(target, current * 1.1)  # must actually brighten
        # A gain above 1 is what clips highlights; gamma alone lifts midtones
        # without ever destroying detail, which is not what overexposure is.
        gain = _interp((1.05, 1.45), t)
        span = _OVER_FULL_SPAN
        issue = "overexposure"
    else:
        target = _interp(_UNDER_TARGET[bucket], t)
        target = min(target, current * 0.9)  # must actually darken
        gain = 1.0
        span = _UNDER_FULL_SPAN
        issue = "underexposure"

    gamma = _solve_gamma(gray_norm, target, gain)
    out = _apply_exposure(image, gamma, gain)

    # Score from what we actually produced, not what we asked for.
    achieved = float(to_gray(out).mean())
    low, high = span
    score = float(np.clip((achieved - low) / (high - low), 0.0, 1.0))
    score = max(score, MIN_SEVERITY_SCORE)

    return DegradationResult(
        out, issue, score, "gamma_gain",
        {
            "gamma": round(gamma, 4),
            "gain": round(gain, 3),
            "target_mean": round(target, 2),
            "achieved_mean": round(achieved, 2),
            "source_mean": round(current, 2),
            "requested_bucket": bucket,
        },
    )


def underexposure(image: np.ndarray, bucket: int, rng: np.random.Generator) -> DegradationResult:
    return _exposure(image, bucket, rng, over=False)


def overexposure(image: np.ndarray, bucket: int, rng: np.random.Generator) -> DegradationResult:
    return _exposure(image, bucket, rng, over=True)


# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------

_GAUSS_NOISE_VAR: dict[int, Range] = {1: (0.0008, 0.004), 2: (0.004, 0.015), 3: (0.015, 0.05)}
_SP_AMOUNT: dict[int, Range] = {1: (0.005, 0.02), 2: (0.02, 0.05), 3: (0.05, 0.12)}
_SPECKLE_VAR: dict[int, Range] = {1: (0.005, 0.02), 2: (0.02, 0.06), 3: (0.06, 0.15)}


def gaussian_noise(image: np.ndarray, bucket: int, rng: np.random.Generator) -> DegradationResult:
    """Additive sensor noise — the dominant term in low-light capture."""
    t = _sample_t(rng)
    variance = _interp(_GAUSS_NOISE_VAR[bucket], t)
    sigma = float(np.sqrt(variance))
    noise = rng.normal(0.0, sigma, image.shape).astype(np.float32)
    out = np.clip(image.astype(np.float32) / 255.0 + noise, 0.0, 1.0) * 255.0
    return DegradationResult(
        out.astype(np.uint8), "noise", _score_for(bucket, t), "gaussian_noise",
        {"variance": round(variance, 5), "sigma_norm": round(sigma, 5)},
    )


def salt_pepper_noise(image: np.ndarray, bucket: int, rng: np.random.Generator) -> DegradationResult:
    """Impulse noise — dead/stuck pixels and transmission bit errors."""
    t = _sample_t(rng)
    amount = _interp(_SP_AMOUNT[bucket], t)
    out = image.copy()
    height, width = image.shape[:2]
    draw = rng.random((height, width))
    out[draw < amount / 2.0] = 0
    out[(draw >= amount / 2.0) & (draw < amount)] = 255
    return DegradationResult(
        out, "noise", _score_for(bucket, t), "salt_pepper_noise", {"amount": round(amount, 5)}
    )


def speckle_noise(image: np.ndarray, bucket: int, rng: np.random.Generator) -> DegradationResult:
    """Multiplicative noise — scales with signal, unlike the additive case."""
    t = _sample_t(rng)
    variance = _interp(_SPECKLE_VAR[bucket], t)
    normalised = image.astype(np.float32) / 255.0
    noise = rng.normal(0.0, float(np.sqrt(variance)), image.shape).astype(np.float32)
    out = np.clip(normalised + normalised * noise, 0.0, 1.0) * 255.0
    return DegradationResult(
        out.astype(np.uint8), "noise", _score_for(bucket, t), "speckle_noise",
        {"variance": round(variance, 5)},
    )


# --------------------------------------------------------------------------
# Corruption
#
# Every method here produces a file that still *decodes cleanly* — the target is
# an image that looks broken, not a byte stream that fails to parse. Genuinely
# unreadable files are generated separately (see generate_synthetic.py) and are
# used only to exercise the API's rejection path.
# --------------------------------------------------------------------------

_JPEG_QUALITY: dict[int, Range] = {1: (40.0, 28.0), 2: (28.0, 18.0), 3: (18.0, 6.0)}
_JPEG_ROUNDS: dict[int, int] = {1: 2, 2: 2, 3: 3}
_BLOCK_COUNT: dict[int, Range] = {1: (3, 10), 2: (10, 30), 3: (30, 80)}
_OCCLUSION_FRAC: dict[int, Range] = {1: (0.05, 0.12), 2: (0.12, 0.22), 3: (0.22, 0.35)}
_TEAR_BANDS: dict[int, Range] = {1: (1, 3), 2: (3, 7), 3: (7, 14)}
_DESYNC_FRAC: dict[int, Range] = {1: (0.003, 0.01), 2: (0.01, 0.025), 3: (0.025, 0.06)}


def jpeg_recompression(image: np.ndarray, bucket: int, rng: np.random.Generator) -> DegradationResult:
    """Repeated low-quality JPEG encoding — generational loss.

    Each round compounds the 8x8 DCT blocking of the last, which is what makes
    a heavily reshared image look degraded in a way one pass does not.
    """
    t = _sample_t(rng)
    quality = max(2, _interp_int(_JPEG_QUALITY[bucket], t))
    rounds = _JPEG_ROUNDS[bucket]
    out = image
    for _ in range(rounds):
        out = jpeg_roundtrip(out, quality)
    return DegradationResult(
        out, "corruption", _score_for(bucket, t), "jpeg_recompression",
        {"quality": quality, "rounds": rounds},
    )


def block_corruption(image: np.ndarray, bucket: int, rng: np.random.Generator) -> DegradationResult:
    """Runs of destroyed 8x8 blocks, as when JPEG entropy data is damaged."""
    t = _sample_t(rng)
    n_blocks = _interp_int(_BLOCK_COUNT[bucket], t)
    out = image.copy()
    height, width = image.shape[:2]
    block = 8
    blocks_y, blocks_x = max(1, height // block), max(1, width // block)

    for _ in range(n_blocks):
        run_w = int(rng.integers(1, 9))
        run_h = int(rng.integers(1, 4))
        by = int(rng.integers(0, max(1, blocks_y - run_h)))
        bx = int(rng.integers(0, max(1, blocks_x - run_w)))
        y0, x0 = by * block, bx * block
        y1, x1 = min(height, y0 + run_h * block), min(width, x0 + run_w * block)
        region = out[y0:y1, x0:x1]
        if region.size == 0:
            continue
        mode = int(rng.integers(0, 3))
        if mode == 0:  # DC-only: the block flattens to its average colour
            out[y0:y1, x0:x1] = region.mean(axis=(0, 1)).astype(np.uint8)
        elif mode == 1:  # garbage colour from a misread coefficient
            out[y0:y1, x0:x1] = rng.integers(0, 256, size=3).astype(np.uint8)
        else:  # sign flip
            out[y0:y1, x0:x1] = (255 - region).astype(np.uint8)

    return DegradationResult(
        out, "corruption", _score_for(bucket, t), "block_corruption", {"n_blocks": n_blocks}
    )


def occlusion(image: np.ndarray, bucket: int, rng: np.random.Generator) -> DegradationResult:
    """A large region of the frame replaced by flat or random data."""
    t = _sample_t(rng)
    fraction = _interp(_OCCLUSION_FRAC[bucket], t)
    out = image.copy()
    height, width = image.shape[:2]

    aspect = float(rng.uniform(0.5, 2.0))
    area = fraction * height * width
    patch_h = int(np.clip(np.sqrt(area / aspect), 8, height))
    patch_w = int(np.clip(area / max(patch_h, 1), 8, width))
    y0 = int(rng.integers(0, max(1, height - patch_h)))
    x0 = int(rng.integers(0, max(1, width - patch_w)))

    if rng.random() < 0.5:
        fill = np.full((patch_h, patch_w, 3), int(rng.integers(0, 60)), dtype=np.uint8)
    else:
        fill = rng.integers(0, 256, size=(patch_h, patch_w, 3)).astype(np.uint8)
    out[y0:y0 + patch_h, x0:x0 + patch_w] = fill

    return DegradationResult(
        out, "corruption", _score_for(bucket, t), "occlusion",
        {"area_fraction": round(fraction, 4), "patch": [x0, y0, patch_w, patch_h]},
    )


def row_tear(image: np.ndarray, bucket: int, rng: np.random.Generator) -> DegradationResult:
    """Horizontally displaced scanline bands — a truncated-decode signature."""
    t = _sample_t(rng)
    n_bands = _interp_int(_TEAR_BANDS[bucket], t)
    out = image.copy()
    height, width = image.shape[:2]

    for _ in range(n_bands):
        band_h = int(rng.integers(4, max(5, height // 25)))
        y0 = int(rng.integers(0, max(1, height - band_h)))
        shift = int(rng.integers(-width // 6, width // 6))
        out[y0:y0 + band_h] = np.roll(out[y0:y0 + band_h], shift, axis=1)

    return DegradationResult(
        out, "corruption", _score_for(bucket, t), "row_tear", {"n_bands": n_bands}
    )


def channel_desync(image: np.ndarray, bucket: int, rng: np.random.Generator) -> DegradationResult:
    """One colour plane offset from the others — a chroma-decode failure."""
    t = _sample_t(rng)
    fraction = _interp(_DESYNC_FRAC[bucket], t)
    out = image.copy()
    width = image.shape[1]
    shift = max(1, int(round(fraction * width)))
    channel = int(rng.integers(0, 3))
    axis = 1 if rng.random() < 0.7 else 0
    out[:, :, channel] = np.roll(out[:, :, channel], shift, axis=axis)
    return DegradationResult(
        out, "corruption", _score_for(bucket, t), "channel_desync",
        {"shift_px": shift, "channel": channel, "axis": axis},
    )


# --------------------------------------------------------------------------
# Localised defects
# --------------------------------------------------------------------------

_SCRATCH_COUNT: dict[int, Range] = {1: (1, 3), 2: (3, 7), 3: (7, 15)}
_SCRATCH_ALPHA: dict[int, Range] = {1: (0.30, 0.50), 2: (0.50, 0.75), 3: (0.75, 1.00)}
_DUST_COUNT: dict[int, Range] = {1: (5, 20), 2: (20, 60), 3: (60, 150)}
_SMUDGE_RADIUS: dict[int, Range] = {1: (0.10, 0.18), 2: (0.18, 0.28), 3: (0.28, 0.40)}
_VIGNETTE_STRENGTH: dict[int, Range] = {1: (0.25, 0.40), 2: (0.40, 0.60), 3: (0.60, 0.80)}


def scratches(image: np.ndarray, bucket: int, rng: np.random.Generator) -> DegradationResult:
    """Thin bright or dark strokes — emulsion scratches, lens hairline damage."""
    t = _sample_t(rng)
    count = _interp_int(_SCRATCH_COUNT[bucket], t)
    alpha = _interp(_SCRATCH_ALPHA[bucket], t)
    overlay = image.copy()
    height, width = image.shape[:2]

    for _ in range(count):
        # A jittered polyline rather than a straight line: real scratches wander.
        points = int(rng.integers(3, 7))
        x = np.sort(rng.integers(0, width, size=points))
        y = rng.integers(0, height, size=points)
        y = np.clip(y[0] + np.cumsum(rng.integers(-height // 12, height // 12, size=points)), 0, height - 1)
        polyline = np.stack([x, y], axis=1).astype(np.int32)
        colour = 245 if rng.random() < 0.6 else 12
        cv2.polylines(
            overlay, [polyline], isClosed=False,
            color=(colour, colour, colour),
            thickness=int(rng.integers(1, 4)), lineType=cv2.LINE_AA,
        )

    out = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0)
    return DegradationResult(
        out, "defect", _score_for(bucket, t), "scratches",
        {"count": count, "alpha": round(alpha, 3)},
    )


def dust_and_hot_pixels(image: np.ndarray, bucket: int, rng: np.random.Generator) -> DegradationResult:
    """Sensor dust (soft dark blobs) plus hot pixels (isolated bright points)."""
    t = _sample_t(rng)
    count = _interp_int(_DUST_COUNT[bucket], t)
    out = image.copy()
    height, width = image.shape[:2]

    dust_layer = np.zeros((height, width), dtype=np.float32)
    for _ in range(count):
        cx, cy = int(rng.integers(0, width)), int(rng.integers(0, height))
        radius = int(rng.integers(2, 9))
        cv2.circle(dust_layer, (cx, cy), radius, float(rng.uniform(0.35, 0.9)), -1)
    # Dust sits off the focal plane, so its shadow is soft rather than hard-edged.
    dust_layer = cv2.GaussianBlur(dust_layer, (0, 0), sigmaX=2.0)
    out = (out.astype(np.float32) * (1.0 - dust_layer[..., None])).clip(0, 255).astype(np.uint8)

    n_hot = max(1, count // 2)
    ys = rng.integers(0, height, size=n_hot)
    xs = rng.integers(0, width, size=n_hot)
    out[ys, xs] = np.array([255, 255, 255], dtype=np.uint8)

    return DegradationResult(
        out, "defect", _score_for(bucket, t), "dust_and_hot_pixels",
        {"n_dust": count, "n_hot_pixels": int(n_hot)},
    )


def lens_smudge(image: np.ndarray, bucket: int, rng: np.random.Generator) -> DegradationResult:
    """A soft elliptical region of local blur and reduced contrast.

    This is the interesting defect case: the frame is globally sharp, so a
    global sharpness metric reports nothing wrong. Catching it requires either
    local analysis or the CNN — which is exactly the argument for the hybrid.
    """
    t = _sample_t(rng)
    radius_frac = _interp(_SMUDGE_RADIUS[bucket], t)
    height, width = image.shape[:2]
    radius = int(radius_frac * min(height, width))

    cx = int(rng.integers(radius, max(radius + 1, width - radius)))
    cy = int(rng.integers(radius, max(radius + 1, height - radius)))
    axes = (radius, int(radius * float(rng.uniform(0.6, 1.0))))

    mask = np.zeros((height, width), dtype=np.float32)
    cv2.ellipse(mask, (cx, cy), axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius * 0.35)[..., None]

    smudged = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0 + 5.0 * t)
    # Smudges also scatter light, washing out local contrast.
    smudged = cv2.addWeighted(smudged, 0.85, np.full_like(smudged, 200), 0.15, 0.0)
    out = (image.astype(np.float32) * (1 - mask) + smudged.astype(np.float32) * mask)

    return DegradationResult(
        out.clip(0, 255).astype(np.uint8), "defect", _score_for(bucket, t), "lens_smudge",
        {"centre": [cx, cy], "axes": list(axes), "radius_fraction": round(radius_frac, 4)},
    )


def vignette(image: np.ndarray, bucket: int, rng: np.random.Generator) -> DegradationResult:
    """Radial corner falloff from optical shading or a mis-seated hood."""
    t = _sample_t(rng)
    strength = _interp(_VIGNETTE_STRENGTH[bucket], t)
    height, width = image.shape[:2]

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cy, cx = height / 2.0, width / 2.0
    distance = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    falloff = np.clip(1.0 - strength * (distance / np.sqrt(2.0)) ** 2.2, 0.0, 1.0)

    out = (image.astype(np.float32) * falloff[..., None]).clip(0, 255).astype(np.uint8)
    return DegradationResult(
        out, "defect", _score_for(bucket, t), "vignette", {"strength": round(strength, 3)}
    )


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

DegradationFn = Callable[[np.ndarray, int, np.random.Generator], DegradationResult]

DEGRADATIONS: dict[str, dict[str, DegradationFn]] = {
    "blur": {"gaussian_blur": gaussian_blur, "motion_blur": motion_blur},
    "underexposure": {"gamma_gain": underexposure},
    "overexposure": {"gamma_gain": overexposure},
    "noise": {
        "gaussian_noise": gaussian_noise,
        "salt_pepper_noise": salt_pepper_noise,
        "speckle_noise": speckle_noise,
    },
    "corruption": {
        "jpeg_recompression": jpeg_recompression,
        "block_corruption": block_corruption,
        "occlusion": occlusion,
        "row_tear": row_tear,
        "channel_desync": channel_desync,
    },
    "defect": {
        "scratches": scratches,
        "dust_and_hot_pixels": dust_and_hot_pixels,
        "lens_smudge": lens_smudge,
        "vignette": vignette,
    },
}

#: Order in which stacked degradations are applied, mirroring a real capture
#: pipeline: exposure is set at capture, optics blur the projected image, the
#: sensor contributes defects and then noise, and compression happens last.
#: Applying them out of order produces combinations that cannot physically occur
#: (e.g. noise that survives being blurred).
APPLICATION_ORDER: tuple[str, ...] = (
    "underexposure",
    "overexposure",
    "blur",
    "defect",
    "noise",
    "corruption",
)


def apply_degradation(
    image: np.ndarray,
    issue: str,
    bucket: int,
    rng: np.random.Generator,
    method: str | None = None,
) -> DegradationResult:
    """Apply one degradation of ``issue`` at the requested severity bucket."""
    if issue not in DEGRADATIONS:
        raise KeyError(f"unknown issue type {issue!r}; expected one of {ISSUE_TYPES}")
    methods = DEGRADATIONS[issue]
    if method is None:
        method = str(rng.choice(sorted(methods)))
    if method not in methods:
        raise KeyError(f"unknown method {method!r} for issue {issue!r}")
    return methods[method](image, bucket, rng)


# --------------------------------------------------------------------------
# Visual smoke test
# --------------------------------------------------------------------------


def _demo(source: Path, out_dir: Path, seed: int) -> None:
    """Render every (issue, severity) pair from one image for eyeball checks."""
    from focal_ml.constants import CANONICAL_LONG_SIDE, GENERATED_JPEG_QUALITY
    from focal_ml.utils import resize_long_side

    image = imread_bgr(source)
    if image is None:
        raise SystemExit(f"could not read image: {source}")
    image = resize_long_side(image, CANONICAL_LONG_SIDE)

    out_dir.mkdir(parents=True, exist_ok=True)
    imwrite_jpeg(out_dir / "00_clean.jpg", image, GENERATED_JPEG_QUALITY)

    rows = []
    for issue in ISSUE_TYPES:
        for method in sorted(DEGRADATIONS[issue]):
            for bucket in (1, 2, 3):
                rng = np.random.default_rng([seed, hash(method) % 10_000, bucket])
                result = apply_degradation(image, issue, bucket, rng, method=method)
                name = f"{issue}__{method}__b{bucket}_s{result.severity_score:.2f}.jpg"
                imwrite_jpeg(out_dir / name, result.image, GENERATED_JPEG_QUALITY)
                rows.append((issue, method, bucket, result.severity_bucket, result.severity_score))

    print(f"wrote {len(rows) + 1} images to {out_dir}\n")
    print(f"{'issue':<14} {'method':<20} {'asked':>5} {'got':>4} {'score':>6}")
    for issue, method, asked, got, score in rows:
        flag = "" if asked == got else "  <- rescored from achieved result"
        print(f"{issue:<14} {method:<20} {asked:>5} {got:>4} {score:>6.3f}{flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", type=Path, required=True, help="clean source image to degrade")
    parser.add_argument("--out", type=Path, default=Path("dataset/demo"), help="output directory")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    _demo(args.demo, args.out, args.seed)


if __name__ == "__main__":
    main()
