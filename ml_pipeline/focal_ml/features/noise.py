"""Noise estimation.

Every noise estimator is confounded by texture: a photograph of gravel, foliage
or fabric contains high-frequency variation that is *signal*, and a naive
estimator reports it as noise. The defence used here is to estimate the noise
floor from the flattest regions of the image, where by construction almost all
remaining variation is noise.

Three estimators are computed because they fail differently, and their
disagreement is itself a feature:

  * Immerkaer — fast, whole-image, inflated by texture.
  * Flat-region — robust to texture, degrades when the image has no flat areas.
  * Median residual — robust to impulse noise, which the other two under-report
    because a few extreme outliers barely move a variance.

Donoho's wavelet-domain estimator (``skimage.restoration.estimate_sigma``) is
deliberately *not* used. It applies the same MAD statistic that
``median_residual_sigma`` already applies in the spatial domain, so it adds
little, and it needs PyWavelets — an optional dependency of scikit-image. Making
it optional here would be worse than either alternative: the feature would be
populated while training and silently zero in a slim production image, which is
train/serve skew that no test would catch.
"""

from __future__ import annotations

import cv2
import numpy as np

EPS = 1e-6

#: Residual above which a pixel is judged an impulse rather than detail.
IMPULSE_THRESHOLD = 60.0


def immerkaer_sigma(gray: np.ndarray) -> float:
    """Immerkaer's estimator: convolve with a kernel that annihilates both
    constant and locally-linear content, leaving mostly noise."""
    height, width = gray.shape
    if height < 3 or width < 3:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    response = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    return float(np.sqrt(np.pi / 2.0) * np.abs(response).sum() / (6.0 * (width - 2) * (height - 2)))


def flat_region_sigma(gray: np.ndarray, block: int = 16, percentile: float = 5.0) -> float:
    """Noise floor read from the flattest blocks in the image.

    Taking a low percentile of per-block standard deviations selects blocks with
    the least structure. Whatever variation survives there is the sensor floor,
    so this is the estimate least polluted by texture.
    """
    height, width = gray.shape
    rows, cols = height // block, width // block
    if rows < 2 or cols < 2:
        return immerkaer_sigma(gray)

    trimmed = gray[: rows * block, : cols * block].astype(np.float32)
    blocks = trimmed.reshape(rows, block, cols, block).swapaxes(1, 2).reshape(-1, block * block)
    return float(np.percentile(blocks.std(axis=1), percentile))


def median_residual_sigma(gray: np.ndarray, ksize: int = 3) -> float:
    """Robust spread of the image minus its median filtering.

    Uses a normalised median absolute deviation rather than a standard
    deviation so that a handful of impulse pixels cannot dominate the estimate.
    """
    # np.median sorts, so this runs on a capped-size sample rather than the full
    # image. A median converges quickly and the noise floor is spatially
    # stationary, so a crop of the image gives the same answer far more cheaply.
    if gray.shape[0] > 384:
        gray = np.ascontiguousarray(gray[::2, ::2])
    residual = gray.astype(np.float32) - cv2.medianBlur(gray, ksize).astype(np.float32)
    return float(1.4826 * np.median(np.abs(residual - np.median(residual))))


def impulse_ratio(gray: np.ndarray, ksize: int = 3, threshold: float = IMPULSE_THRESHOLD) -> float:
    """Fraction of pixels differing wildly from their local median.

    Salt-and-pepper noise and hot pixels produce isolated extreme values that
    variance-based estimators barely register but that dominate this measure.
    """
    residual = np.abs(gray.astype(np.float32) - cv2.medianBlur(gray, ksize).astype(np.float32))
    return float(np.count_nonzero(residual > threshold) / gray.size)


def chroma_noise_sigma(image_bgr: np.ndarray) -> float:
    """Noise floor of the chroma planes.

    Sensor noise at high ISO is strongly chromatic — coloured speckle in what
    should be flat neutral areas — whereas most legitimate fine texture is
    luminance detail. Measuring the colour channels separately therefore
    separates grain from detail better than any luma-only estimator.
    """
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    return float(np.mean([flat_region_sigma(ycrcb[:, :, index]) for index in (1, 2)]))


def noise_features(image_bgr: np.ndarray, gray: np.ndarray) -> dict[str, float]:
    immerkaer = immerkaer_sigma(gray)
    flat = flat_region_sigma(gray)

    return {
        "noise_sigma_immerkaer": immerkaer,
        "noise_sigma_flat": flat,
        "noise_sigma_chroma": chroma_noise_sigma(image_bgr),
        "noise_sigma_median_residual": median_residual_sigma(gray),
        "noise_impulse_ratio": impulse_ratio(gray),
        # High when the whole-image estimate greatly exceeds the flat-region
        # floor, i.e. the image is textured rather than noisy. Lets a rule tell
        # "detailed" apart from "grainy", which no single estimator can.
        "noise_texture_ratio": float(immerkaer / (flat + EPS)),
    }
