"""Sharpness and blur measurement.

The naive blur detector — variance of the Laplacian, thresholded — is famously
content-dependent: a photograph of a blank wall scores lower than a slightly
blurred photograph of foliage, because the metric measures how much detail is
*present* rather than how much has been *lost*. Three things here work around
that:

  * ``sharpness_ratio`` normalises Laplacian energy by local contrast, so the
    measure asks "is the detail this scene contains resolved?" rather than "does
    this scene contain detail?".
  * ``hf_energy_ratio`` works in the frequency domain and is a ratio by
    construction, so overall image energy cancels out.
  * Tile statistics expose *partial* blur. A lens smudge or a shallow depth of
    field leaves the global metrics looking healthy while a region of the frame
    is unresolved; only the distribution across tiles reveals it.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from focal_ml.utils import tile_stats

EPS = 1e-6

#: Tiles flatter than this carry no resolvable detail (blank sky, a white wall),
#: so their sharpness is meaningless and they are excluded from tile statistics.
#: Without this filter an image containing sky always reports partial blur.
MIN_TILE_CONTRAST = 6.0


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian — the standard, content-dependent baseline."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad(gray: np.ndarray) -> float:
    """Mean squared Sobel gradient magnitude.

    Reads first-derivative energy where the Laplacian reads the second, so it
    responds differently to noise — noise inflates the Laplacian much more.
    Disagreement between the two is itself informative.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    # np.mean rather than cv2.mean: OpenCV's threaded reduction is not
    # bit-reproducible between runs. See channel_edge_correlation.
    return float(np.mean(gx * gx + gy * gy, dtype=np.float64))


def edge_density(gray: np.ndarray, low: int = 50, high: int = 150) -> float:
    """Fraction of pixels Canny marks as edges."""
    return float(np.count_nonzero(cv2.Canny(gray, low, high)) / gray.size)


#: The FFT runs at a fixed square size rather than the image's own shape. This
#: keeps the cost constant regardless of upload resolution and lets the radial
#: mask and window be computed once and reused. Squashing the aspect ratio
#: distorts the radial energy distribution, but does so identically for every
#: image, so the measure stays comparable — which is all a ratio needs to be.
FFT_SIZE = 256


@lru_cache(maxsize=4)
def _fft_kernels(size: int, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """Hann window and high-frequency mask, built once per (size, cutoff)."""
    window = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)
    centre = size / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    radius = np.sqrt((yy - centre) ** 2 + (xx - centre) ** 2) / (centre * np.sqrt(2.0))
    return window, radius > cutoff


def hf_energy_ratio(gray: np.ndarray, cutoff: float = 0.25) -> float:
    """Share of Fourier magnitude above ``cutoff`` of the Nyquist radius.

    Blur is a low-pass filter, so it attenuates exactly this band. Because the
    measure is a ratio of energies it is largely invariant to exposure and
    overall contrast, which makes it the most reliable single blur cue here.
    """
    small = cv2.resize(gray, (FFT_SIZE, FFT_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    # Windowing suppresses the cross-shaped spectral artifact produced by the
    # discontinuity between opposite image edges, which otherwise leaks
    # broadband energy and masks the effect of blur.
    window, high_band = _fft_kernels(FFT_SIZE, cutoff)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(small * window)))

    total = float(spectrum.sum())
    if total <= EPS:
        return 0.0
    return float(spectrum[high_band].sum() / total)


def tile_sharpness(gray: np.ndarray, grid: int = 8) -> np.ndarray:
    """Contrast-normalised sharpness per tile, for tiles with enough detail.

    The Laplacian is taken once over the whole image and then tiled, rather than
    being recomputed per tile. Per-tile Laplacian variance is exactly the
    variance of the tiled response, so this is equivalent apart from a one-pixel
    border effect, and is an order of magnitude cheaper.

    Returns an empty array when the image is uniformly featureless.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    _, laplacian_spread = tile_stats(laplacian, grid)
    _, tile_contrast = tile_stats(gray, grid)

    detailed = tile_contrast >= MIN_TILE_CONTRAST
    if not detailed.any():
        return np.empty(0, dtype=np.float64)

    variance = laplacian_spread[detailed] ** 2
    contrast = tile_contrast[detailed] ** 2
    return (variance / (contrast + EPS)).astype(np.float64)


def sharpness_features(gray: np.ndarray) -> dict[str, float]:
    contrast = float(gray.std())
    lap_var = laplacian_variance(gray)
    tiles = tile_sharpness(gray)

    if tiles.size:
        tile_p10 = float(np.percentile(tiles, 10))
        tile_median = float(np.median(tiles))
        # Ratio of the weakest tiles to the typical tile: how unevenly sharpness
        # is spread across the frame.
        #
        # Measured behaviour, which is not the intuitive one: this drops for a
        # lens smudge (0.12 vs 0.29 clean) but drops just as far for *uniform*
        # blur (0.14), because blur flattens weakly-textured tiles faster than
        # strongly-textured ones. So it detects "sharpness is uneven" without
        # identifying the cause, and cannot by itself separate a localised
        # defect from global blur. It is kept as a model input, where it can be
        # weighed against the other blur evidence, but is deliberately not used
        # as a defect rule. See fusion/rules.py.
        uniformity = float(tile_p10 / (tile_median + EPS))
    else:
        tile_p10 = tile_median = 0.0
        uniformity = 1.0

    return {
        "sharpness_laplacian_var": lap_var,
        "sharpness_ratio": lap_var / (contrast * contrast + EPS),
        "sharpness_tenengrad": tenengrad(gray),
        "edge_density": edge_density(gray),
        "hf_energy_ratio": hf_energy_ratio(gray),
        "tile_sharpness_p10": tile_p10,
        "tile_sharpness_median": tile_median,
        "tile_sharpness_uniformity": uniformity,
        "tile_sharpness_count": float(tiles.size),
    }
