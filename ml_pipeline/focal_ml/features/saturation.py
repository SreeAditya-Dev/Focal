"""Colour and saturation measurement.

Colour carries evidence two other feature groups miss. Overexposure desaturates
as channels clip toward white, so falling saturation corroborates a bright
histogram. And the three channels of a natural photograph are strongly
correlated in their edge structure — a decode fault that displaces one plane
breaks that correlation while leaving every intensity statistic intact.
"""

from __future__ import annotations

import cv2
import numpy as np

EPS = 1e-6

#: Below this HSV saturation a pixel reads as neutral grey.
GREY_SATURATION = 20


def _downscale(image: np.ndarray, long_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= long_side:
        return image
    scale = long_side / float(longest)
    return cv2.resize(
        image, (max(2, int(width * scale)), max(2, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def colorfulness(image_bgr: np.ndarray) -> float:
    """Hasler & Suesstrunk's colourfulness metric (2003).

    Measured in an opponent-colour space that approximates how human vision
    encodes chroma, so it tracks perceived vividness rather than raw channel
    spread.
    """
    # A distribution statistic over millions of pixels is unchanged by sampling
    # a quarter of a million of them.
    image_bgr = _downscale(image_bgr, 256)
    blue, green, red = (channel.astype(np.float32) for channel in cv2.split(image_bgr))
    rg = red - green
    yb = 0.5 * (red + green) - blue
    std_root = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2))
    mean_root = float(np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return std_root + 0.3 * mean_root


def channel_edge_correlation(image_bgr: np.ndarray) -> float:
    """Mean correlation between the edge maps of the three colour planes.

    In a real scene an object boundary appears at the same coordinates in every
    channel, so these correlate near 1. Chroma desynchronisation shifts one
    plane and the correlation collapses — a signal no greyscale measurement can
    see, because converting to luma averages the displacement away.
    """
    # Kept at 384 rather than reduced further: the mildest desynchronisation
    # this needs to catch displaces a plane by only a couple of pixels, and
    # aggressive downscaling would average that shift away entirely.
    image_bgr = _downscale(image_bgr, 384)

    channels = []
    for index in range(3):
        plane = image_bgr[:, :, index]
        gx = cv2.Sobel(plane, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(plane, cv2.CV_32F, 0, 1, ksize=3)
        channels.append(cv2.magnitude(gx, gy))

    # Reduced with NumPy in float64 rather than cv2.mean/meanStdDev. OpenCV's
    # reductions are multi-threaded, and floating-point addition is not
    # associative, so their result varies in the last few digits between runs.
    # NumPy's pairwise summation is single-threaded and fixed-order, which keeps
    # the whole feature vector reproducible — cached features have to compare
    # equal to recomputed ones.
    correlations = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        first = channels[a].ravel()
        second = channels[b].ravel()
        first_mean = np.mean(first, dtype=np.float64)
        second_mean = np.mean(second, dtype=np.float64)
        first_spread = np.std(first, dtype=np.float64)
        second_spread = np.std(second, dtype=np.float64)
        if first_spread < EPS or second_spread < EPS:
            continue
        joint = np.mean(first * second, dtype=np.float64)
        correlations.append(float((joint - first_mean * second_mean) / (first_spread * second_spread)))

    return float(np.mean(correlations)) if correlations else 1.0


def saturation_features(image_bgr: np.ndarray) -> dict[str, float]:
    saturation = cv2.cvtColor(_downscale(image_bgr, 384), cv2.COLOR_BGR2HSV)[:, :, 1]

    return {
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
        "colorfulness": colorfulness(image_bgr),
        "grey_pixel_fraction": float(np.count_nonzero(saturation < GREY_SATURATION) / saturation.size),
        "channel_edge_correlation": channel_edge_correlation(image_bgr),
    }
