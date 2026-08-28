"""Brightness, exposure and contrast measurement.

Mean brightness alone is a poor exposure detector: a legitimately dark scene at
night and an underexposed daylight photograph have similar means but differ in
what happened at the ends of the histogram. Clipping fractions carry that
distinction — clipped pixels are *destroyed* detail, which is what makes bad
exposure unrecoverable rather than merely dark.
"""

from __future__ import annotations

import cv2
import numpy as np

EPS = 1e-6

#: Histogram bins treated as crushed shadow / blown highlight.
SHADOW_BINS = 16   # 0-15
HIGHLIGHT_BINS = 16  # 240-255


def exposure_features(image_bgr: np.ndarray, gray: np.ndarray) -> dict[str, float]:
    # The V channel of HSV is max(R,G,B) rather than a luma weighting, so it
    # reaches 255 as soon as any single channel clips. That makes it the correct
    # basis for a highlight-clipping measurement, where luma would hide a blown
    # red channel behind unclipped green and blue.
    value = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]

    histogram = cv2.calcHist([value], [0], None, [256], [0, 256]).ravel()
    total = float(histogram.sum()) + EPS

    percentiles = np.percentile(gray, [1, 5, 50, 95, 99])
    normalised = histogram / total
    nonzero = normalised[normalised > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())

    return {
        "brightness_mean": float(value.mean()),
        "brightness_std": float(value.std()),
        "brightness_p01": float(percentiles[0]),
        "brightness_p05": float(percentiles[1]),
        "brightness_median": float(percentiles[2]),
        "brightness_p95": float(percentiles[3]),
        "brightness_p99": float(percentiles[4]),
        "shadow_clip_fraction": float(histogram[:SHADOW_BINS].sum() / total),
        "highlight_clip_fraction": float(histogram[-HIGHLIGHT_BINS:].sum() / total),
        "contrast_rms": float(gray.std()),
        # Spread of the middle 98% of tones. Collapses when the image is fogged,
        # washed out, or pushed against either end of the range.
        "dynamic_range": float(percentiles[4] - percentiles[0]),
        # Bits of tonal information actually used. Falls when exposure work
        # quantises many distinct input levels onto the same output level.
        "histogram_entropy": entropy,
    }
