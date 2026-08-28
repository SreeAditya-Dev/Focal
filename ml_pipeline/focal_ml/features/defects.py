"""Localised defect signatures.

The hardest of the six classes for hand-written rules, because "defect" is
defined by *where* something is wrong rather than by any global statistic. A
scratch across a corner leaves every whole-image measurement essentially
unchanged.

These features therefore look at spatial structure rather than distribution:
how brightness varies with distance from the centre, and whether the image
contains long thin high-contrast structures. Both are weak individually — real
architecture contains straight lines, real photographs are often darker at the
edges — so the rule layer gives them modest weight and leans on the CNN for
this class. That asymmetry is a deliberate part of the hybrid design, not a gap
in it.
"""

from __future__ import annotations

import cv2
import numpy as np

from focal_ml.utils import tile_stats

EPS = 1e-6


def radial_brightness_falloff(gray: np.ndarray) -> float:
    """Ratio of mean brightness in the outer ring to the central disc.

    Optical vignetting darkens the corners while leaving the centre alone, so
    this drops below 1. Values near or above 1 are normal.
    """
    small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    yy, xx = np.mgrid[0:128, 0:128]
    radius = np.sqrt((yy - 63.5) ** 2 + (xx - 63.5) ** 2) / 63.5

    centre = small[radius < 0.35]
    outer = small[radius > 0.85]
    if centre.size == 0 or outer.size == 0:
        return 1.0
    return float(outer.mean() / (centre.mean() + EPS))


def linear_structure_score(gray: np.ndarray) -> float:
    """Total length of detected straight lines, relative to image size.

    Scratches are long, thin, high-contrast and straight. So are window frames
    and railings, which is why this is only ever corroborating evidence.
    """
    scale = 256.0 / max(gray.shape[:2])
    small = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    edges = cv2.Canny(small, 60, 180)
    diagonal = float(np.hypot(*small.shape[:2]))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=int(diagonal * 0.15),
        )
    if lines is None or len(lines) == 0:
        return 0.0

    return float(total / (diagonal * 20.0))


def local_contrast_spread(gray: np.ndarray, grid: int = 8) -> float:
    """Coefficient of variation of per-tile contrast.

    A smudge or a pasted patch creates a region whose local contrast departs
    sharply from the rest of the frame, widening this spread even when mean
    contrast is unremarkable.
    """
    _, tile_contrast = tile_stats(gray, grid)
    return float(tile_contrast.std() / (tile_contrast.mean() + EPS))


def defect_features(gray: np.ndarray) -> dict[str, float]:
    return {
        "radial_falloff": radial_brightness_falloff(gray),
        "linear_structure": linear_structure_score(gray),
        "local_contrast_spread": local_contrast_spread(gray),
    }
