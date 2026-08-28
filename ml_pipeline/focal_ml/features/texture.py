"""Texture descriptors.

Texture statistics separate two conditions that intensity statistics confuse:
detail that was never there, and detail that has been destroyed. Aggressive
denoising, heavy compression and blur all *raise* GLCM homogeneity and energy,
because they replace varied micro-structure with locally uniform patches.
"""

from __future__ import annotations

import cv2
import numpy as np

#: GLCM cost grows with the square of the level count, so the image is
#: quantised and downscaled first. 32 levels preserves the contrast/homogeneity
#: structure that matters while keeping extraction inside a few milliseconds.
GLCM_LEVELS = 32
GLCM_LONG_SIDE = 192


def glcm_features(gray: np.ndarray) -> dict[str, float]:
    try:
        from skimage.feature import graycomatrix, graycoprops
    except ImportError:
        return {name: 0.0 for name in ("glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation")}

    height, width = gray.shape
    scale = GLCM_LONG_SIDE / float(max(height, width))
    if scale < 1.0:
        gray = cv2.resize(gray, (max(2, int(width * scale)), max(2, int(height * scale))),
                          interpolation=cv2.INTER_AREA)

    quantised = (gray.astype(np.uint16) * GLCM_LEVELS // 256).astype(np.uint8)
    matrix = graycomatrix(
        quantised,
        distances=[1],
        angles=[0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )

    def prop(name: str) -> float:
        # Averaging the four orientations makes the descriptor rotation
        # invariant, which matters because motion blur is directional and would
        # otherwise be read differently depending on the shake angle.
        return float(np.mean(graycoprops(matrix, name)))

    return {
        "glcm_contrast": prop("contrast"),
        "glcm_homogeneity": prop("homogeneity"),
        "glcm_energy": prop("energy"),
        "glcm_correlation": prop("correlation"),
    }


def lbp_uniformity(gray: np.ndarray) -> float:
    """Share of pixels whose local binary pattern is 'uniform'.

    Uniform patterns describe ordinary structure — edges, corners, flat areas.
    Random noise produces non-uniform patterns, so this falls as noise rises and
    rises as detail is smoothed away.
    """
    try:
        from skimage.feature import local_binary_pattern
    except ImportError:
        return 0.0

    height, width = gray.shape
    scale = GLCM_LONG_SIDE / float(max(height, width))
    if scale < 1.0:
        gray = cv2.resize(gray, (max(2, int(width * scale)), max(2, int(height * scale))),
                          interpolation=cv2.INTER_AREA)

    codes = local_binary_pattern(gray, P=8, R=1, method="uniform")
    # method="uniform" maps every non-uniform pattern onto the single code P+1.
    return float(np.count_nonzero(codes < 9) / codes.size)


def texture_features(gray: np.ndarray) -> dict[str, float]:
    return {**glcm_features(gray), "lbp_uniformity": lbp_uniformity(gray)}
