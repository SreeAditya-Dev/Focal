"""Shared fixtures for the feature and rule tests."""

from __future__ import annotations

import cv2
import numpy as np
import pytest


def natural_image(height: int = 512, width: int = 768, seed: int = 0, beta: float = 1.8) -> np.ndarray:
    """Synthesise an image with photograph-like statistics.

    Natural scenes have an approximately 1/f^beta power spectrum — detail at
    every scale, with energy falling off smoothly toward high frequencies. A
    geometric test chart does not: it is dominated by a few hard edges and large
    flat regions, and features measured on it behave nothing like features
    measured on a photograph.

    Since these tests exist to check how features respond to *degradation*, the
    base image has to be statistically plausible or the responses mean nothing.
    """
    rng = np.random.default_rng(seed)

    frequency_y = np.fft.fftshift(np.fft.fftfreq(height))[:, None]
    frequency_x = np.fft.fftshift(np.fft.fftfreq(width))[None, :]
    radius = np.sqrt(frequency_y**2 + frequency_x**2)
    radius[radius == 0] = 1e-6
    falloff = radius ** (-beta / 2.0)

    def plane(local_seed: int) -> np.ndarray:
        white = np.random.default_rng(local_seed).normal(size=(height, width))
        spectrum = np.fft.fftshift(np.fft.fft2(white)) * falloff
        field = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum)))
        field -= field.min()
        return field / (field.max() + 1e-9)

    # The three planes share a dominant luminance component with small
    # independent chroma variation, which is what keeps the colour channels of a
    # real photograph highly correlated.
    luminance = plane(seed)
    channels = [
        np.clip(luminance * 0.82 + plane(seed + 100 + index) * 0.18, 0, 1)
        for index in range(3)
    ]
    image = (np.stack(channels[::-1], axis=-1) * 190 + 35).astype(np.uint8)

    # A few hard edges, so the image also contains the sharp structure that
    # sharpness metrics are built to detect.
    for _ in range(6):
        x0, y0 = int(rng.integers(0, width - 120)), int(rng.integers(0, height - 120))
        colour = tuple(int(v) for v in rng.integers(30, 225, 3))
        cv2.rectangle(image, (x0, y0), (x0 + 100, y0 + 90), colour, -1)

    return image


@pytest.fixture(scope="session")
def clean_image() -> np.ndarray:
    return natural_image(seed=7)


@pytest.fixture(scope="session")
def clean_images() -> list[np.ndarray]:
    """Several independent scenes, so a test cannot pass by luck on one image.

    Three is a deliberate compromise: enough that a directional claim holding
    across all of them is meaningful, few enough that the suite stays runnable
    (each scene costs a full extraction per degradation per test).
    """
    return [natural_image(seed=s) for s in (1, 2, 3)]
