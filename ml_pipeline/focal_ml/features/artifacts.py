"""Structural corruption signatures.

Corruption is not a smooth degradation of an image statistic — it is the
appearance of structure that photography does not produce. Each measure here
targets one such signature, because the corruption class covers several
mechanisms that share no common statistical trend:

  * ``blockiness`` — discontinuity locked to the JPEG 8x8 grid.
  * ``flat_block_fraction`` — blocks collapsed to a single colour by DC-only decode.
  * ``row_discontinuity`` — a horizontal seam, as when a scanline band is displaced.
  * ``largest_uniform_region`` — a contiguous area with no content at all.

They are combined disjunctively by the rule layer: any one firing is evidence,
so a mechanism absent from an image cannot dilute one that is present.
"""

from __future__ import annotations

import cv2
import numpy as np

EPS = 1e-6

BLOCK = 8
#: Standard deviation below which an 8x8 block counts as perfectly flat.
FLAT_BLOCK_STD = 1.0


def blockiness(gray: np.ndarray) -> float:
    """Ratio of intensity steps on the JPEG block grid to steps elsewhere.

    A clean photograph has no reason to place discontinuities every eighth
    pixel, so this sits near 1.0. Quantisation of the DCT makes adjacent blocks
    disagree at their shared boundary, driving it above 1.
    """
    image = gray.astype(np.float32)
    ratios = []

    for axis in (0, 1):
        differences = np.abs(np.diff(image, axis=axis))
        length = differences.shape[axis]
        if length < BLOCK * 2:
            continue
        # Column/row indices where a block boundary falls.
        index = np.arange(length)
        on_grid = (index + 1) % BLOCK == 0
        if not on_grid.any() or on_grid.all():
            continue
        boundary = differences.take(np.flatnonzero(on_grid), axis=axis).mean()
        interior = differences.take(np.flatnonzero(~on_grid), axis=axis).mean()
        ratios.append(float(boundary / (interior + EPS)))

    return float(np.mean(ratios)) if ratios else 1.0


def block_statistics(gray: np.ndarray) -> tuple[float, float]:
    """Fraction of flat 8x8 blocks, and how abruptly blocks differ from
    their right-hand neighbour.

    Zeroing the AC coefficients of a block leaves it a single flat colour, and a
    run of such blocks produces both a spike in flatness and large jumps between
    neighbouring block means.
    """
    height, width = gray.shape
    rows, cols = height // BLOCK, width // BLOCK
    if rows < 2 or cols < 2:
        return 0.0, 0.0

    trimmed = gray[: rows * BLOCK, : cols * BLOCK].astype(np.float32)
    blocks = trimmed.reshape(rows, BLOCK, cols, BLOCK).swapaxes(1, 2)
    flat = blocks.reshape(rows, cols, -1)

    stds = flat.std(axis=2)
    means = flat.mean(axis=2)
    neighbour_jump = np.abs(np.diff(means, axis=1))

    flat_fraction = float(np.count_nonzero(stds < FLAT_BLOCK_STD) / stds.size)
    # A high percentile rather than the mean: corruption affects a small
    # fraction of blocks very severely, which an average would wash out.
    jump = float(np.percentile(neighbour_jump, 99.5)) if neighbour_jump.size else 0.0
    return flat_fraction, jump


def line_discontinuity(gray: np.ndarray, axis: int = 0) -> float:
    """How far the worst row-to-row (or column-to-column) step exceeds typical.

    Natural images change gradually down the frame, so the largest step is a
    small multiple of the median. A displaced band of scanlines creates two
    seams that are outliers by a wide margin.
    """
    image = gray.astype(np.float32)
    steps = np.abs(np.diff(image, axis=axis)).mean(axis=1 - axis)
    if steps.size < 8:
        return 1.0
    median = float(np.median(steps))
    return float(steps.max() / (median + EPS))


def largest_uniform_region(gray: np.ndarray) -> float:
    """Area fraction of the biggest contiguous featureless region.

    Detects an occluding patch pasted over the frame. Sky and studio
    backgrounds also register, so this is deliberately given a high onset in the
    rule layer and never fires on its own.
    """
    small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    # Local standard deviation via the identity Var(X) = E[X^2] - E[X]^2.
    mean = cv2.blur(small, (9, 9))
    mean_square = cv2.blur(small * small, (9, 9))
    local_std = np.sqrt(np.maximum(mean_square - mean * mean, 0.0))

    featureless = (local_std < 1.5).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(featureless, connectivity=8)
    if count <= 1:
        return 0.0
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return float(largest / small.size)


def byte_entropy(raw: bytes | None) -> float:
    """Shannon entropy of the encoded file, in bits per byte.

    Compressed image data is near-random and sits around 7.9. A markedly lower
    value means long runs of repeated bytes, which is what a truncated or
    zero-padded file looks like.
    """
    if not raw:
        return 0.0
    counts = np.bincount(np.frombuffer(raw, dtype=np.uint8), minlength=256).astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def artifact_features(gray: np.ndarray, raw: bytes | None = None) -> dict[str, float]:
    flat_fraction, block_jump = block_statistics(gray)
    return {
        "blockiness": blockiness(gray),
        "flat_block_fraction": flat_fraction,
        "block_mean_jump": block_jump,
        "row_discontinuity": line_discontinuity(gray, axis=0),
        "col_discontinuity": line_discontinuity(gray, axis=1),
        "largest_uniform_region": largest_uniform_region(gray),
        "byte_entropy": byte_entropy(raw),
    }
