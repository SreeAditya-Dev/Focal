"""Small image helpers shared by the dataset, training and inference code."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from focal_ml.constants import CANONICAL_LONG_SIDE


def resize_long_side(image: np.ndarray, long_side: int = CANONICAL_LONG_SIDE) -> np.ndarray:
    """Scale an image so its longest edge is ``long_side``, preserving aspect.

    Images already at or below the target are returned untouched — upscaling a
    small image would fabricate sharpness the original never had.
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= long_side:
        return image
    scale = long_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    # INTER_AREA is the correct choice for downscaling; it averages rather than
    # point-samples, so it does not alias fine texture into false noise.
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def to_gray(image: np.ndarray) -> np.ndarray:
    """Return a single-channel uint8 luma view of a BGR or grayscale image."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def imread_bgr(path: str | Path) -> np.ndarray | None:
    """Read an image as BGR uint8, tolerating non-ASCII paths on Windows.

    ``cv2.imread`` cannot open paths outside the active code page, so decode
    from bytes instead. Returns ``None`` if the file is not a decodable image.
    """
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if raw.size == 0:
        return None
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    return image


def imwrite_jpeg(path: str | Path, image: np.ndarray, quality: int) -> bool:
    """Write a BGR image as JPEG, tolerating non-ASCII paths on Windows."""
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer.tofile(str(path))
    return True


def jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    """Encode to JPEG at ``quality`` and decode back, in memory."""
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return image
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return image if decoded is None else decoded
