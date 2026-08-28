"""The single definition of how an image becomes a model input.

Both the training dataset and the inference predictor import this. Keeping one
copy is not tidiness — a divergence between how images are prepared for
training and how they are prepared for serving is the most common way a model
that scored well offline performs badly in production, and it produces no error
message when it happens.
"""

from __future__ import annotations

import cv2
import numpy as np

from focal_ml.constants import CNN_INPUT_SIZE

#: The backbone's pretrained filters were fitted in this normalised space.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_for_cnn(
    image_bgr: np.ndarray, size: int = CNN_INPUT_SIZE, flip: bool = False
) -> np.ndarray:
    """Turn a BGR uint8 image into a normalised CHW float array.

    The whole frame is squashed to ``size`` x ``size`` rather than being resized
    and centre-cropped. The ImageNet convention discards roughly a quarter of
    the image, and here a defect can sit anywhere in the frame — cropping one
    away turns a correctly-labelled defective image into a mislabelled clean
    one. Distorted aspect ratio costs less than lost edges.

    Args:
        image_bgr: decoded BGR uint8 image, any size.
        flip: horizontal flip, the only label-preserving augmentation available.
    """
    resized = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if flip:
        rgb = rgb[:, ::-1]
    normalised = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(normalised.transpose(2, 0, 1))
