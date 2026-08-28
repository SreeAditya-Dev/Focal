"""Grad-CAM localisation for the presence heads.

Answers "where in this frame is the problem?" by weighting the last
convolutional feature maps with the gradient of one issue's logit with respect
to them. Regions whose activation most increases that logit light up.

Two details specific to this model:

  * The gradient is taken from the presence **logit**, not the sigmoid output.
    Once a confident prediction saturates the sigmoid its derivative approaches
    zero, and the resulting map is numerical noise — exactly for the confident
    predictions a user is most likely to ask about.

  * The map is computed per issue rather than once per image. "Where is the
    blur" and "where is the corruption" are different questions with different
    answers, and a single map averaged over issues answers neither.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np
import torch
import torch.nn as nn

from focal_ml.constants import ISSUE_INDEX


class GradCAM:
    """Hooks the backbone's final convolution and produces per-issue heatmaps.

    Hooks are registered for the object's lifetime and removed by ``close()``.
    The predictor holds one instance rather than building one per request, since
    re-registering hooks on every call leaks them.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        if not getattr(model.config, "use_image", False):
            raise ValueError("Grad-CAM needs the image branch; this model has none")

        self.model = model
        # The last block of MobileNetV3's feature trunk: the deepest layer that
        # still has spatial extent (7x7 at 224 input). Deeper is more semantic,
        # but there is nothing deeper before global pooling discards position.
        self.layer = target_layer if target_layer is not None else model.features[-1]

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            self.layer.register_forward_hook(self._save_activations),
            self.layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(self, _module, _inputs, output) -> None:
        self._activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def generate(
        self,
        image: torch.Tensor,
        issue: str,
        features: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Return a 0-1 saliency map at the backbone's spatial resolution.

        Args:
            image: a single normalised image tensor, shape (1, 3, H, W).
            issue: which issue's logit to explain.
            features: the classical feature vector, if the model uses one.
        """
        if issue not in ISSUE_INDEX:
            raise KeyError(f"unknown issue {issue!r}")

        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        # Gradients are the entire mechanism, so this cannot run under
        # no_grad(), even though every other inference path does.
        with torch.enable_grad():
            image = image.detach().requires_grad_(False)
            output = self.model(
                image=image,
                features=features if self.model.config.use_features else None,
            )
            output["presence_logits"][0, ISSUE_INDEX[issue]].backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("hooks captured nothing; the target layer may not be on the forward path")

        # One weight per channel: how much increasing that channel's activation
        # anywhere would raise this issue's logit.
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        weighted = (weights * self._activations).sum(dim=1, keepdim=False)[0]
        self.model.train(was_training)

        cam_array = weighted.cpu().numpy().astype(np.float64)
        peak = float(cam_array.max())

        # The test for "no localised evidence" has to be the *sign* of the peak,
        # not its magnitude. A CAM's absolute scale is the product of activation
        # and gradient magnitudes, which vary by orders of magnitude across
        # models, layers and training states — an untrained network in eval mode
        # produces activations around 1e-8, a trained one values near unity. Any
        # fixed floor therefore rejects perfectly good maps on one model and
        # accepts noise on another. What genuinely means "nothing here" is every
        # channel contributing negatively, which puts the pre-ReLU maximum at or
        # below zero.
        if peak <= 0.0:
            return np.zeros(cam_array.shape, dtype=np.float32)

        cam_array = np.maximum(cam_array, 0.0) / peak
        return cam_array.astype(np.float32)


def overlay_heatmap(
    image_bgr: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a saliency map over the original image at its native size."""
    height, width = image_bgr.shape[:2]
    # Bilinear on purpose: the map is 7x7 and nearest-neighbour upscaling would
    # render it as blocks, implying a spatial precision the model does not have.
    resized = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
    coloured = cv2.applyColorMap((np.clip(resized, 0, 1) * 255).astype(np.uint8), colormap)
    return cv2.addWeighted(coloured, alpha, image_bgr, 1.0 - alpha, 0.0)


def encode_png_base64(image_bgr: np.ndarray, max_side: int = 512) -> str:
    """PNG-encode an image as base64 for embedding in a JSON response.

    Capped in size because the result travels inside the analysis payload, and
    a full-resolution overlay would dwarf every other field in it.
    """
    height, width = image_bgr.shape[:2]
    longest = max(height, width)
    if longest > max_side:
        scale = max_side / longest
        image_bgr = cv2.resize(
            image_bgr, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
        )

    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        return ""
    return base64.b64encode(buffer.tobytes()).decode("ascii")
