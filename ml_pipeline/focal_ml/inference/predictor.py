"""End-to-end inference: bytes in, analysis out.

``FocalPredictor`` is the only thing the backend imports from the ML package.
It owns the whole path — decode, canonical resize, classical features, CNN
forward, calibration, rule evaluation, fusion, and optionally a Grad-CAM
overlay and an uncertainty estimate.

It is built to be constructed once at process start and called concurrently:
the model is held in eval mode, inference runs under ``no_grad`` except for the
Grad-CAM pass, and nothing per-request is stored on the instance.

**Degrading rather than failing.** If no checkpoint is available the predictor
still works, running the rule layer alone and reporting ``model_loaded: false``.
A quality analyser that returns a slightly worse answer is far more useful than
one that returns a 500, and it means the API can boot and serve before a model
has ever been trained.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from focal_ml.constants import CANONICAL_LONG_SIDE, ISSUE_TYPES
from focal_ml.features import REPORTED_FEATURES, extract_features
from focal_ml.features.extract_all import FEATURE_NAMES
from focal_ml.fusion.rules import RuleConfig, evaluate_rules
from focal_ml.fusion.scorer import DetectedIssue, fuse, summarise
from focal_ml.model.calibration import Calibration, mc_dropout_uncertainty
from focal_ml.model.preprocessing import preprocess_for_cnn
from focal_ml.utils import resize_long_side, to_gray


class ImageDecodeError(ValueError):
    """The bytes given are not a decodable image.

    Distinct from an image that decodes and merely looks damaged — that is the
    `corruption` class, which is a normal analysis result, not an error.
    """


@dataclass
class AnalysisResult:
    quality_score: float
    quality_label: str
    issues: list[DetectedIssue]
    stats: dict[str, float]
    summary: str
    model_version: str
    model_loaded: bool
    width: int
    height: int
    processing_time_ms: float
    timings_ms: dict[str, float] = field(default_factory=dict)
    heatmap_base64: str | None = None
    heatmap_issue: str | None = None
    uncertainty: list[dict] | None = None

    def to_dict(self) -> dict:
        payload = {
            "quality_score": self.quality_score,
            "quality_label": self.quality_label,
            "issues": [issue.to_dict() for issue in self.issues],
            "stats": self.stats,
            "summary": self.summary,
            "model_version": self.model_version,
            "model_loaded": self.model_loaded,
            "width": self.width,
            "height": self.height,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
        }
        if self.heatmap_base64 is not None:
            payload["heatmap_base64"] = self.heatmap_base64
            payload["heatmap_issue"] = self.heatmap_issue
        if self.uncertainty is not None:
            payload["uncertainty"] = self.uncertainty
        return payload


class FocalPredictor:
    """Loads the model once and analyses images."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        calibration_path: str | Path | None = None,
        rules_path: str | Path | None = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = None
        self.model_version = "rules-only"
        self.checkpoint_features: list[str] = list(FEATURE_NAMES)
        self._gradcam = None

        self.rules = RuleConfig.from_json(rules_path) if rules_path and Path(rules_path).exists() else RuleConfig()
        self.calibration = (
            Calibration.from_json(calibration_path)
            if calibration_path and Path(calibration_path).exists()
            else Calibration.identity()
        )

        if model_path and Path(model_path).exists():
            self._load_model(Path(model_path))

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_model(self, path: Path) -> None:
        from focal_ml.model.architecture import FocalNet, ModelConfig

        # weights_only=True refuses to unpickle arbitrary objects. The
        # checkpoint is ours, but this code path is reachable from a deployment
        # where MODEL_PATH is operator-configurable.
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        saved_features = checkpoint.get("feature_names", list(FEATURE_NAMES))
        if saved_features != list(FEATURE_NAMES):
            # The feature encoder's first layer is indexed positionally. Loading
            # weights trained against a different feature ordering would run
            # perfectly and silently read every input from the wrong column.
            raise ValueError(
                f"checkpoint {path.name} was trained on a different feature set "
                f"({len(saved_features)} features) than this build defines "
                f"({len(FEATURE_NAMES)}). Retrain, or check out the matching commit."
            )

        config = ModelConfig(**checkpoint["model_config"])
        # Never fetch ImageNet weights at load time: they are about to be
        # overwritten by the checkpoint, and a container without egress would
        # hang or crash on startup trying.
        config.pretrained = False

        model = FocalNet(config)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval().to(self.device)

        self.model = model
        self.model_version = checkpoint.get("version", path.stem)
        self.checkpoint_features = saved_features

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    def warmup(self) -> float:
        """Run one dummy analysis so the first real request is not the slowest.

        Lazy allocations inside torch and OpenCV, plus any first-call kernel
        selection, otherwise land on whoever arrives first after a deploy.
        """
        started = time.perf_counter()
        dummy = np.random.default_rng(0).integers(0, 255, (256, 384, 3), dtype=np.uint8)
        self.analyse(dummy, include_heatmap=self.model_loaded)
        return (time.perf_counter() - started) * 1000.0

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def decode(self, data: bytes) -> np.ndarray:
        """Decode image bytes to BGR, raising ``ImageDecodeError`` if unreadable."""
        import cv2

        if not data:
            raise ImageDecodeError("empty file")
        buffer = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ImageDecodeError("file is not a decodable image")
        if image.size == 0 or min(image.shape[:2]) < 8:
            raise ImageDecodeError(f"image too small to analyse ({image.shape[1]}x{image.shape[0]})")
        return image

    def analyse(
        self,
        image: np.ndarray | bytes,
        raw_bytes: bytes | None = None,
        include_heatmap: bool = True,
        uncertainty: bool = False,
        uncertainty_passes: int = 20,
    ) -> AnalysisResult:
        """Analyse one image.

        Args:
            image: decoded BGR array, or the encoded bytes.
            raw_bytes: the original file, used for the byte-entropy feature.
            include_heatmap: render a Grad-CAM overlay for the top issue.
            uncertainty: run Monte-Carlo dropout passes. Off by default; it
                multiplies the model's share of the latency by the pass count.
        """
        started = time.perf_counter()
        timings: dict[str, float] = {}

        if isinstance(image, (bytes, bytearray)):
            raw_bytes = raw_bytes or bytes(image)
            mark = time.perf_counter()
            image = self.decode(raw_bytes)
            timings["decode"] = (time.perf_counter() - mark) * 1000.0

        original_height, original_width = image.shape[:2]

        mark = time.perf_counter()
        canonical = resize_long_side(image, CANONICAL_LONG_SIDE)
        timings["resize"] = (time.perf_counter() - mark) * 1000.0

        mark = time.perf_counter()
        features = extract_features(canonical, raw_bytes, already_canonical=True)
        timings["features"] = (time.perf_counter() - mark) * 1000.0

        mark = time.perf_counter()
        rule_outcomes = evaluate_rules(features, self.rules)
        timings["rules"] = (time.perf_counter() - mark) * 1000.0

        cnn_presence = cnn_severity = None
        image_tensor = feature_tensor = None

        if self.model is not None:
            mark = time.perf_counter()
            image_tensor, feature_tensor = self._to_tensors(canonical, features)
            with torch.no_grad():
                output = self.model(
                    image=image_tensor if self.model.config.use_image else None,
                    features=feature_tensor if self.model.config.use_features else None,
                )
            # Calibration is applied to the logits, before anything reads a
            # probability — the rest of the pipeline must never see an
            # uncalibrated confidence.
            probabilities = self.calibration.apply(output["presence_logits"].cpu().numpy())[0]
            severities = torch.sigmoid(output["severity_logits"]).cpu().numpy()[0]
            cnn_presence = {issue: float(probabilities[i]) for i, issue in enumerate(ISSUE_TYPES)}
            cnn_severity = {issue: float(severities[i]) for i, issue in enumerate(ISSUE_TYPES)}
            timings["model"] = (time.perf_counter() - mark) * 1000.0

        mark = time.perf_counter()
        fused = fuse(rule_outcomes, cnn_presence, cnn_severity)
        timings["fusion"] = (time.perf_counter() - mark) * 1000.0

        result = AnalysisResult(
            quality_score=fused.quality_score,
            quality_label=fused.quality_label,
            issues=fused.issues,
            stats={name: round(features[name], 4) for name in REPORTED_FEATURES},
            summary=summarise(fused),
            model_version=self.model_version,
            model_loaded=self.model_loaded,
            width=original_width,
            height=original_height,
            processing_time_ms=0.0,
        )

        if include_heatmap and self.model is not None and fused.issues:
            mark = time.perf_counter()
            result.heatmap_base64, result.heatmap_issue = self._heatmap(
                canonical, image_tensor, feature_tensor, fused.issues[0].type
            )
            timings["heatmap"] = (time.perf_counter() - mark) * 1000.0

        if uncertainty and self.model is not None:
            mark = time.perf_counter()
            estimates = mc_dropout_uncertainty(
                self.model,
                image_tensor if self.model.config.use_image else None,
                feature_tensor if self.model.config.use_features else None,
                passes=uncertainty_passes,
                calibration=self.calibration,
            )
            result.uncertainty = [
                {"issue": e.issue, "mean": e.mean, "std": e.std, "flagged": e.flagged}
                for e in estimates
            ]
            timings["uncertainty"] = (time.perf_counter() - mark) * 1000.0

        result.processing_time_ms = (time.perf_counter() - started) * 1000.0
        result.timings_ms = timings
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_tensors(
        self, canonical: np.ndarray, features: dict[str, float]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_tensor = torch.from_numpy(preprocess_for_cnn(canonical)).unsqueeze(0).to(self.device)
        vector = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)
        return image_tensor, torch.from_numpy(vector).unsqueeze(0).to(self.device)

    def _heatmap(
        self,
        canonical: np.ndarray,
        image_tensor: torch.Tensor,
        feature_tensor: torch.Tensor,
        issue: str,
    ) -> tuple[str | None, str | None]:
        from focal_ml.model.gradcam import GradCAM, encode_png_base64, overlay_heatmap

        if not self.model.config.use_image:
            return None, None
        try:
            if self._gradcam is None:
                # Built once and kept: registering hooks per request leaks them.
                self._gradcam = GradCAM(self.model)
            cam = self._gradcam.generate(image_tensor, issue, feature_tensor)
            if float(cam.max()) <= 0.0:
                return None, None
            return encode_png_base64(overlay_heatmap(canonical, cam)), issue
        except (RuntimeError, ValueError, KeyError):
            # An explanation is a nice-to-have; never fail an analysis over one.
            return None, None

    def health(self) -> dict:
        return {
            "model_loaded": self.model_loaded,
            "model_version": self.model_version,
            "calibration_version": self.calibration.version,
            "rules_version": self.rules.version,
            "device": str(self.device),
            "n_features": len(FEATURE_NAMES),
            "issues": list(ISSUE_TYPES),
        }
