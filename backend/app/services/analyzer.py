from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
from focal_ml.inference.predictor import FocalPredictor, ImageDecodeError, AnalysisResult
from app.config import settings

logger = logging.getLogger(__name__)


class AnalyzerService:
    """Manages the lifecycle of FocalPredictor and wraps analysis operations."""
    
    _instance: Optional[AnalyzerService] = None
    
    def __init__(self) -> None:
        self.predictor: Optional[FocalPredictor] = None
        self._initialize()

    @classmethod
    def get_instance(cls) -> AnalyzerService:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _initialize(self) -> None:
        """Load the model and rules if available."""
        model_path = Path(settings.MODEL_PATH)
        rules_path = Path(settings.RULES_PATH)
        calibration_path = Path(settings.CALIBRATION_PATH)
        
        logger.info(
            "Initializing FocalPredictor (model: %s, rules: %s, calib: %s)",
            model_path,
            rules_path,
            calibration_path,
        )
        
        try:
            self.predictor = FocalPredictor(
                model_path=model_path if model_path.exists() else None,
                rules_path=rules_path if rules_path.exists() else None,
                calibration_path=calibration_path if calibration_path.exists() else None,
                device=settings.DEVICE,
            )
            logger.info(
                "FocalPredictor loaded successfully (model_loaded=%s, version=%s)",
                self.predictor.model_loaded,
                self.predictor.model_version,
            )
        except Exception as e:
            logger.exception("Failed to initialize FocalPredictor with full weights; falling back to rules-only: %s", e)
            self.predictor = FocalPredictor(
                rules_path=rules_path if rules_path.exists() else None,
                device="cpu",
            )

    def analyze_bytes(
        self,
        image_bytes: bytes,
        include_heatmap: bool = True,
        compute_uncertainty: bool = True,
    ) -> AnalysisResult:
        """Run full hybrid quality and defect analysis on raw image bytes."""
        if self.predictor is None:
            self._initialize()
            
        return self.predictor.analyse(
            image=image_bytes,
            raw_bytes=image_bytes,
            include_heatmap=include_heatmap,
            uncertainty=compute_uncertainty,
        )


analyzer_service = AnalyzerService.get_instance()

