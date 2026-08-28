from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
from focal_ml.inference.predictor import FocalPredictor, ImageDecodeError, AnalysisResult
from app.config import settings

logger = logging.getLogger(__name__)


class AnalyzerService:
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
        model_path = Path(settings.MODEL_PATH)
        rules_path = Path(settings.RULES_PATH)
        calibration_path = Path(settings.CALIBRATION_PATH)
        
        try:
            self.predictor = FocalPredictor(
                model_path=model_path if model_path.exists() else None,
                rules_path=rules_path if rules_path.exists() else None,
                calibration_path=calibration_path if calibration_path.exists() else None,
                device=settings.DEVICE,
            )
        except Exception as e:
            logger.warning("FocalPredictor full initialization fallback: %s", e)
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
        if self.predictor is None:
            self._initialize()
            
        return self.predictor.analyse(
            image=image_bytes,
            raw_bytes=image_bytes,
            include_heatmap=include_heatmap,
            uncertainty=compute_uncertainty,
        )


analyzer_service = AnalyzerService.get_instance()
