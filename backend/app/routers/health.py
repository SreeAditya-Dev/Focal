from __future__ import annotations

import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.config import settings
from app.db import get_db
from app.schemas import HealthResponse
from app.services.analyzer import analyzer_service

router = APIRouter(tags=["Health & Status"])


@router.get("/health", response_model=HealthResponse)
def get_health(db: Session = Depends(get_db)) -> HealthResponse:
    db_status = "ok"
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_status = "error"

    predictor = analyzer_service.predictor
    model_loaded = predictor.model_loaded if predictor else False
    model_version = predictor.model_version if predictor else "unavailable"

    return HealthResponse(
        status="ok" if db_status == "ok" else "degraded",
        version=settings.VERSION,
        model_version=model_version,
        model_loaded=model_loaded,
        device=settings.DEVICE,
        database=db_status,
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )
