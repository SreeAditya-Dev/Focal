from __future__ import annotations

import time
import logging
from typing import List
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import AnalysisRecord
from app.schemas import AnalysisResponse, BatchAnalysisResponse, IssueSchema
from app.services.analyzer import analyzer_service
from focal_ml.inference.predictor import ImageDecodeError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["Image Quality Analysis"])

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}


def _validate_image_file(file: UploadFile) -> None:
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        # Also check extension if content_type is generic
        ext = file.filename.split(".")[-1].lower() if file.filename else ""
        if ext not in {"jpg", "jpeg", "png", "webp", "bmp", "tiff"}:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file format: {file.content_type or ext}. Supported formats: JPEG, PNG, WebP, BMP, TIFF.",
            )


@router.post("", response_model=AnalysisResponse, status_code=status.HTTP_200_OK)
async def analyze_image(
    file: UploadFile = File(..., description="Image file to analyze"),
    include_heatmap: bool = Query(True, description="Generate Grad-CAM activation heatmap overlay"),
    uncertainty: bool = Query(True, description="Compute MC Dropout uncertainty estimate"),
    save_record: bool = Query(True, description="Persist analysis in database history"),
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    """Analyze a single uploaded image for blur, exposure, noise, compression artifacts, and defects."""
    _validate_image_file(file)

    try:
        content = await file.read()
    except Exception as e:
        logger.error("Failed to read uploaded file: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded image stream.",
        )

    file_size = len(content)
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if file_size > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB.",
        )

    try:
        result = analyzer_service.analyze_bytes(
            image_bytes=content,
            include_heatmap=include_heatmap,
            compute_uncertainty=uncertainty,
        )
    except ImageDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or invalid image encoding: {str(e)}",
        )
    except Exception as e:
        logger.exception("Unexpected error during analysis: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )

    issues_schemas = [
        IssueSchema(
            type=issue.type,
            severity=issue.severity,
            severity_score=issue.severity_score,
            confidence=issue.confidence,
            rule_confidence=issue.rule_confidence,
            cnn_confidence=issue.cnn_confidence,
            evidence=issue.evidence,
        )
        for issue in result.issues
    ]

    record_id = None
    created_at = None

    if save_record:
        try:
            record = AnalysisRecord(
                filename=file.filename or "uploaded_image.jpg",
                file_size=file_size,
                width=result.width,
                height=result.height,
                quality_score=result.quality_score,
                quality_label=result.quality_label,
                issues=[i.model_dump() for i in issues_schemas],
                stats=result.stats,
                summary=result.summary,
                model_version=result.model_version,
                processing_time_ms=result.processing_time_ms,
                uncertainty=result.uncertainty,
                heatmap_issue=result.heatmap_issue,
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            record_id = record.id
            created_at = record.created_at
        except Exception as e:
            logger.warning("Failed to save analysis record to database: %s", e)
            db.rollback()

    return AnalysisResponse(
        id=record_id,
        filename=file.filename or "uploaded_image.jpg",
        width=result.width,
        height=result.height,
        file_size=file_size,
        quality_score=result.quality_score,
        quality_label=result.quality_label,
        issues=issues_schemas,
        stats=result.stats,
        summary=result.summary,
        model_version=result.model_version,
        model_loaded=result.model_loaded,
        processing_time_ms=result.processing_time_ms,
        timings_ms=result.timings_ms,
        uncertainty=result.uncertainty,
        heatmap_base64=result.heatmap_base64,
        heatmap_issue=result.heatmap_issue,
        created_at=created_at,
    )


@router.post("/batch", response_model=BatchAnalysisResponse, status_code=status.HTTP_200_OK)
async def analyze_batch(
    files: List[UploadFile] = File(..., description="Batch of images to evaluate"),
    include_heatmap: bool = Query(False, description="Omit heatmap in batch to maximize throughput"),
    save_records: bool = Query(True, description="Persist all results in database"),
    db: Session = Depends(get_db),
) -> BatchAnalysisResponse:
    """Evaluate multiple images in a single batch request."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided in batch upload.",
        )

    if len(files) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch limit is 50 images per request.",
        )

    started = time.perf_counter()
    results: list[AnalysisResponse] = []
    successful = 0
    failed = 0

    for file in files:
        try:
            _validate_image_file(file)
            content = await file.read()
            if len(content) == 0:
                failed += 1
                continue

            result = analyzer_service.analyze_bytes(
                image_bytes=content,
                include_heatmap=include_heatmap,
                compute_uncertainty=False,
            )

            issues_schemas = [
                IssueSchema(
                    type=issue.type,
                    severity=issue.severity,
                    severity_score=issue.severity_score,
                    confidence=issue.confidence,
                    rule_confidence=issue.rule_confidence,
                    cnn_confidence=issue.cnn_confidence,
                    evidence=issue.evidence,
                )
                for issue in result.issues
            ]

            record_id = None
            created_at = None

            if save_records:
                try:
                    record = AnalysisRecord(
                        filename=file.filename or "batch_image.jpg",
                        file_size=len(content),
                        width=result.width,
                        height=result.height,
                        quality_score=result.quality_score,
                        quality_label=result.quality_label,
                        issues=[i.model_dump() for i in issues_schemas],
                        stats=result.stats,
                        summary=result.summary,
                        model_version=result.model_version,
                        processing_time_ms=result.processing_time_ms,
                        uncertainty=result.uncertainty,
                        heatmap_issue=result.heatmap_issue,
                    )
                    db.add(record)
                    db.commit()
                    db.refresh(record)
                    record_id = record.id
                    created_at = record.created_at
                except Exception:
                    db.rollback()

            results.append(
                AnalysisResponse(
                    id=record_id,
                    filename=file.filename or "batch_image.jpg",
                    width=result.width,
                    height=result.height,
                    file_size=len(content),
                    quality_score=result.quality_score,
                    quality_label=result.quality_label,
                    issues=issues_schemas,
                    stats=result.stats,
                    summary=result.summary,
                    model_version=result.model_version,
                    model_loaded=result.model_loaded,
                    processing_time_ms=result.processing_time_ms,
                    timings_ms=result.timings_ms,
                    uncertainty=result.uncertainty,
                    heatmap_base64=result.heatmap_base64,
                    heatmap_issue=result.heatmap_issue,
                    created_at=created_at,
                )
            )
            successful += 1
        except Exception as e:
            logger.warning("Error processing batch item %s: %s", file.filename, e)
            failed += 1

    total_time_ms = (time.perf_counter() - started) * 1000.0

    return BatchAnalysisResponse(
        total=len(files),
        successful=successful,
        failed=failed,
        results=results,
        total_processing_time_ms=round(total_time_ms, 2),
    )

