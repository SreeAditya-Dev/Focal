from __future__ import annotations

import math
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.db import get_db
from app.models import AnalysisRecord
from app.schemas import AnalysisResponse, HistoryItemSchema, HistoryListResponse, IssueSchema

router = APIRouter(prefix="/history", tags=["Analysis History"])


@router.get("", response_model=HistoryListResponse)
def get_history(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    quality_label: Optional[str] = Query(None, description="Filter by quality band"),
    search: Optional[str] = Query(None, description="Search by filename"),
    db: Session = Depends(get_db),
) -> HistoryListResponse:
    query = db.query(AnalysisRecord)

    if quality_label:
        query = query.filter(AnalysisRecord.quality_label == quality_label.upper())

    if search:
        query = query.filter(AnalysisRecord.filename.ilike(f"%{search}%"))

    total = query.count()
    total_pages = math.ceil(total / limit) if total > 0 else 1

    records = (
        query.order_by(desc(AnalysisRecord.created_at))
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    items: list[HistoryItemSchema] = []
    for r in records:
        issues_data = r.issues or []
        issue_types = [i.get("type", "unknown") for i in issues_data]
        items.append(
            HistoryItemSchema(
                id=r.id,
                filename=r.filename,
                file_size=r.file_size,
                width=r.width,
                height=r.height,
                quality_score=r.quality_score,
                quality_label=r.quality_label,
                issue_count=len(issues_data),
                issues_summary=issue_types,
                processing_time_ms=r.processing_time_ms,
                created_at=r.created_at,
            )
        )

    return HistoryListResponse(
        items=items,
        total=total,
        page=page,
        limit=limit,
        total_pages=total_pages,
    )


@router.get("/{record_id}", response_model=AnalysisResponse)
def get_history_detail(
    record_id: int,
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis record with ID {record_id} not found.",
        )

    issues_schemas = [IssueSchema(**i) for i in (record.issues or [])]

    return AnalysisResponse(
        id=record.id,
        filename=record.filename,
        width=record.width,
        height=record.height,
        file_size=record.file_size,
        quality_score=record.quality_score,
        quality_label=record.quality_label,
        issues=issues_schemas,
        stats=record.stats or {},
        summary=record.summary,
        model_version=record.model_version,
        model_loaded=True,
        processing_time_ms=record.processing_time_ms,
        timings_ms={},
        uncertainty=record.uncertainty,
        heatmap_base64=None,
        heatmap_issue=record.heatmap_issue,
        created_at=record.created_at,
    )


@router.delete("/{record_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_history_record(
    record_id: int,
    db: Session = Depends(get_db),
) -> None:
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis record with ID {record_id} not found.",
        )
    db.delete(record)
    db.commit()


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
def clear_all_history(
    db: Session = Depends(get_db),
) -> None:
    db.query(AnalysisRecord).delete()
    db.commit()
