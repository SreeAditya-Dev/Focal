from __future__ import annotations

import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class IssueSchema(BaseModel):
    type: str = Field(..., description="Degradation type (blur, noise, overexposure, underexposure, corruption, defect)")
    severity: str = Field(..., description="Severity category (low, medium, high, severe)")
    severity_score: float = Field(..., description="Continuous severity score 0.0 to 1.0")
    confidence: float = Field(..., description="Fused confidence score 0.0 to 1.0")
    rule_confidence: float = Field(..., description="Rule heuristic confidence")
    cnn_confidence: float = Field(..., description="Neural network confidence")
    evidence: list[str] = Field(default_factory=list, description="Human-interpretable reasons")

    model_config = ConfigDict(from_attributes=True)


class AnalysisResponse(BaseModel):
    id: int | None = Field(None, description="Database record ID if persisted")
    filename: str
    width: int
    height: int
    file_size: int
    quality_score: float = Field(..., description="Overall quality score (0 to 100)")
    quality_label: str = Field(..., description="EXCELLENT, ACCEPTABLE, POOR, or UNUSABLE")
    issues: list[IssueSchema] = Field(default_factory=list)
    stats: dict[str, float] = Field(default_factory=dict, description="47 classical feature measurements")
    summary: str = Field(..., description="Natural language analysis explanation")
    model_version: str
    model_loaded: bool = True
    processing_time_ms: float
    timings_ms: dict[str, float] = Field(default_factory=dict)
    uncertainty: Any = None
    heatmap_base64: str | None = None
    heatmap_issue: str | None = None
    created_at: datetime.datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class BatchAnalysisResponse(BaseModel):
    total: int
    successful: int
    failed: int
    results: list[AnalysisResponse]
    total_processing_time_ms: float


class HistoryItemSchema(BaseModel):
    id: int
    filename: str
    file_size: int
    width: int
    height: int
    quality_score: float
    quality_label: str
    issue_count: int
    issues_summary: list[str]
    processing_time_ms: float
    created_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)


class HistoryListResponse(BaseModel):
    items: list[HistoryItemSchema]
    total: int
    page: int
    limit: int
    total_pages: int


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    model_version: str
    model_loaded: bool
    device: str
    database: str
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

