from __future__ import annotations

import datetime
from sqlalchemy import Column, Integer, Float, String, Text, DateTime, JSON
from app.db import Base


class AnalysisRecord(Base):
    """Database model for storing image quality analysis results."""
    __tablename__ = "analysis_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    filename = Column(String(255), nullable=False, index=True)
    file_size = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    
    # Core Quality Assessment
    quality_score = Column(Float, nullable=False, index=True)
    quality_label = Column(String(50), nullable=False, index=True)
    
    # Detailed Analysis JSON
    issues = Column(JSON, nullable=False, default=list)
    stats = Column(JSON, nullable=False, default=dict)
    summary = Column(Text, nullable=False)
    
    # Model & Timing Metadata
    model_version = Column(String(100), nullable=False)
    processing_time_ms = Column(Float, nullable=False)
    uncertainty = Column(JSON, nullable=True)
    heatmap_issue = Column(String(50), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False, index=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "file_size": self.file_size,
            "width": self.width,
            "height": self.height,
            "quality_score": self.quality_score,
            "quality_label": self.quality_label,
            "issues": self.issues,
            "stats": self.stats,
            "summary": self.summary,
            "model_version": self.model_version,
            "processing_time_ms": self.processing_time_ms,
            "uncertainty": self.uncertainty,
            "heatmap_issue": self.heatmap_issue,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

