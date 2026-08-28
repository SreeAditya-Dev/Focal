from __future__ import annotations

import datetime
from sqlalchemy import Column, Integer, Float, String, Text, DateTime, JSON
from app.db import Base


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    filename = Column(String(255), nullable=False, index=True)
    file_size = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    quality_score = Column(Float, nullable=False, index=True)
    quality_label = Column(String(50), nullable=False, index=True)
    issues = Column(JSON, nullable=False, default=list)
    stats = Column(JSON, nullable=False, default=dict)
    summary = Column(Text, nullable=False)
    model_version = Column(String(100), nullable=False)
    processing_time_ms = Column(Float, nullable=False)
    uncertainty = Column(JSON, nullable=True)
    heatmap_issue = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False, index=True)
