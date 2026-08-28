from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db import init_db
from app.routers import analyze, health, history
from app.services.analyzer import analyzer_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("focal_backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database schema...")
    init_db()
    
    logger.info("Pre-warming Focal AI analyzer service...")
    try:
        analyzer_service._initialize()
    except Exception as e:
        logger.warning("Analyzer pre-warming issue: %s", e)
        
    yield
    logger.info("Shutting down Focal backend...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Full-stack AI API for detecting blur, exposure, noise, compression artifacts, and visual defects.",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix=settings.API_V1_STR)
app.include_router(analyze.router, prefix=settings.API_V1_STR)
app.include_router(history.router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Root"])
def root():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": f"{settings.API_V1_STR}/docs",
        "health_url": f"{settings.API_V1_STR}/health",
    }
