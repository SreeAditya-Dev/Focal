from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Focal — AI Image Quality & Defect Detection"
    VERSION: str = "1.0.0"
    
    # Database
    DATABASE_URL: str = "sqlite:///./focal.db"
    
    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*",
    ]
    
    # Uploads & Model Artifacts
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE_MB: int = 25
    
    # Default search paths for model artifacts
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    MODEL_PATH: str = str(BASE_DIR / "ml_pipeline" / "models" / "focal_cnn_v1.pt")
    RULES_PATH: str = str(BASE_DIR / "ml_pipeline" / "models" / "rules_v1.json")
    CALIBRATION_PATH: str = str(BASE_DIR / "ml_pipeline" / "models" / "calibration_v1.json")
    FUSION_WEIGHTS_PATH: str = str(BASE_DIR / "ml_pipeline" / "models" / "fusion_weights.json")
    DEVICE: str = "cpu"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

