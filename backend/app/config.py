from __future__ import annotations

import os
from pathlib import Path
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Focal — AI Image Quality & Defect Detection"
    VERSION: str = "1.0.0"
    
    # Base Directory
    BASE_DIR: Path = BASE_DIR
    
    # Database
    DATABASE_URL: str = "sqlite:///./focal.db"
    PGHOST: str | None = None
    PGDATABASE: str | None = None
    PGUSER: str | None = None
    PGPASSWORD: str | None = None
    PGPORT: int = 5432
    PGSSLMODE: str | None = None
    PGCHANNELBINDING: str | None = None
    
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
    MODEL_PATH: str = str(BASE_DIR / "ml_pipeline" / "models" / "focal_cnn_v1.pt")
    RULES_PATH: str = str(BASE_DIR / "ml_pipeline" / "models" / "rules_v1.json")
    CALIBRATION_PATH: str = str(BASE_DIR / "ml_pipeline" / "models" / "calibration_v1.json")
    FUSION_WEIGHTS_PATH: str = str(BASE_DIR / "ml_pipeline" / "models" / "fusion_weights.json")
    DEVICE: str = "cpu"
    
    model_config = SettingsConfigDict(
        env_file=[
            str(BASE_DIR / ".env"),
            str(BASE_DIR / "backend" / ".env"),
            ".env",
        ],
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def assemble_db_connection(self) -> "Settings":
        if self.PGHOST and (self.DATABASE_URL == "sqlite:///./focal.db" or not self.DATABASE_URL):
            user = self.PGUSER or "postgres"
            pwd = f":{self.PGPASSWORD}" if self.PGPASSWORD else ""
            db = self.PGDATABASE or "postgres"
            ssl = f"?sslmode={self.PGSSLMODE}" if self.PGSSLMODE else ""
            self.DATABASE_URL = f"postgresql://{user}{pwd}@{self.PGHOST}:{self.PGPORT}/{db}{ssl}"
        return self


settings = Settings()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
