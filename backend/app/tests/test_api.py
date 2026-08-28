from __future__ import annotations

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.db import init_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_db():
    init_db()


def _create_sample_jpeg() -> bytes:
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"ok", "degraded"}
    assert "version" in data
    assert "model_version" in data


def test_analyze_valid_image():
    image_bytes = _create_sample_jpeg()
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("test_image.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "quality_score" in data
    assert 0 <= data["quality_score"] <= 100
    assert data["quality_label"] in {"EXCELLENT", "ACCEPTABLE", "POOR", "UNUSABLE"}
    assert "issues" in data
    assert "stats" in data
    assert "summary" in data
    assert data["width"] == 256
    assert data["height"] == 256


def test_analyze_invalid_file():
    corrupt_bytes = b"this is not an image file"
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("bad_file.jpg", corrupt_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Corrupted or invalid image" in response.json()["detail"]


def test_analyze_unsupported_media_type():
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("text.txt", b"plain text", "text/plain")},
    )
    assert response.status_code == 415


def test_history_workflow():
    # Submit an image
    image_bytes = _create_sample_jpeg()
    post_res = client.post(
        "/api/v1/analyze",
        files={"file": ("history_sample.jpg", image_bytes, "image/jpeg")},
        params={"save_record": "true"},
    )
    assert post_res.status_code == 200
    record_id = post_res.json()["id"]
    assert record_id is not None

    # Query history list
    list_res = client.get("/api/v1/history")
    assert list_res.status_code == 200
    history_data = list_res.json()
    assert history_data["total"] >= 1
    assert any(item["id"] == record_id for item in history_data["items"])

    # Query history detail
    detail_res = client.get(f"/api/v1/history/{record_id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["filename"] == "history_sample.jpg"

    # Delete history record
    del_res = client.delete(f"/api/v1/history/{record_id}")
    assert del_res.status_code == 204

    # Verify deleted
    verify_res = client.get(f"/api/v1/history/{record_id}")
    assert verify_res.status_code == 404

