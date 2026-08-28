FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and image decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy ml_pipeline and backend code
COPY ml_pipeline /app/ml_pipeline
COPY backend /app/backend

# Install dependencies and local packages
WORKDIR /app/backend
RUN uv pip install --system --no-cache -e /app/ml_pipeline
RUN uv pip install --system --no-cache -e ".[postgres]"

ENV PYTHONPATH=/app/backend:/app/ml_pipeline
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

