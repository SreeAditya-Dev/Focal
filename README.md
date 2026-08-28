# Focal — AI-Powered Image Quality & Defect Detection System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![React](https://img.shields.io/badge/React-18.3+-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7+-3178C6.svg?logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![Tailwind CSS](https://img.shields.io/badge/TailwindCSS-3.4+-38B2AC.svg?logo=tailwind-css&logoColor=white)](https://tailwindcss.com)

**Focal** is a production-grade full-stack artificial intelligence application that automatically inspects, scores, and diagnoses quality degradations and visual defects in digital images.

It combines a **47-feature classical computer vision extraction engine** (Laplacian variance, Tenengrad energy, Immerkaer noise, flattest-block noise floor, GLCM texture, JPEG blockiness) with a **MobileNetV3 convolutional neural network**, temperature-scaled calibration, and asymmetric Bayesian fusion.

---

## 🌟 Key Capabilities

- **Multi-Degradation Detection**: Detects **Blur**, **Overexposure**, **Underexposure**, **Image Noise**, **Compression Artifacts**, and **Localized Visual Defects** (smudges, scratches, sensor dust).
- **Explainable AI (XAI)**: Generates real-time **Grad-CAM attention heatmaps** overlaid directly on high-resolution images to localize defect regions.
- **Physical Heuristic Evidence**: Provides human-interpretable reasons grounded in physical radiometric measurements for every prediction.
- **Uncertainty Quantification**: Calculates **Monte-Carlo Dropout uncertainty** estimates across prediction passes.
- **Batch Processing**: Concurrently analyzes collections of images with summary throughput KPIs and CSV export.
- **Audit History Dashboard**: Fully persisted database history (SQLite / PostgreSQL) with filtering, searching, and detail inspection.

---

## 📐 Architecture Overview

```
                          ┌───────────────────────────┐
                          │    Uploaded Image File    │
                          └─────────────┬─────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
  ┌─────────────────────────────┐               ┌─────────────────────────────┐
  │   47 Classical CV Features  │               │   MobileNetV3-Small (CNN)   │
  │   (Laplacian, FFT, Noise,   │               │   (224x224 RGB, Multi-Task  │
  │    Clipping, Blockiness)    │               │    Presence & Severity)     │
  └──────────────┬──────────────┘               └──────────────┬──────────────┘
                 │                                             │
                 ▼                                             ▼
  ┌─────────────────────────────┐               ┌─────────────────────────────┐
  │ Deterministic Heuristic     │               │ Temperature Scaling         │
  │ Ramps & Decision Rules      │               │ Confidence Calibration      │
  └──────────────┬──────────────┘               └──────────────┬──────────────┘
                 │                                             │
                 └──────────────────────┬──────────────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │  Asymmetric Bayesian Fusion │
                         │   (Physical Anchor Bounds)  │
                         └──────────────┬──────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
  ┌─────────────────────────────┐               ┌─────────────────────────────┐
  │   Continuous Quality Score  │               │   Detected Issues, Evidence │
  │   (0-100) & Quality Band    │               │   & Grad-CAM Heatmap Map    │
  └─────────────────────────────┘               └─────────────────────────────┘
```

---

## 🚀 Quick Start (Local Development)

### 1. Prerequisites
- **Python 3.10+** (with `uv` installed: `pip install uv`)
- **Node.js 18+** & `npm`

### 2. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Install dependencies and local ML pipeline
uv sync --extra dev

# Run FastAPI development server (runs at http://localhost:8000)
uv run uvicorn app.main:app --reload --port 8000
```
Interactive Swagger API documentation will be available at [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs).

### 3. Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install packages
npm install

# Start Vite development server (runs at http://localhost:5173)
npm run dev
```

---

## 🐳 Docker Deployment

To launch the full stack (PostgreSQL database, FastAPI backend, and React Nginx frontend) in one command:

```bash
# Start all containers
docker compose up --build -d
```
- **Frontend App**: [http://localhost:3000](http://localhost:3000)
- **Backend API**: [http://localhost:8000](http://localhost:8000)
- **Health Check**: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

## 🧪 Automated Testing

### Backend & API Tests
```bash
cd backend
uv run --extra dev pytest app/tests/test_api.py -v
```

### ML Pipeline Unit & Contract Tests (111 Tests)
```bash
cd ml_pipeline
uv run --extra train --with pytest pytest tests/ -v
```

### Frontend Typecheck & Build
```bash
cd frontend
npm run build
```

---

## 📊 Held-Out Test Evaluation Benchmark (3,130 Images)

| Degradation Category | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Blur / Insufficient Sharpness** | **0.968** | **0.974** | **0.971** | **0.998** | **0.993** |
| **Overexposure (Highlights)** | **0.841** | **0.926** | **0.881** | **0.990** | **0.945** |
| **Sensor Noise / Grain** | **0.782** | **0.964** | **0.864** | **0.987** | **0.935** |
| **Underexposure (Shadows)** | 0.603 | 0.862 | 0.710 | **0.958** | 0.835 |
| **Compression Artifacts** | 0.595 | **0.898** | 0.716 | **0.946** | 0.865 |
| **Localized Visual Defects** | 0.246 | 0.845 | 0.381 | 0.792 | 0.410 |
| **MACRO TOTAL** | **0.672** | **0.912** | **0.754** | **0.945** | **0.830** |

---

## 📜 API Reference

- `POST /api/v1/analyze`: Analyze a single image (`multipart/form-data`) with optional Grad-CAM heatmap generation.
- `POST /api/v1/analyze/batch`: Multi-image batch processing.
- `GET /api/v1/history`: Paginated analysis history with filtering and search.
- `GET /api/v1/history/{id}`: Detailed record lookup.
- `DELETE /api/v1/history/{id}`: Delete an analysis entry.
- `GET /api/v1/health`: Live health and model load check.
