"""
FastAPI Inference Service
==========================
Endpoints:
  POST /predict         — accepts a telemetry window, returns anomaly score + LSTM prediction
  POST /analyze-logs    — accepts log text, returns root-cause classification
  GET  /model-info      — returns model version and evaluation metrics

Run locally:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

Or via Docker:
    docker-compose up
"""

import os
import json
import time
import logging
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("5g-api")

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_DIR = os.environ.get("MODEL_DIR", "outputs/models")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
KNN_PATH = os.path.join(MODEL_DIR, "knn_classifier.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "model_comparison.csv")

WINDOW_SIZE = 30
DEVICE = torch.device("cpu")  # Inference always on CPU in the container

# ─── LSTM Model Architecture (must match training) ────────────────────────────


class LSTMAnomalyDetector(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(),
            nn.Dropout(dropout / 2), nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


# ─── Model Registry (loaded at startup) ───────────────────────────────────────

_registry: dict = {}


def _load_models() -> None:
    """Load all models and artefacts into _registry on startup."""
    global _registry

    # LSTM
    if Path(LSTM_MODEL_PATH).exists():
        ckpt = torch.load(LSTM_MODEL_PATH, map_location=DEVICE, weights_only=False)
        hparams = ckpt["hparams"]
        model = LSTMAnomalyDetector(
            input_size=hparams["input_size"],
            hidden_size=hparams["hidden_size"],
            num_layers=hparams["num_layers"],
            dropout=hparams["dropout"],
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _registry["lstm"] = model
        _registry["lstm_threshold"] = ckpt.get("threshold", 0.5)
        _registry["lstm_metrics"] = ckpt.get("metrics", {})
        _registry["lstm_hparams"] = hparams
        logger.info(f"LSTM loaded — threshold={_registry['lstm_threshold']:.3f}")
    else:
        logger.warning(f"LSTM model not found at {LSTM_MODEL_PATH}")
        _registry["lstm"] = None

    # Scaler
    if Path(SCALER_PATH).exists():
        _registry["scaler"] = joblib.load(SCALER_PATH)
        logger.info("Scaler loaded")
    else:
        logger.warning(f"Scaler not found at {SCALER_PATH}")
        _registry["scaler"] = None

    # KNN + LabelEncoder
    if Path(KNN_PATH).exists():
        knn_bundle = joblib.load(KNN_PATH)
        _registry["knn"] = knn_bundle["knn"]
        _registry["le"] = knn_bundle["le"]
        logger.info("KNN classifier loaded")
    else:
        logger.warning(f"KNN not found at {KNN_PATH}")
        _registry["knn"] = None
        _registry["le"] = None

    # sentence-transformer (lazy-loaded to speed startup)
    _registry["encoder"] = None


def _get_encoder():
    """Lazy-load sentence transformer."""
    if _registry.get("encoder") is None:
        try:
            from sentence_transformers import SentenceTransformer
            _registry["encoder"] = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Sentence transformer loaded")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
    return _registry.get("encoder")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="5G Network Anomaly Detection API",
    description=(
        "Real-time anomaly detection for 5G network telemetry using LSTM "
        "and root-cause classification of network error logs."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    _load_models()
    logger.info("5G Anomaly Detection API ready")


# ─── Request / Response Schemas ───────────────────────────────────────────────


class TelemetryWindow(BaseModel):
    """
    A sliding window of telemetry feature vectors.
    Each inner list is one timestep with all engineered features in order.
    """
    features: list[list[float]] = Field(
        ...,
        description=(
            f"Sequence of {WINDOW_SIZE} timestep feature vectors. "
            "Each vector must have the same length as the scaler was fitted on."
        ),
        min_length=1,
    )


class PredictionResponse(BaseModel):
    anomaly_score: float = Field(..., description="LSTM sigmoid probability of anomaly [0, 1]")
    is_anomaly: bool = Field(..., description="Binary anomaly flag using optimised threshold")
    threshold: float = Field(..., description="Decision threshold used")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class LogAnalysisRequest(BaseModel):
    log_text: str = Field(..., description="Raw log line or free-text description")
    top_k: int = Field(3, ge=1, le=10, description="Number of similar logs to retrieve")


class LogAnalysisResponse(BaseModel):
    predicted_category: str = Field(..., description="Root-cause category")
    confidence: float = Field(..., description="KNN vote fraction [0, 1]")
    similar_logs: list[dict] = Field(..., description="Top-k similar logs from ChromaDB")
    latency_ms: float


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    threshold: float
    metrics: dict
    feature_count: int
    window_size: int


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.post("/predict", response_model=PredictionResponse, summary="Predict anomaly from telemetry")
async def predict(window: TelemetryWindow):
    """
    Accepts a sliding window of telemetry feature vectors and returns:
    - anomaly_score : float in [0, 1]
    - is_anomaly    : bool
    - threshold     : decision threshold
    - latency_ms    : inference time
    """
    t0 = time.perf_counter()

    lstm = _registry.get("lstm")
    scaler = _registry.get("scaler")

    if lstm is None:
        raise HTTPException(status_code=503, detail="LSTM model not loaded")

    # Validate window shape
    features = window.features
    n_timesteps = len(features)
    if n_timesteps < WINDOW_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Expected at least {WINDOW_SIZE} timesteps, got {n_timesteps}",
        )

    # Use last WINDOW_SIZE timesteps
    arr = np.array(features[-WINDOW_SIZE:], dtype=np.float32)  # (30, n_features)

    if scaler is not None:
        # Reshape for scaler: (30 * n_features,) → scale → reshape back
        n_feat = arr.shape[1]
        arr_flat = arr.reshape(-1, n_feat)
        arr_flat = scaler.transform(arr_flat)
        arr = arr_flat.reshape(WINDOW_SIZE, n_feat)

    x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, 30, n_feat)

    with torch.no_grad():
        logit = lstm(x)
        score = float(torch.sigmoid(logit).item())

    threshold = _registry.get("lstm_threshold", 0.5)
    latency_ms = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        anomaly_score=score,
        is_anomaly=score >= threshold,
        threshold=threshold,
        latency_ms=round(latency_ms, 3),
    )


@app.post("/analyze-logs", response_model=LogAnalysisResponse,
          summary="Root-cause classify a log entry")
async def analyze_logs(request: LogAnalysisRequest):
    """
    Accepts a log text and returns:
    - predicted_category : root-cause label
    - confidence         : KNN vote fraction
    - similar_logs       : semantically similar historical entries
    - latency_ms         : inference time
    """
    t0 = time.perf_counter()

    encoder = _get_encoder()
    knn = _registry.get("knn")
    le = _registry.get("le")

    if encoder is None:
        raise HTTPException(status_code=503, detail="Sentence transformer not available")
    if knn is None:
        raise HTTPException(status_code=503, detail="KNN classifier not loaded")

    # Embed the query
    emb = encoder.encode([request.log_text], convert_to_numpy=True)  # (1, dim)

    # KNN predict with vote confidence
    distances, indices = knn.kneighbors(emb, n_neighbors=min(knn.n_neighbors, 10))
    neighbor_labels = knn._y[indices[0]]
    unique, counts = np.unique(neighbor_labels, return_counts=True)
    best_idx = np.argmax(counts)
    predicted_label = int(unique[best_idx])
    confidence = float(counts[best_idx]) / len(neighbor_labels)
    predicted_category = le.inverse_transform([predicted_label])[0]

    # ChromaDB semantic search (if available)
    similar_logs = []
    try:
        import chromadb
        client = chromadb.PersistentClient(path="outputs/chromadb")
        collection = client.get_collection("network_logs")
        results = collection.query(
            query_embeddings=emb.tolist(),
            n_results=request.top_k,
        )
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similar_logs.append({"document": doc[:200], "metadata": meta,
                                   "similarity": round(1 - dist, 4)})
    except Exception as e:
        logger.warning(f"ChromaDB query failed: {e}")

    latency_ms = (time.perf_counter() - t0) * 1000
    return LogAnalysisResponse(
        predicted_category=predicted_category,
        confidence=round(confidence, 4),
        similar_logs=similar_logs,
        latency_ms=round(latency_ms, 3),
    )


@app.get("/model-info", response_model=ModelInfoResponse, summary="Get model metadata")
async def model_info():
    """Return model version, evaluation metrics, and configuration."""
    lstm = _registry.get("lstm")
    if lstm is None:
        raise HTTPException(status_code=503, detail="LSTM model not loaded")

    hparams = _registry.get("lstm_hparams", {})
    metrics = _registry.get("lstm_metrics", {})

    # Count parameters
    n_params = sum(p.numel() for p in lstm.parameters())

    return ModelInfoResponse(
        model_name="5G_LSTM_Anomaly_Detector",
        model_version="1.0.0",
        threshold=_registry.get("lstm_threshold", 0.5),
        metrics=metrics,
        feature_count=hparams.get("input_size", -1),
        window_size=WINDOW_SIZE,
    )


@app.get("/health", summary="Health check")
async def health():
    return {
        "status": "ok",
        "models_loaded": {
            "lstm": _registry.get("lstm") is not None,
            "scaler": _registry.get("scaler") is not None,
            "knn": _registry.get("knn") is not None,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
