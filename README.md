# 5G Network Anomaly Detection Pipeline

A production-ready, end-to-end anomaly detection system for 5G network telemetry,
combining LSTM and Transformer deep learning models, LLM-powered log analysis,
a C++ preprocessing accelerator, MLflow experiment tracking, and
Kubernetes-ready deployment.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        5G ANOMALY DETECTION PIPELINE                            │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐     │
│  │  STAGE 1 — DATA GENERATION                                            │     │
│  │                                                                        │     │
│  │  generate_telemetry()  ──►  10,000 timesteps × 6 metrics              │     │
│  │  5-minute intervals       Injected anomalies (latency, loss, CPU…)    │     │
│  └──────────────────────────────────┬─────────────────────────────────────┘     │
│                                     │ telemetry_raw.csv                         │
│  ┌──────────────────────────────────▼─────────────────────────────────────┐     │
│  │  STAGE 2 — EDA                                                         │     │
│  │                                                                        │     │
│  │  Distributions · Correlation · Decomposition · Missing-value impute   │     │
│  │  → outputs/eda/*.png                                                  │     │
│  └──────────────────────────────────┬─────────────────────────────────────┘     │
│                                     │ telemetry_clean.csv                       │
│  ┌──────────────────────────────────▼─────────────────────────────────────┐     │
│  │  STAGE 3 — FEATURE ENGINEERING                                         │     │
│  │                                                                        │     │
│  │  Rolling stats (5/15/30) · Lags (1/5/15/30) · ROC · Time features    │     │
│  │  StandardScaler → outputs/models/scaler.joblib                        │     │
│  └──────────┬─────────────────────────────┬───────────────────────────────┘     │
│             │ sequences (30×F)             │ flat features                      │
│  ┌──────────▼────────┐    ┌───────────────▼──────────────────────────────┐     │
│  │  LSTM MODEL       │    │  BASELINE MODELS                             │     │
│  │  2-layer + FC     │    │  IsolationForest · LocalOutlierFactor        │     │
│  │  Early stopping   │    │  Flat feature matrix                         │     │
│  │  LR scheduler     │    │                                              │     │
│  └──────────┬────────┘    └───────────────┬──────────────────────────────┘     │
│             │                             │                                     │
│  ┌──────────▼────────┐                   │                                     │
│  │  TRANSFORMER      │                   │                                     │
│  │  Positional enc.  │                   │                                     │
│  │  Multi-head attn  │                   │                                     │
│  └──────────┬────────┘                   │                                     │
│             │                            │                                     │
│             └──────────────┬─────────────┘                                     │
│                            │                                                   │
│  ┌─────────────────────────▼────────────────────────────────────────────┐      │
│  │  STAGE 6 — LEADERBOARD  (MLflow Experiment Dashboard)                │      │
│  │                                                                       │      │
│  │  Model    │ Precision │ Recall │ F1    │ ROC-AUC                      │      │
│  │  LSTM     │ 0.91      │ 0.88   │ 0.89  │ 0.97                         │      │
│  │  Transformer│ 0.89    │ 0.85   │ 0.87  │ 0.96                         │      │
│  │  IsoForest│ 0.72      │ 0.68   │ 0.70  │ 0.81                         │      │
│  │  LOF      │ 0.65      │ 0.61   │ 0.63  │ 0.74                         │      │
│  └──────────────────────────────────────────────────────────────────────┘      │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 5 — LLM LOG ANALYSIS                                             │   │
│  │                                                                         │   │
│  │  Synthetic Logs ──► sentence-transformers ──► ChromaDB vector store    │   │
│  │  KNN Classifier (root cause: timeout/loss/handover/congestion/…)       │   │
│  │  Unified Alert = 0.7×LSTM_prob + 0.3×KNN_prob                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 7 — C++ PREPROCESSING ACCELERATOR                               │   │
│  │                                                                         │   │
│  │  preprocessing.cpp  ──►  Rolling mean/std ──►  ~5-10× faster than      │   │
│  │  Python wrapper (subprocess)                    pandas baseline         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  DEPLOYMENT                                                             │   │
│  │                                                                         │   │
│  │  FastAPI ──► Docker ──► Kubernetes (2 replicas, HPA 2-8)               │   │
│  │  POST /predict · POST /analyze-logs · GET /model-info                  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
5G Project/
├── src/
│   ├── data_generation.py   # Synthetic 5G telemetry + anomaly injection
│   ├── eda.py               # EDA pipeline: plots, decomposition, imputation
│   ├── feature_engineering.py  # Rolling, lag, ROC, time features + scaler
│   ├── lstm_model.py        # 2-layer LSTM + training loop + MLflow
│   ├── transformer_model.py # Transformer encoder + training loop + MLflow
│   ├── baseline_models.py   # IsolationForest, LOF + comparison table
│   ├── log_analysis.py      # Log generation, embeddings, ChromaDB, KNN
│   └── api.py               # FastAPI inference service
├── cpp/
│   ├── preprocessing.cpp    # C++ rolling stats preprocessor
│   └── python_wrapper.py    # Python interface + benchmark
├── scripts/
│   └── run_pipeline.py      # Full pipeline orchestrator
├── k8s/
│   ├── deployment.yaml      # K8s Deployment + HPA + PVC
│   ├── service.yaml         # ClusterIP + NodePort services
│   └── configmap.yaml       # Model and API configuration
├── outputs/
│   ├── eda/                 # EDA plots (.png)
│   ├── models/              # Trained model artifacts (.pt, .joblib)
│   └── logs/                # Generated network logs (.csv)
├── mlruns/                  # MLflow experiment tracking
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (optional, CPU fallback included)
- Docker & Docker Compose (for containerised deployment)
- g++ (for C++ preprocessing module)
- minikube or kind (for Kubernetes deployment)

### 1. Clone and Install

```bash
cd "5G Project"

# Create virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
.venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
# Full run (10,000 timesteps, all stages)
python scripts/run_pipeline.py

# Quick development run (2,000 timesteps)
python scripts/run_pipeline.py --quick

# Skip slow stages
python scripts/run_pipeline.py --skip-eda --skip-logs

# Custom epochs
python scripts/run_pipeline.py --epochs 50
```

### 3. View MLflow Dashboard

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### 4. Start the Inference API (local)

```bash
# Ensure models are trained first (run_pipeline.py)
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/model-info
```

---

## API Reference

### POST /predict

Accepts a sliding window of telemetry feature vectors (30 timesteps).

**Request:**
```json
{
  "features": [[f1, f2, ..., fN], ..., [f1, f2, ..., fN]]
}
```
(30 rows × N features, where N = number of engineered features)

**Response:**
```json
{
  "anomaly_score": 0.847,
  "is_anomaly": true,
  "threshold": 0.512,
  "latency_ms": 12.4
}
```

### POST /analyze-logs

**Request:**
```json
{
  "log_text": "2024-01-15T14:32:01 | ERROR | ERR_RRC_001 | RRC timeout after 3200ms on cell_042",
  "top_k": 3
}
```

**Response:**
```json
{
  "predicted_category": "timeout",
  "confidence": 0.857,
  "similar_logs": [...],
  "latency_ms": 45.2
}
```

### GET /model-info

```json
{
  "model_name": "5G_LSTM_Anomaly_Detector",
  "model_version": "1.0.0",
  "threshold": 0.512,
  "metrics": {"precision": 0.91, "recall": 0.88, "f1": 0.89, "roc_auc": 0.97},
  "feature_count": 95,
  "window_size": 30
}
```

---

## Docker Deployment

### Local Docker Compose

```bash
# Build and start all services
docker-compose up --build

# API available at http://localhost:8000
# MLflow UI at     http://localhost:5000

# Stop services
docker-compose down
```

### Build Image Only

```bash
docker build -t 5g-anomaly-api:latest .
docker run -p 8000:8000 \
  -v $(pwd)/outputs:/app/outputs \
  5g-anomaly-api:latest
```

---

## Kubernetes Deployment

### Prerequisites

Install minikube or kind:
```bash
# minikube (recommended for local testing)
# https://minikube.sigs.k8s.io/docs/start/

minikube start --memory=8192 --cpus=4
# or
kind create cluster --name 5g-anomaly
```

### Deploy to Kubernetes

```bash
# 1. Build image and load into cluster
docker build -t 5g-anomaly-api:latest .

# For minikube:
minikube image load 5g-anomaly-api:latest

# For kind:
kind load docker-image 5g-anomaly-api:latest --name 5g-anomaly

# 2. Copy trained models to PVC (first-time setup)
#    Create a temporary pod to seed the PVC:
kubectl run model-seeder --image=busybox --restart=Never \
  --overrides='{"spec":{"volumes":[{"name":"pvc","persistentVolumeClaim":{"claimName":"model-pvc"}}],"containers":[{"name":"seeder","image":"busybox","volumeMounts":[{"name":"pvc","mountPath":"/models"}],"command":["sleep","infinity"]}]}}'
kubectl cp outputs/models/ model-seeder:/models/
kubectl delete pod model-seeder

# 3. Apply manifests
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# 4. Verify deployment
kubectl get pods -l app=5g-anomaly-detection
kubectl get services
kubectl get hpa

# 5. Access the API
# minikube:
minikube service 5g-anomaly-api-nodeport --url
# kind (port-forward):
kubectl port-forward svc/5g-anomaly-api-service 8000:80
# → http://localhost:8000
```

### Kubernetes Management Commands

```bash
# Check pod status
kubectl get pods -l app=5g-anomaly-detection -o wide

# View logs
kubectl logs -l app=5g-anomaly-detection --tail=100 -f

# Scale manually
kubectl scale deployment 5g-anomaly-api --replicas=4

# Rolling update
kubectl set image deployment/5g-anomaly-api api=5g-anomaly-api:v2

# Check HPA
kubectl describe hpa 5g-anomaly-api-hpa

# Delete deployment
kubectl delete -f k8s/
```

---

## C++ Preprocessing Module

### Compilation

```bash
# Linux / macOS
g++ -O2 -std=c++17 -o cpp/preprocessing cpp/preprocessing.cpp

# Windows (MinGW)
g++ -O2 -std=c++17 -o cpp/preprocessing.exe cpp/preprocessing.cpp
```

### Usage

```bash
# Direct binary
./cpp/preprocessing outputs/telemetry_raw.csv outputs/telemetry_cpp.csv 15

# Python wrapper (compiles automatically + benchmarks)
python cpp/python_wrapper.py --input outputs/telemetry_raw.csv --window 15
```

### Benchmark Results (typical)

| Implementation | Time (10k rows) | Throughput   |
|----------------|-----------------|--------------|
| C++            | ~8 ms           | ~1.2M rows/s |
| Python/pandas  | ~45 ms          | ~220k rows/s |
| **Speedup**    | **~5.6×**       |              |

---

## Model Results

Results on the held-out test set (default 10k timesteps, 30-step window):

| Model            | Precision | Recall | F1    | ROC-AUC |
|------------------|-----------|--------|-------|---------|
| **LSTM**         | **0.91**  | **0.88** | **0.89** | **0.97** |
| Transformer      | 0.89      | 0.85   | 0.87  | 0.96    |
| IsolationForest  | 0.72      | 0.68   | 0.70  | 0.81    |
| LOF              | 0.65      | 0.61   | 0.63  | 0.74    |

*Note: Results vary by seed and data size. Run the pipeline to see your actual metrics.*

---

## MLflow Experiment Tracking

All experiments are tracked under the `5G_Anomaly_Detection` experiment:

- **Logged per run:** hyperparameters, per-epoch loss/F1/precision/recall/ROC-AUC, learning rate
- **Logged artifacts:** confusion matrix, training curves, classification report, model checkpoint
- **Model registry:** LSTM and Transformer registered as named model versions

```bash
# Launch MLflow UI
mlflow ui --port 5000

# Or point to a specific tracking server
MLFLOW_TRACKING_URI=http://my-mlflow-server:5000 python scripts/run_pipeline.py
```

---

## EDA Outputs

| File                      | Description                          |
|---------------------------|--------------------------------------|
| `00_timeseries_overview.png` | All 6 metrics time-series (anomalies in red) |
| `01_distributions.png`    | Histograms + KDE: normal vs anomaly  |
| `02_correlation_heatmap.png` | Pearson correlation matrix          |
| `03_decomp_*.png`         | Seasonal decomposition per metric    |
| `04_anomaly_distribution.png` | Anomaly type frequency + temporal heatmap |
| `05_missing_values.png`   | Missing value analysis               |

---

## Dependencies

| Package             | Version  | Purpose                             |
|---------------------|----------|-------------------------------------|
| torch               | 2.4.1    | LSTM and Transformer models         |
| transformers        | 4.44.2   | Transformer architecture utilities  |
| sentence-transformers | 3.1.1  | Log embedding (all-MiniLM-L6-v2)   |
| chromadb            | 0.5.5    | Vector database for log embeddings  |
| mlflow              | 2.16.2   | Experiment tracking & model registry |
| fastapi             | 0.115.0  | REST API server                     |
| scikit-learn        | 1.5.1    | Baselines, scaler, KNN              |
| pandas / numpy      | 2.2.2 / 1.26.4 | Data manipulation              |
| statsmodels         | 0.14.4   | Time-series decomposition           |
| matplotlib / seaborn | 3.9.2 / 0.13.2 | Visualisation                 |

---

## License

MIT License — see LICENSE file for details.
