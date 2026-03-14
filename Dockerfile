# ──────────────────────────────────────────────────────────────────────────────
# 5G Network Anomaly Detection — FastAPI Inference Service
# ──────────────────────────────────────────────────────────────────────────────
# Build:
#   docker build -t 5g-anomaly-api:latest .
#
# Run:
#   docker run -p 8000:8000 \
#     -v $(pwd)/outputs:/app/outputs \
#     5g-anomaly-api:latest
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS builder

# Install build dependencies (for compiling some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies (split for better layer caching)
COPY requirements.txt /tmp/requirements.txt

# Core dependencies (no torch yet)
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir \
        numpy==1.26.4 \
        pandas==2.2.2 \
        scikit-learn==1.5.1 \
        joblib==1.4.2 \
        scipy==1.13.1

# PyTorch (CPU-only for slim container)
RUN pip install --no-cache-dir \
    torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

# Remaining ML/serving packages
RUN pip install --no-cache-dir \
        sentence-transformers==3.1.1 \
        chromadb==0.5.5 \
        fastapi==0.115.0 \
        uvicorn[standard]==0.30.6 \
        pydantic==2.9.2 \
        mlflow==2.16.2

# ─── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Application setup
WORKDIR /app

# Copy source code
COPY src/ ./src/
COPY cpp/ ./cpp/

# Create output directories
RUN mkdir -p outputs/models outputs/eda outputs/logs outputs/chromadb mlruns

# Compile C++ preprocessor
RUN g++ -O2 -std=c++17 -o cpp/preprocessing cpp/preprocessing.cpp \
    || echo "C++ compilation skipped (g++ unavailable)"

# Environment variables
ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/outputs/models
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]
