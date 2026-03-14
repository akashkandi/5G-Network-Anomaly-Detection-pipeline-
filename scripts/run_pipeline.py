"""
5G Network Anomaly Detection — Full Pipeline Orchestrator
==========================================================
Runs all stages end-to-end in a single script:

  Stage 1 : Data generation
  Stage 2 : EDA
  Stage 3 : Feature engineering
  Stage 4 : Model training  (LSTM + Transformer + Baselines)
  Stage 5 : Log analysis    (Embedding + KNN + ChromaDB)
  Stage 6 : Model comparison leaderboard

Usage:
    cd "5G Project"
    python scripts/run_pipeline.py [--quick] [--skip-eda] [--skip-logs]

Flags:
    --quick      Use 2,000 timesteps (fast dev run)
    --skip-eda   Skip plot generation
    --skip-logs  Skip log embedding (slow, needs internet for first download)
"""

import sys
import os
import argparse
import time
import warnings

# ── Make src importable ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import mlflow

warnings.filterwarnings("ignore")

from src.data_generation import generate_telemetry
from src.eda import run_eda
from src.feature_engineering import (
    build_features, get_feature_columns, prepare_splits,
    fit_and_scale, create_sequences,
)
from src.lstm_model import train_lstm
from src.transformer_model import train_transformer
from src.baseline_models import (
    train_isolation_forest, train_lof, build_comparison_table,
)

EXPERIMENT_NAME = "5G_Anomaly_Detection"
WINDOW_SIZE = 30


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="5G Anomaly Detection Pipeline")
    p.add_argument("--quick", action="store_true",
                    help="Fast dev run with 2,000 timesteps")
    p.add_argument("--skip-eda", action="store_true",
                    help="Skip EDA plot generation")
    p.add_argument("--skip-logs", action="store_true",
                    help="Skip log embedding pipeline")
    p.add_argument("--epochs", type=int, default=30,
                    help="Max training epochs (default 30)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ─── Stage Helpers ────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    print("\n" + "═" * 60)
    print(f"  {text}")
    print("═" * 60)


def _elapsed(t0: float) -> str:
    s = time.perf_counter() - t0
    return f"{s//60:.0f}m {s%60:.1f}s"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    pipeline_start = time.perf_counter()

    np.random.seed(args.seed)
    os.makedirs("outputs/models", exist_ok=True)

    # ── Stage 1: Data Generation ──────────────────────────────────────────────
    _banner("Stage 1 — Data Generation")
    t0 = time.perf_counter()
    n_steps = 2_000 if args.quick else 10_000
    df_raw = generate_telemetry(n_steps=n_steps, seed=args.seed)
    df_raw.to_csv("outputs/telemetry_raw.csv", index=False)
    print(f"  Saved outputs/telemetry_raw.csv  [{_elapsed(t0)}]")

    # ── Stage 2: EDA ──────────────────────────────────────────────────────────
    _banner("Stage 2 — Exploratory Data Analysis")
    t0 = time.perf_counter()
    if args.skip_eda:
        print("  [SKIP] EDA skipped")
        df_clean = df_raw.copy()
        for col in ["latency_ms","packet_loss_pct","throughput_mbps",
                     "handover_count","signal_strength_dbm","cpu_util_pct"]:
            df_clean[col] = df_clean[col].interpolate(method="linear",
                limit_direction="both").fillna(method="bfill")
    else:
        df_clean = run_eda(df_raw)
    df_clean.to_csv("outputs/telemetry_clean.csv", index=False)
    print(f"  EDA complete  [{_elapsed(t0)}]")

    # ── Stage 3: Feature Engineering ─────────────────────────────────────────
    _banner("Stage 3 — Feature Engineering")
    t0 = time.perf_counter()
    df_feat = build_features(df_clean)
    df_feat.to_csv("outputs/telemetry_features.csv", index=False)

    feature_cols = get_feature_columns(df_feat)
    splits = prepare_splits(df_feat, feature_cols)

    X_tr_s, X_val_s, X_te_s, scaler = fit_and_scale(
        splits["X_train"], splits["X_val"], splits["X_test"]
    )
    print(f"  Feature engineering complete  [{_elapsed(t0)}]")

    # ── Create sliding-window sequences ───────────────────────────────────────
    X_tr_seq, y_tr_seq   = create_sequences(X_tr_s,  splits["y_train"], WINDOW_SIZE)
    X_val_seq, y_val_seq = create_sequences(X_val_s, splits["y_val"],   WINDOW_SIZE)
    X_te_seq, y_te_seq   = create_sequences(X_te_s,  splits["y_test"],  WINDOW_SIZE)

    print(f"  Sequence shapes — train: {X_tr_seq.shape}, "
          f"val: {X_val_seq.shape}, test: {X_te_seq.shape}")

    # ── Stage 4a: LSTM ────────────────────────────────────────────────────────
    _banner("Stage 4a — LSTM Training")
    t0 = time.perf_counter()
    lstm_result = train_lstm(
        X_tr_seq, y_tr_seq, X_val_seq, y_val_seq, X_te_seq, y_te_seq,
        max_epochs=args.epochs,
        experiment_name=EXPERIMENT_NAME,
    )
    print(f"  LSTM training complete  [{_elapsed(t0)}]")
    print(f"  Best metrics: {lstm_result['best_metrics']}")

    # ── Stage 4b: Transformer ─────────────────────────────────────────────────
    _banner("Stage 4b — Transformer Training")
    t0 = time.perf_counter()
    trans_result = train_transformer(
        X_tr_seq, y_tr_seq, X_val_seq, y_val_seq, X_te_seq, y_te_seq,
        max_epochs=args.epochs,
        experiment_name=EXPERIMENT_NAME,
    )
    print(f"  Transformer training complete  [{_elapsed(t0)}]")
    print(f"  Best metrics: {trans_result['best_metrics']}")

    # ── Stage 4c: Baselines ───────────────────────────────────────────────────
    _banner("Stage 4c — Baseline Models (IsolationForest + LOF)")
    t0 = time.perf_counter()
    # Use flat (non-sequential) features for baselines
    iso_result = train_isolation_forest(
        X_tr_s, splits["y_train"], X_te_s, splits["y_test"],
        experiment_name=EXPERIMENT_NAME,
    )
    lof_result = train_lof(
        X_tr_s, splits["y_train"], X_te_s, splits["y_test"],
        experiment_name=EXPERIMENT_NAME,
    )
    print(f"  Baselines complete  [{_elapsed(t0)}]")

    # ── Stage 5: Log Analysis ─────────────────────────────────────────────────
    _banner("Stage 5 — LLM Log Analysis")
    t0 = time.perf_counter()
    if args.skip_logs:
        print("  [SKIP] Log analysis skipped (--skip-logs)")
        log_result = None
    else:
        try:
            from src.log_analysis import run_log_analysis
            log_result = run_log_analysis(df_feat)
            print(f"  Log analysis complete  [{_elapsed(t0)}]")
        except Exception as e:
            print(f"  [WARN] Log analysis failed: {e}")
            log_result = None

    # ── Stage 6: Leaderboard ──────────────────────────────────────────────────
    _banner("Stage 6 — Model Comparison Leaderboard")
    all_results = {
        "LSTM":            lstm_result,
        "Transformer":     trans_result,
        "IsolationForest": iso_result,
        "LOF":             lof_result,
    }
    build_comparison_table(all_results)

    # ── Final Summary ─────────────────────────────────────────────────────────
    _banner("Pipeline Complete")
    print(f"  Total time: {_elapsed(pipeline_start)}")
    print(f"  Outputs saved to: outputs/")
    print(f"\n  To view MLflow dashboard:")
    print(f"    mlflow ui --port 5000")
    print(f"\n  To start the inference API:")
    print(f"    uvicorn src.api:app --host 0.0.0.0 --port 8000")
    print(f"\n  To run with Docker:")
    print(f"    docker-compose up --build")


if __name__ == "__main__":
    main()
