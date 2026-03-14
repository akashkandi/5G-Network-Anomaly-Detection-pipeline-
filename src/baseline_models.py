"""
Baseline Anomaly Detection Models
====================================
Implements IsolationForest and LocalOutlierFactor (LOF) baselines
for comparison against LSTM / Transformer.

Both models are trained on the flat (non-sequential) feature matrix,
logged to the same MLflow experiment, and return standardised metric dicts.
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
import mlflow

warnings.filterwarnings("ignore")
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/eda", exist_ok=True)


# ─── Evaluation Utility ───────────────────────────────────────────────────────


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    """Compute and return a standard metrics dictionary."""
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, scores) if y_true.sum() > 0 else 0.0,
    }


def _save_cm(y_true, y_pred, title: str) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {title}", fontweight="bold")
    path = f"outputs/eda/cm_{title.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ─── IsolationForest ──────────────────────────────────────────────────────────


def train_isolation_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 200,
    contamination: float = 0.07,
    max_samples: str = "auto",
    random_state: int = 42,
    experiment_name: str = "5G_Anomaly_Detection",
) -> dict:
    """
    Train and evaluate an IsolationForest model.

    IsolationForest predicts -1 for anomalies; we remap to 1.
    Anomaly score = -decision_function(X) (higher = more anomalous).

    Returns
    -------
    dict with keys: model, metrics, y_pred, scores, run_id
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="IsolationForest") as run:
        run_id = run.info.run_id

        hparams = dict(
            model_type="IsolationForest",
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
        )
        mlflow.log_params(hparams)

        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train)

        # Predict on test set
        raw_preds = model.predict(X_test)          # +1 = normal, -1 = anomaly
        y_pred = (raw_preds == -1).astype(int)     # 0 = normal, 1 = anomaly
        scores = -model.decision_function(X_test)  # higher = more anomalous

        metrics = _evaluate(y_test, y_pred, scores)

        print(f"\n[IsolationForest] Test results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        print("\n" + classification_report(y_test, y_pred,
                                            target_names=["Normal", "Anomaly"]))

        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        cm_path = _save_cm(y_test, y_pred, "IsolationForest")
        mlflow.log_artifact(cm_path)

        print(f"[IsolationForest] MLflow run_id: {run_id}")

    return {
        "model": model,
        "metrics": metrics,
        "y_pred": y_pred,
        "scores": scores,
        "run_id": run_id,
    }


# ─── Local Outlier Factor ─────────────────────────────────────────────────────


def train_lof(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_neighbors: int = 20,
    contamination: float = 0.07,
    experiment_name: str = "5G_Anomaly_Detection",
) -> dict:
    """
    Train and evaluate a LocalOutlierFactor model (novelty=True mode).

    LOF predicts -1 for anomalies; we remap to 1.
    Anomaly score = -negative_outlier_factor_ (higher = more anomalous).

    Returns
    -------
    dict with keys: model, metrics, y_pred, scores, run_id
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="LOF") as run:
        run_id = run.info.run_id

        hparams = dict(
            model_type="LOF",
            n_neighbors=n_neighbors,
            contamination=contamination,
        )
        mlflow.log_params(hparams)

        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
            n_jobs=-1,
        )
        model.fit(X_train)

        raw_preds = model.predict(X_test)
        y_pred = (raw_preds == -1).astype(int)
        scores = -model.decision_function(X_test)

        metrics = _evaluate(y_test, y_pred, scores)

        print(f"\n[LOF] Test results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        print("\n" + classification_report(y_test, y_pred,
                                            target_names=["Normal", "Anomaly"]))

        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        cm_path = _save_cm(y_test, y_pred, "LOF")
        mlflow.log_artifact(cm_path)

        print(f"[LOF] MLflow run_id: {run_id}")

    return {
        "model": model,
        "metrics": metrics,
        "y_pred": y_pred,
        "scores": scores,
        "run_id": run_id,
    }


# ─── Model Comparison Table ───────────────────────────────────────────────────


def build_comparison_table(results: dict) -> None:
    """
    Print and save a comparison table of all model metrics.

    Parameters
    ----------
    results : dict
        Keys are model names; values are dicts with a 'metrics' sub-dict.
        E.g. {"LSTM": {"metrics": {...}}, "Transformer": {...}, ...}
    """
    import pandas as pd

    rows = []
    for model_name, res in results.items():
        m = res.get("best_metrics") or res.get("metrics", {})
        rows.append(
            {
                "Model": model_name,
                "Precision": round(m.get("precision", 0.0), 4),
                "Recall":    round(m.get("recall", 0.0), 4),
                "F1":        round(m.get("f1", 0.0), 4),
                "ROC-AUC":   round(m.get("roc_auc", 0.0), 4),
            }
        )

    df = pd.DataFrame(rows).sort_values("F1", ascending=False)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON LEADERBOARD")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60 + "\n")

    # Save CSV
    path = "outputs/models/model_comparison.csv"
    df.to_csv(path, index=False)
    print(f"[Baselines] Comparison table saved to {path}")

    # Save bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.2
    ax.bar(x - 1.5 * width, df["Precision"], width, label="Precision")
    ax.bar(x - 0.5 * width, df["Recall"],    width, label="Recall")
    ax.bar(x + 0.5 * width, df["F1"],        width, label="F1")
    ax.bar(x + 1.5 * width, df["ROC-AUC"],   width, label="ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=15)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    chart_path = "outputs/eda/model_comparison.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Baselines] Comparison chart saved to {chart_path}")

    return df
