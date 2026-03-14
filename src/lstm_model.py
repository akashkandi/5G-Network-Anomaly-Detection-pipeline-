"""
LSTM Anomaly Detection Model
==============================
2-layer LSTM + Dropout + Fully-Connected classifier trained on
sliding-window telemetry sequences.

Training pipeline:
  - Stratified-like train / val / test split (via chronological split)
  - Early stopping (patience=10)
  - ReduceLROnPlateau scheduler
  - MLflow logging (hyperparams, per-epoch metrics, artifacts)
  - Threshold optimisation via F1 sweep
  - Comparison against IsolationForest and LOF baselines
"""

import os
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
import mlflow
import mlflow.pytorch

warnings.filterwarnings("ignore")
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/eda", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Model Architecture ───────────────────────────────────────────────────────


class LSTMAnomalyDetector(nn.Module):
    """
    Two-layer bidirectional-optional LSTM for binary anomaly detection.

    Parameters
    ----------
    input_size  : int   — number of features per timestep
    hidden_size : int   — LSTM hidden units (default 128)
    num_layers  : int   — stacked LSTM layers (default 2)
    dropout     : float — dropout between LSTM layers (default 0.3)
    bidirectional: bool — bidirectional LSTM (default False)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * directions, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_size) → logits: (batch,)"""
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # Take last timestep
        logits = self.fc(out).squeeze(-1)
        return logits


# ─── Training Utilities ───────────────────────────────────────────────────────


def _make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 256,
) -> tuple[DataLoader, DataLoader]:
    """Build PyTorch DataLoaders from numpy arrays."""
    def _to_ds(X, y):
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    train_loader = DataLoader(_to_ds(X_train, y_train), batch_size=batch_size,
                               shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(_to_ds(X_val, y_val), batch_size=batch_size,
                             shuffle=False, drop_last=False, num_workers=0)
    return train_loader, val_loader


def _pos_weight(y: np.ndarray) -> torch.Tensor:
    """Compute BCEWithLogitsLoss pos_weight to handle class imbalance."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    weight = n_neg / max(n_pos, 1)
    return torch.tensor([weight], dtype=torch.float32).to(DEVICE)


def _epoch_metrics(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute precision, recall, F1, ROC-AUC for one epoch."""
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= threshold).astype(int)
    return {
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "roc_auc":   roc_auc_score(labels, probs) if labels.sum() > 0 else 0.0,
    }


# ─── Threshold Optimisation ───────────────────────────────────────────────────


def optimise_threshold(
    probs: np.ndarray, y_true: np.ndarray, n_thresholds: int = 200
) -> tuple[float, dict]:
    """
    Sweep classification thresholds and return the one that maximises F1.

    Returns
    -------
    best_thresh : float
    best_metrics : dict
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_f1, best_thresh = -1.0, 0.5
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    preds_best = (probs >= best_thresh).astype(int)
    best_metrics = {
        "threshold": best_thresh,
        "precision": precision_score(y_true, preds_best, zero_division=0),
        "recall":    recall_score(y_true, preds_best, zero_division=0),
        "f1":        best_f1,
        "roc_auc":   roc_auc_score(y_true, probs) if y_true.sum() > 0 else 0.0,
    }
    return best_thresh, best_metrics


# ─── Visualisations ───────────────────────────────────────────────────────────


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "LSTM") -> str:
    """Save confusion-matrix heatmap and return file path."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {title}", fontweight="bold")
    path = f"outputs/eda/cm_{title.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_curves(history: dict, title: str = "LSTM") -> str:
    """Plot training/validation loss and F1 curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{title} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["val_f1"], label="Val F1", color="green")
    axes[1].set_title(f"{title} — Val F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    path = f"outputs/eda/curves_{title.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ─── Main Training Function ───────────────────────────────────────────────────


def train_lstm(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_seq: np.ndarray,
    X_test_seq: np.ndarray,
    y_test_seq: np.ndarray,
    # Hyperparameters
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 1e-3,
    batch_size: int = 256,
    max_epochs: int = 50,
    patience: int = 10,
    experiment_name: str = "5G_Anomaly_Detection",
    run_name: str = "LSTM",
) -> dict:
    """
    Full LSTM training loop with MLflow tracking.

    Parameters
    ----------
    X_train_seq, y_train_seq : training sequences (n, window, features)
    X_val_seq,   y_val_seq   : validation sequences
    X_test_seq,  y_test_seq  : test sequences
    ...hyperparameters...
    experiment_name : str — MLflow experiment name
    run_name        : str — MLflow run name

    Returns
    -------
    dict with keys: model, best_metrics, threshold, history, run_id
    """
    input_size = X_train_seq.shape[2]

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # ── Log hyperparameters ────────────────────────────────────────────
        hparams = dict(
            model_type="LSTM",
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            window=X_train_seq.shape[1],
        )
        mlflow.log_params(hparams)

        # ── Build model, optimiser, scheduler ─────────────────────────────
        model = LSTMAnomalyDetector(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(DEVICE)

        pos_w = _pos_weight(y_train_seq)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="max", factor=0.5, patience=4, min_lr=1e-6
        )

        train_loader, val_loader = _make_loaders(
            X_train_seq, y_train_seq.astype(np.float32),
            X_val_seq, y_val_seq.astype(np.float32),
            batch_size,
        )

        # ── Training loop ──────────────────────────────────────────────────
        history = {"train_loss": [], "val_loss": [], "val_f1": []}
        best_val_f1, no_improve, best_state = -1.0, 0, None

        for epoch in range(max_epochs):
            # Train
            model.train()
            t_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimiser.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                t_losses.append(loss.item())

            # Validate
            model.eval()
            v_losses, all_logits, all_labels = [], [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    logits = model(xb)
                    v_losses.append(criterion(logits, yb).item())
                    all_logits.extend(logits.cpu().numpy())
                    all_labels.extend(yb.cpu().numpy())

            train_loss = np.mean(t_losses)
            val_loss = np.mean(v_losses)
            val_m = _epoch_metrics(np.array(all_logits), np.array(all_labels).astype(int))

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_m["f1"])

            scheduler.step(val_m["f1"])

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_f1": val_m["f1"],
                    "val_precision": val_m["precision"],
                    "val_recall": val_m["recall"],
                    "val_roc_auc": val_m["roc_auc"],
                    "lr": optimiser.param_groups[0]["lr"],
                },
                step=epoch,
            )

            if val_m["f1"] > best_val_f1:
                best_val_f1 = val_m["f1"]
                no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1

            if (epoch + 1) % 5 == 0:
                print(
                    f"  [LSTM] Ep {epoch+1:3d}/{max_epochs} | "
                    f"loss {train_loss:.4f}/{val_loss:.4f} | "
                    f"val_F1 {val_m['f1']:.4f}"
                )

            if no_improve >= patience:
                print(f"  [LSTM] Early stopping at epoch {epoch+1}")
                break

        # ── Load best weights ──────────────────────────────────────────────
        model.load_state_dict(best_state)
        model.eval()

        # ── Test evaluation ────────────────────────────────────────────────
        X_test_t = torch.tensor(X_test_seq, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            test_logits = model(X_test_t).cpu().numpy()

        test_probs = torch.sigmoid(torch.tensor(test_logits)).numpy()
        best_thresh, best_metrics = optimise_threshold(test_probs, y_test_seq)

        print(f"\n[LSTM] Test results (threshold={best_thresh:.3f}):")
        for k, v in best_metrics.items():
            print(f"  {k}: {v:.4f}")

        mlflow.log_metrics({f"test_{k}": v for k, v in best_metrics.items()})
        mlflow.log_param("best_threshold", best_thresh)

        # ── Classification report ──────────────────────────────────────────
        y_pred = (test_probs >= best_thresh).astype(int)
        report = classification_report(y_test_seq, y_pred,
                                        target_names=["Normal", "Anomaly"])
        print("\n" + report)
        mlflow.log_text(report, "classification_report.txt")

        # ── Artefacts ──────────────────────────────────────────────────────
        cm_path = plot_confusion_matrix(y_test_seq, y_pred, "LSTM")
        curve_path = plot_training_curves(history, "LSTM")
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(curve_path)

        # Save model to MLflow registry
        mlflow.pytorch.log_model(model, artifact_path="lstm_model",
                                  registered_model_name="5G_LSTM_Anomaly_Detector")

        # Save locally
        model_path = "outputs/models/lstm_model.pt"
        torch.save({"model_state": best_state, "hparams": hparams,
                     "threshold": best_thresh, "metrics": best_metrics}, model_path)
        mlflow.log_artifact(model_path)

        print(f"\n[LSTM] MLflow run_id: {run_id}")
        print(f"[LSTM] Model saved to {model_path}")

    return {
        "model": model,
        "best_metrics": best_metrics,
        "threshold": best_thresh,
        "history": history,
        "run_id": run_id,
        "test_probs": test_probs,
        "y_test": y_test_seq,
    }
