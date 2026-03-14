"""
Transformer Encoder Anomaly Detection Model
=============================================
Lightweight Transformer encoder with positional encoding and
multi-head self-attention for binary anomaly classification.

Architecture:
  PositionalEncoding → TransformerEncoder (N layers) →
  Global Average Pool → Dropout → FC(64) → FC(1)

Logged to the same MLflow experiment as LSTM for side-by-side comparison.
"""

import os
import math
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
)
import mlflow
import mlflow.pytorch

from lstm_model import (
    _make_loaders, _pos_weight, _epoch_metrics,
    optimise_threshold, plot_confusion_matrix, plot_training_curves,
)

warnings.filterwarnings("ignore")
os.makedirs("outputs/models", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Positional Encoding ──────────────────────────────────────────────────────


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding injected into the feature dimension.

    Parameters
    ----------
    d_model : int   — feature/embedding dimension
    max_len : int   — maximum sequence length
    dropout : float — dropout probability
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                       # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float() # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─── Input Projection ─────────────────────────────────────────────────────────


class InputProjection(nn.Module):
    """Project raw features to d_model dimensions."""

    def __init__(self, input_size: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


# ─── Transformer Anomaly Detector ─────────────────────────────────────────────


class TransformerAnomalyDetector(nn.Module):
    """
    Transformer Encoder for time-series anomaly detection.

    Parameters
    ----------
    input_size  : int   — number of raw features per timestep
    d_model     : int   — internal embedding dimension (default 64)
    nhead       : int   — number of attention heads (default 4)
    num_layers  : int   — number of TransformerEncoderLayer blocks (default 2)
    dim_ff      : int   — feedforward layer size (default 256)
    dropout     : float — dropout probability (default 0.2)
    max_len     : int   — maximum sequence length for positional encoding
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.2,
        max_len: int = 512,
    ):
        super().__init__()
        # d_model must be divisible by nhead
        if d_model % nhead != 0:
            d_model = (d_model // nhead) * nhead or nhead

        self.projection = InputProjection(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                              enable_nested_tensor=False)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_size) → logits: (batch,)"""
        x = self.projection(x)          # (batch, seq, d_model)
        x = self.pos_encoding(x)        # add positional info
        x = self.encoder(x)             # (batch, seq, d_model)
        x = x.mean(dim=1)              # Global average pooling over time
        x = self.dropout(x)
        return self.classifier(x).squeeze(-1)


# ─── Training Function ────────────────────────────────────────────────────────


def train_transformer(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_seq: np.ndarray,
    X_test_seq: np.ndarray,
    y_test_seq: np.ndarray,
    # Hyperparameters
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_ff: int = 256,
    dropout: float = 0.2,
    lr: float = 3e-4,
    batch_size: int = 256,
    max_epochs: int = 50,
    patience: int = 10,
    experiment_name: str = "5G_Anomaly_Detection",
    run_name: str = "Transformer",
) -> dict:
    """
    Full Transformer training loop with MLflow tracking.

    Returns
    -------
    dict with keys: model, best_metrics, threshold, history, run_id
    """
    input_size = X_train_seq.shape[2]

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        hparams = dict(
            model_type="Transformer",
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_ff=dim_ff,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            window=X_train_seq.shape[1],
        )
        mlflow.log_params(hparams)

        model = TransformerAnomalyDetector(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_ff=dim_ff,
            dropout=dropout,
        ).to(DEVICE)

        pos_w = _pos_weight(y_train_seq)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=max_epochs, eta_min=1e-6
        )

        train_loader, val_loader = _make_loaders(
            X_train_seq, y_train_seq.astype(np.float32),
            X_val_seq, y_val_seq.astype(np.float32),
            batch_size,
        )

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

            scheduler.step()

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
            val_loss   = np.mean(v_losses)
            val_m = _epoch_metrics(np.array(all_logits), np.array(all_labels).astype(int))

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_m["f1"])

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
                    f"  [Transformer] Ep {epoch+1:3d}/{max_epochs} | "
                    f"loss {train_loss:.4f}/{val_loss:.4f} | "
                    f"val_F1 {val_m['f1']:.4f}"
                )

            if no_improve >= patience:
                print(f"  [Transformer] Early stopping at epoch {epoch+1}")
                break

        model.load_state_dict(best_state)
        model.eval()

        X_test_t = torch.tensor(X_test_seq, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            test_logits = model(X_test_t).cpu().numpy()

        test_probs = torch.sigmoid(torch.tensor(test_logits)).numpy()
        best_thresh, best_metrics = optimise_threshold(test_probs, y_test_seq)

        print(f"\n[Transformer] Test results (threshold={best_thresh:.3f}):")
        for k, v in best_metrics.items():
            print(f"  {k}: {v:.4f}")

        mlflow.log_metrics({f"test_{k}": v for k, v in best_metrics.items()})
        mlflow.log_param("best_threshold", best_thresh)

        y_pred = (test_probs >= best_thresh).astype(int)
        report = classification_report(y_test_seq, y_pred,
                                        target_names=["Normal", "Anomaly"])
        mlflow.log_text(report, "classification_report.txt")

        cm_path = plot_confusion_matrix(y_test_seq, y_pred, "Transformer")
        curve_path = plot_training_curves(history, "Transformer")
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(curve_path)

        mlflow.pytorch.log_model(
            model, artifact_path="transformer_model",
            registered_model_name="5G_Transformer_Anomaly_Detector",
        )

        model_path = "outputs/models/transformer_model.pt"
        torch.save({"model_state": best_state, "hparams": hparams,
                     "threshold": best_thresh, "metrics": best_metrics}, model_path)
        mlflow.log_artifact(model_path)

        print(f"\n[Transformer] MLflow run_id: {run_id}")

    return {
        "model": model,
        "best_metrics": best_metrics,
        "threshold": best_thresh,
        "history": history,
        "run_id": run_id,
        "test_probs": test_probs,
        "y_test": y_test_seq,
    }
