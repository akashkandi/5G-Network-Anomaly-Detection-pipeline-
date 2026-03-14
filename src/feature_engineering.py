"""
Feature Engineering Pipeline
==============================
Builds rolling statistics, lag features, rate-of-change features,
and time-based features. Fits a StandardScaler and saves the artifact.

Output: engineered feature matrix X, target vector y, scaler
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

METRICS = [
    "latency_ms",
    "packet_loss_pct",
    "throughput_mbps",
    "handover_count",
    "signal_strength_dbm",
    "cpu_util_pct",
]

ROLLING_WINDOWS = [5, 15, 30]
LAG_STEPS = [1, 5, 15, 30]
SCALER_PATH = "outputs/models/scaler.joblib"

os.makedirs("outputs/models", exist_ok=True)


# ─── Rolling Statistics ───────────────────────────────────────────────────────


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling mean, std, min, and max for each metric over windows
    of [5, 15, 30] timesteps.

    New columns follow the pattern: {metric}_{stat}_{window}
    e.g.  latency_ms_mean_5
    """
    new_cols = {}
    for col in METRICS:
        for w in ROLLING_WINDOWS:
            roll = df[col].rolling(window=w, min_periods=1)
            new_cols[f"{col}_mean_{w}"] = roll.mean()
            new_cols[f"{col}_std_{w}"] = roll.std().fillna(0.0)
            new_cols[f"{col}_min_{w}"] = roll.min()
            new_cols[f"{col}_max_{w}"] = roll.max()

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ─── Lag Features ─────────────────────────────────────────────────────────────


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features for each metric at [1, 5, 15, 30] timesteps.

    New columns follow the pattern: {metric}_lag_{step}
    e.g.  latency_ms_lag_5
    """
    new_cols = {}
    for col in METRICS:
        for lag in LAG_STEPS:
            new_cols[f"{col}_lag_{lag}"] = df[col].shift(lag)

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ─── Rate of Change ───────────────────────────────────────────────────────────


def add_roc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rate-of-change (first-order difference) for each metric.

    New column: {metric}_roc
    """
    new_cols = {}
    for col in METRICS:
        new_cols[f"{col}_roc"] = df[col].diff().fillna(0.0)

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ─── Time-Based Features ──────────────────────────────────────────────────────


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract cyclical time features from the timestamp column.

    Added features:
      hour_sin, hour_cos  — hour of day encoded cyclically
      dow_sin,  dow_cos   — day of week encoded cyclically
      is_weekend          — binary weekend indicator
    """
    ts = df["timestamp"]
    hour = ts.dt.hour
    dow = ts.dt.dayofweek  # 0=Mon, 6=Sun

    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(int)
    return df


# ─── Scaler ───────────────────────────────────────────────────────────────────


def fit_and_scale(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    save_path: str = SCALER_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on the training set and transform all splits.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray
        Feature arrays with shape (n_samples, n_features).
    save_path : str
        Path to save the fitted scaler artifact.

    Returns
    -------
    (X_train_s, X_val_s, X_test_s, scaler)
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    joblib.dump(scaler, save_path)
    print(f"  [FE] Scaler saved to {save_path}")
    return X_train_s, X_val_s, X_test_s, scaler


def load_scaler(path: str = SCALER_PATH) -> StandardScaler:
    """Load a previously saved StandardScaler."""
    return joblib.load(path)


# ─── Full Pipeline ────────────────────────────────────────────────────────────


def build_features(df: pd.DataFrame, drop_rows: int = 30) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Clean (imputed) telemetry DataFrame.
    drop_rows : int
        Number of initial rows to drop to remove NaN artefacts from
        rolling and lag operations (default 30 = max lag).

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame with all new columns.
    """
    print("[FE] Building features …")
    df = add_rolling_features(df)
    df = add_lag_features(df)
    df = add_roc_features(df)
    df = add_time_features(df)

    # Drop rows with NaN artefacts from lag/rolling operations
    df = df.iloc[drop_rows:].reset_index(drop=True)

    # Forward fill any residual NaNs
    feature_cols = get_feature_columns(df)
    df[feature_cols] = df[feature_cols].fillna(method="ffill").fillna(0.0)

    print(f"  [FE] Feature matrix shape: {df[feature_cols].shape}")
    print(f"  [FE] Feature count: {len(feature_cols)}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return list of feature columns (excludes metadata columns).

    Excludes: timestamp, is_anomaly, anomaly_type
    """
    exclude = {"timestamp", "is_anomaly", "anomaly_type"}
    return [c for c in df.columns if c not in exclude]


def prepare_splits(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict:
    """
    Chronological (non-shuffled) train / val / test split.

    Parameters
    ----------
    df : pd.DataFrame
        Engineered DataFrame.
    feature_cols : list[str]
        Columns to use as features.
    train_frac : float
        Fraction for training set.
    val_frac : float
        Fraction for validation set.

    Returns
    -------
    dict with keys:
        X_train, X_val, X_test  (np.ndarray, float32)
        y_train, y_val, y_test  (np.ndarray, int)
        train_idx, val_idx, test_idx  (np.ndarray, int)
    """
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    idx_train = np.arange(0, n_train)
    idx_val = np.arange(n_train, n_train + n_val)
    idx_test = np.arange(n_train + n_val, n)

    X = df[feature_cols].values.astype(np.float32)
    y = df["is_anomaly"].values.astype(np.int64)

    splits = {
        "X_train": X[idx_train],
        "X_val": X[idx_val],
        "X_test": X[idx_test],
        "y_train": y[idx_train],
        "y_val": y[idx_val],
        "y_test": y[idx_test],
        "train_idx": idx_train,
        "val_idx": idx_val,
        "test_idx": idx_test,
    }
    for k in ("train", "val", "test"):
        n_k = len(splits[f"X_{k}"])
        pos = splits[f"y_{k}"].sum()
        print(f"  [FE] {k:5s}: {n_k:6,} samples | anomaly rate {pos/n_k*100:.2f}%")

    return splits


# ─── Sliding-Window Dataset ───────────────────────────────────────────────────


def create_sequences(
    X: np.ndarray, y: np.ndarray, window: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert flat feature arrays into sliding-window sequences.

    Parameters
    ----------
    X : np.ndarray, shape (n, features)
    y : np.ndarray, shape (n,)
    window : int
        Sequence length (default 30 → 2.5 hours at 5-min intervals).

    Returns
    -------
    X_seq : np.ndarray, shape (n - window, window, features)
    y_seq : np.ndarray, shape (n - window,)  — label at sequence end
    """
    n = len(X)
    xs, ys = [], []
    for i in range(n - window):
        xs.append(X[i : i + window])
        ys.append(y[i + window - 1])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64)


if __name__ == "__main__":
    from data_generation import generate_telemetry
    from eda import run_eda

    df_raw = generate_telemetry()
    df_clean = run_eda(df_raw)
    df_feat = build_features(df_clean)

    feature_cols = get_feature_columns(df_feat)
    splits = prepare_splits(df_feat, feature_cols)

    X_train_s, X_val_s, X_test_s, scaler = fit_and_scale(
        splits["X_train"], splits["X_val"], splits["X_test"]
    )
    print("Feature engineering complete.")
    print(f"Training set shape: {X_train_s.shape}")
