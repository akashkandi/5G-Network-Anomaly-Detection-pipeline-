"""
5G Network Telemetry Data Generation
=====================================
Generates realistic synthetic 5G network telemetry with injected anomalies.

Output DataFrame columns:
    timestamp, latency_ms, packet_loss_pct, throughput_mbps,
    handover_count, signal_strength_dbm, cpu_util_pct,
    is_anomaly, anomaly_type
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────

METRICS = [
    "latency_ms",
    "packet_loss_pct",
    "throughput_mbps",
    "handover_count",
    "signal_strength_dbm",
    "cpu_util_pct",
]

ANOMALY_TYPES = [
    "none",
    "latency_spike",
    "packet_loss_burst",
    "throughput_drop",
    "handover_storm",
    "signal_degradation",
    "cpu_overload",
]

# ─── Helper Functions ─────────────────────────────────────────────────────────


def _daily_pattern(timestamps: pd.DatetimeIndex, amplitude: float, offset: float) -> np.ndarray:
    """Return a sinusoidal daily pattern (peak at ~14:00)."""
    hour_rad = (timestamps.hour + timestamps.minute / 60) * 2 * np.pi / 24
    return amplitude * np.sin(hour_rad - np.pi / 2) + offset


def _weekly_pattern(timestamps: pd.DatetimeIndex, amplitude: float) -> np.ndarray:
    """Return a weekly load pattern (higher weekdays)."""
    # day of week: 0=Mon, 6=Sun; business days have higher load
    dow_factor = np.where(timestamps.dayofweek < 5, 1.0, 0.6)
    return amplitude * dow_factor


def _add_noise(signal: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise to a signal."""
    return signal + rng.normal(0, scale, len(signal))


# ─── Base Signal Generators ───────────────────────────────────────────────────


def _gen_latency(timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Baseline latency_ms: 15–45 ms with daily peak."""
    base = 25.0
    daily = _daily_pattern(timestamps, amplitude=8.0, offset=0.0)
    weekly = _weekly_pattern(timestamps, amplitude=3.0)
    signal = base + daily + weekly
    return np.clip(_add_noise(signal, scale=3.0, rng=rng), 5.0, 200.0)


def _gen_packet_loss(timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Baseline packet_loss_pct: 0.01–0.5 % with occasional bursts."""
    base = 0.08
    daily = _daily_pattern(timestamps, amplitude=0.04, offset=0.0)
    signal = base + daily
    return np.clip(_add_noise(signal, scale=0.02, rng=rng), 0.0, 1.0)


def _gen_throughput(timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Baseline throughput_mbps: 200–900 Mbps, inversely correlated with load."""
    base = 600.0
    daily = _daily_pattern(timestamps, amplitude=-150.0, offset=0.0)  # drops at peak load
    weekly = _weekly_pattern(timestamps, amplitude=-80.0)
    signal = base + daily + weekly
    return np.clip(_add_noise(signal, scale=30.0, rng=rng), 50.0, 1000.0)


def _gen_handover(timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Baseline handover_count per 5 min: 1–12."""
    base = 5.0
    daily = _daily_pattern(timestamps, amplitude=3.0, offset=0.0)
    signal = base + daily
    counts = np.round(np.clip(_add_noise(signal, scale=1.5, rng=rng), 0.0, 30.0))
    return counts.astype(float)


def _gen_signal_strength(timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Baseline signal_strength_dbm: -70 to -55 dBm."""
    base = -62.0
    daily = _daily_pattern(timestamps, amplitude=3.0, offset=0.0)
    signal = base + daily
    return np.clip(_add_noise(signal, scale=2.0, rng=rng), -120.0, -40.0)


def _gen_cpu_util(timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Baseline cpu_util_pct: 20–75 %."""
    base = 45.0
    daily = _daily_pattern(timestamps, amplitude=18.0, offset=0.0)
    weekly = _weekly_pattern(timestamps, amplitude=8.0)
    signal = base + daily + weekly
    return np.clip(_add_noise(signal, scale=5.0, rng=rng), 5.0, 99.0)


# ─── Anomaly Injectors ────────────────────────────────────────────────────────


def _inject_latency_spikes(
    df: pd.DataFrame, indices: np.ndarray, rng: np.random.Generator
) -> pd.DataFrame:
    """Latency 5–20× normal for a burst of 2–8 steps."""
    for start in indices:
        duration = int(rng.integers(2, 9))
        end = min(start + duration, len(df))
        multiplier = rng.uniform(5.0, 20.0)
        df.loc[start:end, "latency_ms"] = np.clip(
            df.loc[start:end, "latency_ms"] * multiplier, 0, 5000
        )
        df.loc[start:end, "is_anomaly"] = 1
        df.loc[start:end, "anomaly_type"] = "latency_spike"
    return df


def _inject_packet_loss_bursts(
    df: pd.DataFrame, indices: np.ndarray, rng: np.random.Generator
) -> pd.DataFrame:
    """Packet loss jumps to 2–15 % for 3–10 steps."""
    for start in indices:
        duration = int(rng.integers(3, 11))
        end = min(start + duration, len(df))
        burst_val = rng.uniform(2.0, 15.0)
        df.loc[start:end, "packet_loss_pct"] = burst_val
        df.loc[start:end, "is_anomaly"] = 1
        df.loc[start:end, "anomaly_type"] = "packet_loss_burst"
    return df


def _inject_throughput_drops(
    df: pd.DataFrame, indices: np.ndarray, rng: np.random.Generator
) -> pd.DataFrame:
    """Throughput drops to 10–30 % of normal for 4–12 steps."""
    for start in indices:
        duration = int(rng.integers(4, 13))
        end = min(start + duration, len(df))
        drop_factor = rng.uniform(0.10, 0.30)
        df.loc[start:end, "throughput_mbps"] = (
            df.loc[start:end, "throughput_mbps"] * drop_factor
        )
        df.loc[start:end, "is_anomaly"] = 1
        df.loc[start:end, "anomaly_type"] = "throughput_drop"
    return df


def _inject_handover_storms(
    df: pd.DataFrame, indices: np.ndarray, rng: np.random.Generator
) -> pd.DataFrame:
    """Handover count spikes to 30–60 for 2–5 steps."""
    for start in indices:
        duration = int(rng.integers(2, 6))
        end = min(start + duration, len(df))
        df.loc[start:end, "handover_count"] = rng.uniform(30, 60)
        df.loc[start:end, "is_anomaly"] = 1
        df.loc[start:end, "anomaly_type"] = "handover_storm"
    return df


def _inject_signal_degradation(
    df: pd.DataFrame, indices: np.ndarray, rng: np.random.Generator
) -> pd.DataFrame:
    """Signal drops to -95 to -115 dBm for 5–15 steps."""
    for start in indices:
        duration = int(rng.integers(5, 16))
        end = min(start + duration, len(df))
        df.loc[start:end, "signal_strength_dbm"] = rng.uniform(-115, -95)
        df.loc[start:end, "is_anomaly"] = 1
        df.loc[start:end, "anomaly_type"] = "signal_degradation"
    return df


def _inject_cpu_overload(
    df: pd.DataFrame, indices: np.ndarray, rng: np.random.Generator
) -> pd.DataFrame:
    """CPU spikes to 90–100 % for 3–8 steps."""
    for start in indices:
        duration = int(rng.integers(3, 9))
        end = min(start + duration, len(df))
        df.loc[start:end, "cpu_util_pct"] = rng.uniform(90, 100)
        df.loc[start:end, "is_anomaly"] = 1
        df.loc[start:end, "anomaly_type"] = "cpu_overload"
    return df


# ─── Public API ───────────────────────────────────────────────────────────────


def generate_telemetry(
    n_steps: int = 10_000,
    interval_minutes: int = 5,
    start_date: str = "2024-01-01",
    anomaly_fraction: float = 0.07,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic 5G network telemetry data with labelled anomalies.

    Parameters
    ----------
    n_steps : int
        Number of timesteps (default 10,000).
    interval_minutes : int
        Sampling interval in minutes (default 5).
    start_date : str
        ISO start date string (default "2024-01-01").
    anomaly_fraction : float
        Approximate fraction of timesteps that are anomalous (default 0.07).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, latency_ms, packet_loss_pct,
        throughput_mbps, handover_count, signal_strength_dbm, cpu_util_pct,
        is_anomaly (int 0/1), anomaly_type (str).
    """
    rng = np.random.default_rng(seed)

    # Build timestamp index
    start = pd.Timestamp(start_date)
    timestamps = pd.date_range(start, periods=n_steps, freq=f"{interval_minutes}min")

    # Generate baseline signals
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "latency_ms": _gen_latency(timestamps, rng),
            "packet_loss_pct": _gen_packet_loss(timestamps, rng),
            "throughput_mbps": _gen_throughput(timestamps, rng),
            "handover_count": _gen_handover(timestamps, rng),
            "signal_strength_dbm": _gen_signal_strength(timestamps, rng),
            "cpu_util_pct": _gen_cpu_util(timestamps, rng),
            "is_anomaly": 0,
            "anomaly_type": "none",
        }
    )

    # ── Anomaly injection ──────────────────────────────────────────────────────
    # Each anomaly type gets ~1/6 of the anomaly budget
    total_anomaly_events = int(n_steps * anomaly_fraction / 6)
    # Keep injection starts well away from boundaries
    valid_starts = np.arange(50, n_steps - 50)

    def _sample_starts(k: int) -> np.ndarray:
        chosen = rng.choice(valid_starts, size=k * 3, replace=False)
        # Ensure minimum spacing of 20 steps between events of same type
        chosen = np.sort(chosen)
        filtered = [chosen[0]]
        for idx in chosen[1:]:
            if idx - filtered[-1] >= 20:
                filtered.append(idx)
                if len(filtered) == k:
                    break
        return np.array(filtered[:k])

    n = max(total_anomaly_events, 5)
    df = _inject_latency_spikes(df, _sample_starts(n), rng)
    df = _inject_packet_loss_bursts(df, _sample_starts(n), rng)
    df = _inject_throughput_drops(df, _sample_starts(n), rng)
    df = _inject_handover_storms(df, _sample_starts(n), rng)
    df = _inject_signal_degradation(df, _sample_starts(n), rng)
    df = _inject_cpu_overload(df, _sample_starts(n), rng)

    # ── Introduce ~1 % missing values for EDA realism ─────────────────────────
    for col in METRICS:
        miss_idx = rng.choice(n_steps, size=int(n_steps * 0.01), replace=False)
        df.loc[miss_idx, col] = np.nan

    df = df.reset_index(drop=True)
    print(
        f"[DataGen] Generated {n_steps:,} timesteps | "
        f"Anomalies: {df['is_anomaly'].sum():,} "
        f"({df['is_anomaly'].mean()*100:.1f}%)"
    )
    return df


if __name__ == "__main__":
    df = generate_telemetry()
    df.to_csv("outputs/telemetry_raw.csv", index=False)
    print(df.head())
    print(df["anomaly_type"].value_counts())
