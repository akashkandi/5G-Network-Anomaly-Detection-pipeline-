"""
Exploratory Data Analysis (EDA) Pipeline
==========================================
Generates and saves all EDA plots to outputs/eda/.

Analyses:
  - Distribution plots per metric
  - Correlation heatmap
  - Time-series decomposition (trend, seasonality, residual)
  - Anomaly distribution analysis
  - Missing value analysis + imputation strategy
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

warnings.filterwarnings("ignore")

OUTPUT_DIR = "outputs/eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICS = [
    "latency_ms",
    "packet_loss_pct",
    "throughput_mbps",
    "handover_count",
    "signal_strength_dbm",
    "cpu_util_pct",
]

PALETTE = sns.color_palette("tab10")
sns.set_theme(style="whitegrid", font_scale=1.1)


# ─── Helper ──────────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [EDA] Saved {path}")


# ─── 1. Distribution Plots ────────────────────────────────────────────────────


def plot_distributions(df: pd.DataFrame) -> None:
    """Plot histogram + KDE for each metric, split by anomaly/normal."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(METRICS):
        ax = axes[i]
        normal = df.loc[df["is_anomaly"] == 0, col].dropna()
        anomaly = df.loc[df["is_anomaly"] == 1, col].dropna()

        ax.hist(normal, bins=60, alpha=0.55, color=PALETTE[0], density=True, label="Normal")
        ax.hist(anomaly, bins=60, alpha=0.55, color=PALETTE[3], density=True, label="Anomaly")

        # KDE overlays
        if len(normal) > 10:
            xmin, xmax = ax.get_xlim()
            xs = np.linspace(xmin, xmax, 300)
            kde_n = stats.gaussian_kde(normal)
            kde_a = stats.gaussian_kde(anomaly) if len(anomaly) > 10 else None
            ax.plot(xs, kde_n(xs), color=PALETTE[0], lw=2)
            if kde_a:
                ax.plot(xs, kde_a(xs), color=PALETTE[3], lw=2, linestyle="--")

        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=9)

        # Annotation: mean, std
        ax.axvline(normal.mean(), color=PALETTE[0], linestyle=":", lw=1.5,
                   label=f"μ={normal.mean():.2f}")
        ax.axvline(anomaly.mean(), color=PALETTE[3], linestyle=":", lw=1.5,
                   label=f"μ_anom={anomaly.mean():.2f}")

    fig.suptitle("5G Metric Distributions: Normal vs Anomaly", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, "01_distributions.png")


# ─── 2. Correlation Heatmap ───────────────────────────────────────────────────


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Pearson correlation heatmap for all metrics."""
    corr = df[METRICS].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Metric Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "02_correlation_heatmap.png")


# ─── 3. Time-Series Decomposition ────────────────────────────────────────────


def plot_decomposition(df: pd.DataFrame, period: int = 288) -> None:
    """
    Seasonal decomposition for each metric.
    period=288 corresponds to one 24-hour day at 5-min intervals.
    """
    for col in METRICS:
        series = df[col].interpolate(method="linear").fillna(method="bfill")
        try:
            result = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")
        except Exception:
            continue

        fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
        components = [
            ("Observed", result.observed),
            ("Trend", result.trend),
            ("Seasonal", result.seasonal),
            ("Residual", result.resid),
        ]
        for ax, (label, data) in zip(axes, components):
            ax.plot(df["timestamp"], data, lw=0.8, color=PALETTE[0])
            ax.set_ylabel(label, fontsize=10)
            # Shade anomaly windows
            anom_mask = df["is_anomaly"] == 1
            for idx in df.index[anom_mask]:
                ax.axvspan(
                    df.loc[idx, "timestamp"],
                    df.loc[idx, "timestamp"] + pd.Timedelta(minutes=5),
                    alpha=0.12,
                    color="red",
                )
        axes[0].set_title(f"Time-Series Decomposition: {col}", fontsize=13, fontweight="bold")
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=5))
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()
        _save(fig, f"03_decomp_{col}.png")


# ─── 4. Anomaly Distribution Analysis ────────────────────────────────────────


def plot_anomaly_distribution(df: pd.DataFrame) -> None:
    """Bar chart of anomaly type counts + temporal heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Bar chart ─────────────────────────────────────────────────────────────
    counts = df[df["is_anomaly"] == 1]["anomaly_type"].value_counts()
    axes[0].bar(counts.index, counts.values, color=PALETTE[:len(counts)])
    axes[0].set_title("Anomaly Type Frequency", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Anomaly Type")
    axes[0].set_ylabel("Count")
    for i, (x, y) in enumerate(zip(counts.index, counts.values)):
        axes[0].text(i, y + 2, str(y), ha="center", va="bottom", fontsize=10)
    axes[0].tick_params(axis="x", rotation=30)

    # ── Heatmap: anomalies per hour per day ───────────────────────────────────
    anom_df = df[df["is_anomaly"] == 1].copy()
    anom_df["hour"] = anom_df["timestamp"].dt.hour
    anom_df["dayofweek"] = anom_df["timestamp"].dt.day_name()
    pivot = (
        anom_df.pivot_table(index="dayofweek", columns="hour", values="is_anomaly",
                             aggfunc="count", fill_value=0)
    )
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])

    sns.heatmap(pivot, ax=axes[1], cmap="YlOrRd", linewidths=0.3, annot=False)
    axes[1].set_title("Anomaly Temporal Heatmap (Day × Hour)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_ylabel("Day of Week")

    fig.suptitle("Anomaly Distribution Analysis", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, "04_anomaly_distribution.png")


# ─── 5. Missing Value Analysis ────────────────────────────────────────────────


def plot_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Visualise missing value patterns and apply linear interpolation.

    Returns the imputed DataFrame.
    """
    # ── Missing value bar chart ───────────────────────────────────────────────
    missing = df[METRICS].isnull().sum()
    missing_pct = missing / len(df) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(missing.index, missing.values, color=PALETTE[1])
    axes[0].set_xlabel("Missing Count")
    axes[0].set_title("Missing Values per Metric", fontsize=12, fontweight="bold")
    for i, (v, p) in enumerate(zip(missing.values, missing_pct.values)):
        axes[0].text(v + 0.5, i, f"{v} ({p:.1f}%)", va="center", fontsize=9)

    # ── Missingness heatmap (first 500 rows) ──────────────────────────────────
    sample = df[METRICS].head(500).isnull().astype(int)
    sns.heatmap(sample.T, ax=axes[1], cmap="Blues", yticklabels=True,
                xticklabels=False, cbar_kws={"label": "Missing"})
    axes[1].set_title("Missingness Pattern (first 500 rows)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Row index")

    fig.suptitle("Missing Value Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_missing_values.png")

    # ── Imputation: linear interpolation then forward/backward fill ───────────
    df_imp = df.copy()
    for col in METRICS:
        df_imp[col] = (
            df_imp[col]
            .interpolate(method="linear", limit_direction="both")
            .fillna(method="bfill")
            .fillna(method="ffill")
        )
    print(f"  [EDA] Missing after imputation: {df_imp[METRICS].isnull().sum().sum()}")
    return df_imp


# ─── 6. Time-Series Overview ─────────────────────────────────────────────────


def plot_timeseries_overview(df: pd.DataFrame) -> None:
    """Plot all metrics as time-series with anomaly highlights."""
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(18, 14), sharex=True)

    for i, (col, ax) in enumerate(zip(METRICS, axes)):
        ax.plot(df["timestamp"], df[col], lw=0.7, color=PALETTE[i % len(PALETTE)], label=col)
        # Highlight anomalous points
        anom = df[df["is_anomaly"] == 1]
        ax.scatter(anom["timestamp"], anom[col], color="red", s=4, zorder=5, alpha=0.6)
        ax.set_ylabel(col, fontsize=9)
        ax.yaxis.set_label_position("right")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator())
    fig.autofmt_xdate(rotation=30)
    fig.suptitle("5G Telemetry Time-Series Overview (red = anomaly)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "00_timeseries_overview.png")


# ─── Public API ───────────────────────────────────────────────────────────────


def run_eda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the full EDA pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw telemetry DataFrame from data_generation.generate_telemetry().

    Returns
    -------
    pd.DataFrame
        Imputed DataFrame ready for feature engineering.
    """
    print("[EDA] Starting EDA pipeline …")
    print(f"  Shape: {df.shape} | Anomaly rate: {df['is_anomaly'].mean()*100:.2f}%")

    plot_timeseries_overview(df)
    plot_distributions(df)
    plot_correlation_heatmap(df)
    plot_decomposition(df)
    plot_anomaly_distribution(df)
    df_clean = plot_missing_values(df)

    print("[EDA] Pipeline complete. All plots saved to outputs/eda/")
    return df_clean


if __name__ == "__main__":
    from data_generation import generate_telemetry
    df = generate_telemetry()
    df_clean = run_eda(df)
    df_clean.to_csv("outputs/telemetry_clean.csv", index=False)
