"""
LLM Log Analysis Module
=========================
1. Generates synthetic 5G network error logs aligned with anomaly events.
2. Embeds logs with sentence-transformers (all-MiniLM-L6-v2).
3. Stores embeddings in ChromaDB.
4. Trains a KNN root-cause classifier on the embeddings.
5. Combines KNN predictions with LSTM anomaly scores for a unified alert.

Log format:
    <timestamp> | <severity> | <error_code> | <message>

Root-cause categories:
    timeout, packet_loss, handover_fail, congestion,
    signal_degradation, cpu_overload, normal
"""

import os
import re
import warnings
import random
import json
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import mlflow

warnings.filterwarnings("ignore")
os.makedirs("outputs/logs", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

# ─── Log Templates ────────────────────────────────────────────────────────────

_LOG_TEMPLATES = {
    "timeout": [
        "RRC connection setup timeout after {ms}ms on cell {cell}",
        "S1-AP procedure timeout: UE context release request failed",
        "PDCP data transfer timeout — retransmission limit exceeded on bearer {bid}",
        "NAS timer T3410 expired for UE {ue} after {ms}ms",
        "GTP-U path failure: heartbeat response timeout from {ip}",
    ],
    "packet_loss": [
        "High packet loss detected: {pct:.1f}% on interface {iface}",
        "HARQ max retransmissions ({n}) exceeded on PRB {prb}",
        "IP fragmentation drop: packet too large ({size}B) on GTP tunnel {tun}",
        "Excessive uplink NACK ratio: {pct:.1f}% for UE {ue}",
        "RLC retransmission buffer overflow — {n} PDUs dropped on bearer {bid}",
    ],
    "handover_fail": [
        "X2 handover preparation failure to target eNB {enb}: cause {cause}",
        "Handover execution failed: radio link failure at target cell {cell}",
        "Inter-frequency handover rejected: no suitable target cell found",
        "S1 handover cancelled: MME response timeout after {ms}ms",
        "RACH failure post-handover: UE {ue} unable to access target cell {cell}",
    ],
    "congestion": [
        "PRB utilisation at {pct:.0f}% — scheduling backlog growing on cell {cell}",
        "S-GW bearer binding rejected: user plane capacity limit reached",
        "PDCP SDU buffer congestion — {n} packets queued on DRB {drb}",
        "Backhaul link utilisation critical: {pct:.0f}% on link {lnk}",
        "QoS admission control: bearer establishment rejected for QCI {qci}",
    ],
    "signal_degradation": [
        "RSRP dropped to {rsrp:.1f} dBm on cell {cell} — coverage hole suspected",
        "SINR degraded: {sinr:.1f} dB for UE {ue} — possible interference",
        "CQI feedback {cqi} below threshold on subframe {sf}",
        "DL beam failure detected: PDCCH BLER {bler:.1f}% exceeds 10% limit",
        "CSI-RS measurement gap: {n} consecutive missed reports for UE {ue}",
    ],
    "cpu_overload": [
        "CU-CP CPU utilisation at {pct:.0f}% — task queue latency elevated",
        "DU baseband processing overload: real-time deadline missed ({n} times)",
        "RRC state machine thread stall detected — {ms}ms scheduling delay",
        "O-RAN near-RT RIC xApp CPU quota exceeded: {pct:.0f}% for {ms}ms",
        "SMF session management CPU spike: {n} concurrent PDU session setups",
    ],
    "normal": [
        "Periodic neighbour cell measurement report received from UE {ue}",
        "Cell load report: PRB util {pct:.0f}%, active UEs {n}",
        "SON algorithm completed mobility parameter optimisation on cell {cell}",
        "UE {ue} successfully attached — IMSI validated, bearer established",
        "Periodic MDT log collected from {n} UEs in area {area}",
    ],
}

_SEVERITY_MAP = {
    "timeout":           ["ERROR",   "CRITICAL"],
    "packet_loss":       ["WARNING", "ERROR"],
    "handover_fail":     ["ERROR",   "CRITICAL"],
    "congestion":        ["WARNING", "ERROR"],
    "signal_degradation":["WARNING", "ERROR"],
    "cpu_overload":      ["ERROR",   "CRITICAL"],
    "normal":            ["INFO",    "DEBUG"],
}

_ERROR_CODES = {
    "timeout":           ["ERR_RRC_001", "ERR_S1AP_042", "ERR_PDCP_007"],
    "packet_loss":       ["ERR_MAC_012", "ERR_RLC_003",  "ERR_IP_021"],
    "handover_fail":     ["ERR_HO_005",  "ERR_X2_018",   "ERR_S1_033"],
    "congestion":        ["ERR_SCHED_009","ERR_ADMCTRL_002","ERR_QOS_014"],
    "signal_degradation":["ERR_PHY_006", "ERR_BEAM_011", "ERR_CSI_020"],
    "cpu_overload":      ["ERR_CPU_019", "ERR_RT_004",   "ERR_SMF_027"],
    "normal":            ["INFO_MEAS_001","INFO_LOAD_002","INFO_SON_003"],
}

_RNG = random.Random(42)


def _render_template(category: str) -> str:
    tpl = _RNG.choice(_LOG_TEMPLATES[category])
    fmt = dict(
        ms=_RNG.randint(200, 9000),
        cell=f"cell_{_RNG.randint(1, 50):03d}",
        ue=f"UE_{_RNG.randint(1000, 9999)}",
        bid=_RNG.randint(1, 11),
        ip=f"10.{_RNG.randint(0,255)}.{_RNG.randint(0,255)}.{_RNG.randint(1,254)}",
        pct=_RNG.uniform(5, 98),
        iface=f"eth{_RNG.randint(0, 3)}",
        n=_RNG.randint(3, 20),
        prb=_RNG.randint(0, 99),
        tun=f"tun_{_RNG.randint(10000, 99999)}",
        size=_RNG.randint(1500, 9000),
        enb=_RNG.randint(100, 999),
        cause=_RNG.choice(["radio", "transport", "o&m", "load"]),
        rsrp=_RNG.uniform(-115, -80),
        sinr=_RNG.uniform(-5, 15),
        cqi=_RNG.randint(1, 5),
        sf=_RNG.randint(0, 9),
        bler=_RNG.uniform(10, 40),
        drb=_RNG.randint(1, 5),
        lnk=f"BH-{_RNG.randint(1, 8)}",
        qci=_RNG.randint(1, 9),
        area=f"Z{_RNG.randint(1, 20):02d}",
    )
    try:
        return tpl.format(**fmt)
    except KeyError:
        return tpl


# ─── Log Generator ────────────────────────────────────────────────────────────


def generate_logs(
    df: pd.DataFrame,
    logs_per_anomaly: int = 3,
    normal_log_fraction: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic network error logs aligned with telemetry anomalies.

    Parameters
    ----------
    df : pd.DataFrame
        Telemetry DataFrame with 'timestamp', 'is_anomaly', 'anomaly_type'.
    logs_per_anomaly : int
        Number of log lines generated per anomalous timestep.
    normal_log_fraction : float
        Fraction of normal timesteps that also generate a log entry.

    Returns
    -------
    pd.DataFrame with columns:
        log_text, timestamp, severity, error_code, category
    """
    rng_local = random.Random(seed)
    rows = []

    # Anomaly-type → log category mapping
    atype_to_cat = {
        "latency_spike":     "timeout",
        "packet_loss_burst": "packet_loss",
        "throughput_drop":   "congestion",
        "handover_storm":    "handover_fail",
        "signal_degradation":"signal_degradation",
        "cpu_overload":      "cpu_overload",
        "none":              "normal",
    }

    for _, row in df.iterrows():
        ts = row["timestamp"]
        is_anom = row["is_anomaly"]
        atype = row.get("anomaly_type", "none")
        category = atype_to_cat.get(atype, "normal")

        # Decide how many logs to generate
        if is_anom:
            n_logs = logs_per_anomaly
        elif rng_local.random() < normal_log_fraction:
            n_logs = 1
            category = "normal"
        else:
            continue

        for _ in range(n_logs):
            severity = rng_local.choice(_SEVERITY_MAP[category])
            error_code = rng_local.choice(_ERROR_CODES[category])
            message = _render_template(category)
            offset = timedelta(seconds=rng_local.randint(0, 290))
            log_ts = (ts + offset).strftime("%Y-%m-%dT%H:%M:%S")
            log_text = f"{log_ts} | {severity} | {error_code} | {message}"
            rows.append({
                "log_text": log_text,
                "timestamp": ts,
                "severity": severity,
                "error_code": error_code,
                "category": category,
            })

    logs_df = pd.DataFrame(rows)
    log_path = "outputs/logs/network_logs.csv"
    logs_df.to_csv(log_path, index=False)
    print(f"[Logs] Generated {len(logs_df):,} log entries → {log_path}")
    print(logs_df["category"].value_counts().to_string())
    return logs_df


# ─── Embedding Pipeline ───────────────────────────────────────────────────────


def embed_logs(logs_df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embed log texts using sentence-transformers.

    Parameters
    ----------
    logs_df   : pd.DataFrame  — must have 'log_text' column
    model_name: str           — sentence-transformer model id

    Returns
    -------
    np.ndarray, shape (n_logs, embedding_dim)
    """
    from sentence_transformers import SentenceTransformer

    print(f"[Logs] Loading sentence-transformer: {model_name} …")
    encoder = SentenceTransformer(model_name)

    texts = logs_df["log_text"].tolist()
    print(f"[Logs] Embedding {len(texts):,} logs …")
    embeddings = encoder.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"[Logs] Embedding shape: {embeddings.shape}")
    return embeddings


def store_in_chromadb(
    logs_df: pd.DataFrame,
    embeddings: np.ndarray,
    collection_name: str = "network_logs",
    persist_dir: str = "outputs/chromadb",
) -> object:
    """
    Store log embeddings in ChromaDB with metadata.

    Returns
    -------
    ChromaDB collection object.
    """
    import chromadb
    from chromadb.config import Settings

    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 5000
    n = len(logs_df)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_df = logs_df.iloc[start:end]
        batch_emb = embeddings[start:end].tolist()
        ids = [f"log_{start + i}" for i in range(len(batch_df))]
        metadatas = [
            {
                "category": row["category"],
                "severity": row["severity"],
                "error_code": row["error_code"],
            }
            for _, row in batch_df.iterrows()
        ]
        collection.add(
            ids=ids,
            embeddings=batch_emb,
            documents=batch_df["log_text"].tolist(),
            metadatas=metadatas,
        )

    print(f"[Logs] Stored {collection.count()} embeddings in ChromaDB ({persist_dir})")
    return collection


# ─── KNN Root-Cause Classifier ────────────────────────────────────────────────


def train_knn_classifier(
    embeddings: np.ndarray,
    labels: np.ndarray,
    test_frac: float = 0.2,
    n_neighbors: int = 7,
    experiment_name: str = "5G_Anomaly_Detection",
) -> dict:
    """
    Train a KNN classifier on log embeddings for root-cause classification.

    Parameters
    ----------
    embeddings : np.ndarray  — shape (n, embedding_dim)
    labels     : np.ndarray  — integer class labels
    test_frac  : float       — fraction for test split
    n_neighbors: int         — KNN neighbours

    Returns
    -------
    dict: model, label_encoder, metrics, run_id
    """
    from sklearn.model_selection import train_test_split

    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_tr, X_te, y_tr, y_te = train_test_split(
        embeddings, y, test_size=test_frac, random_state=42, stratify=y
    )

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="KNN_LogClassifier") as run:
        run_id = run.info.run_id
        mlflow.log_params({"model_type": "KNN", "n_neighbors": n_neighbors})

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine", n_jobs=-1)
        knn.fit(X_tr, y_tr)

        y_pred = knn.predict(X_te)
        f1_macro = f1_score(y_te, y_pred, average="macro", zero_division=0)
        f1_per_class = f1_score(y_te, y_pred, average=None, zero_division=0)

        report = classification_report(
            y_te, y_pred, target_names=le.classes_, zero_division=0
        )
        print(f"\n[KNN] Root-cause classification report:\n{report}")

        metrics = {
            "f1_macro": f1_macro,
            **{f"f1_{cls}": float(sc) for cls, sc in zip(le.classes_, f1_per_class)},
        }
        mlflow.log_metrics({f"knn_{k}": v for k, v in metrics.items()})
        mlflow.log_text(report, "knn_classification_report.txt")

        print(f"[KNN] run_id: {run_id}")

    import joblib
    joblib.dump({"knn": knn, "le": le}, "outputs/models/knn_classifier.joblib")
    return {"model": knn, "label_encoder": le, "metrics": metrics, "run_id": run_id}


# ─── Unified Alert Score ─────────────────────────────────────────────────────


def unified_alert_score(
    lstm_probs: np.ndarray,
    knn_anomaly_prob: np.ndarray,
    alpha: float = 0.7,
) -> np.ndarray:
    """
    Combine LSTM anomaly probability with KNN anomaly probability.

    Score = alpha * lstm_prob + (1 - alpha) * knn_prob

    Parameters
    ----------
    lstm_probs        : np.ndarray — LSTM sigmoid output, shape (n,)
    knn_anomaly_prob  : np.ndarray — KNN P(anomaly), shape (n,)
    alpha             : float      — weight for LSTM (default 0.7)

    Returns
    -------
    np.ndarray — unified alert score in [0, 1]
    """
    return alpha * lstm_probs + (1 - alpha) * knn_anomaly_prob


# ─── Query Interface ──────────────────────────────────────────────────────────


def query_similar_logs(
    query_text: str,
    collection,
    encoder,
    n_results: int = 5,
) -> list[dict]:
    """
    Semantic search: return the n most similar log entries for a query text.

    Parameters
    ----------
    query_text : str         — free-text query
    collection : ChromaDB collection
    encoder    : SentenceTransformer model
    n_results  : int

    Returns
    -------
    List of dicts with keys: document, metadata, distance
    """
    q_emb = encoder.encode([query_text], convert_to_numpy=True).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=n_results)
    out = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        out.append({"document": doc, "metadata": meta, "distance": dist})
    return out


# ─── Full Pipeline ────────────────────────────────────────────────────────────


def run_log_analysis(df: pd.DataFrame) -> dict:
    """
    Execute the complete log analysis pipeline.

    Parameters
    ----------
    df : pd.DataFrame — telemetry DataFrame (with is_anomaly, anomaly_type)

    Returns
    -------
    dict with: logs_df, embeddings, collection, knn_results
    """
    print("[LogAnalysis] Generating synthetic logs …")
    logs_df = generate_logs(df)

    print("[LogAnalysis] Embedding logs …")
    embeddings = embed_logs(logs_df)

    print("[LogAnalysis] Storing in ChromaDB …")
    collection = store_in_chromadb(logs_df, embeddings)

    print("[LogAnalysis] Training KNN classifier …")
    knn_results = train_knn_classifier(embeddings, logs_df["category"].values)

    return {
        "logs_df": logs_df,
        "embeddings": embeddings,
        "collection": collection,
        "knn_results": knn_results,
    }


if __name__ == "__main__":
    from data_generation import generate_telemetry
    df = generate_telemetry(n_steps=1000)
    result = run_log_analysis(df)
    print("Log analysis complete.")
