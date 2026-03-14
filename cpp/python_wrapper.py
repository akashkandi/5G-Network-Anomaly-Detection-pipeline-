"""
Python Wrapper for C++ Preprocessing Module
=============================================
Compiles (if needed) and runs the C++ preprocessor via subprocess,
then benchmarks it against an equivalent pandas implementation.

Usage:
    python cpp/python_wrapper.py [--input outputs/telemetry_raw.csv]
                                 [--output outputs/telemetry_cpp.csv]
                                 [--window 15]
"""

import os
import sys
import time
import argparse
import subprocess
import platform
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CPP_SRC = os.path.join(os.path.dirname(__file__), "preprocessing.cpp")

# Compiled binary name differs by platform
if platform.system() == "Windows":
    CPP_BIN = os.path.join(os.path.dirname(__file__), "preprocessing.exe")
else:
    CPP_BIN = os.path.join(os.path.dirname(__file__), "preprocessing")

METRICS = [
    "latency_ms", "packet_loss_pct", "throughput_mbps",
    "handover_count", "signal_strength_dbm", "cpu_util_pct",
]


# ─── Compilation ─────────────────────────────────────────────────────────────


def compile_cpp(force: bool = False) -> bool:
    """
    Compile the C++ source with g++ if the binary doesn't exist or force=True.

    Returns
    -------
    True if compilation succeeded (or binary already exists), False otherwise.
    """
    if not force and os.path.exists(CPP_BIN):
        print(f"[CPP Wrapper] Binary already exists: {CPP_BIN}")
        return True

    if not os.path.exists(CPP_SRC):
        print(f"[CPP Wrapper] ERROR: Source not found: {CPP_SRC}")
        return False

    cmd = ["g++", "-O2", "-std=c++17", "-o", CPP_BIN, CPP_SRC]
    print(f"[CPP Wrapper] Compiling: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"[CPP Wrapper] Compilation FAILED:\n{result.stderr}")
            return False
        print(f"[CPP Wrapper] Compilation succeeded → {CPP_BIN}")
        return True
    except FileNotFoundError:
        print("[CPP Wrapper] ERROR: g++ not found. Install MinGW (Windows) or gcc (Linux/macOS).")
        return False
    except subprocess.TimeoutExpired:
        print("[CPP Wrapper] ERROR: Compilation timed out.")
        return False


# ─── C++ Runner ──────────────────────────────────────────────────────────────


def run_cpp(input_csv: str, output_csv: str, window: int = 15) -> float:
    """
    Run the compiled C++ preprocessor.

    Returns
    -------
    Wall-clock time in seconds, or -1.0 on failure.
    """
    if not os.path.exists(CPP_BIN):
        print(f"[CPP Wrapper] Binary not found: {CPP_BIN}")
        return -1.0

    cmd = [CPP_BIN, input_csv, output_csv, str(window)]
    print(f"\n[CPP Wrapper] Running C++ preprocessor …")

    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed = time.perf_counter() - t0

        print(result.stdout)
        if result.returncode != 0:
            print(f"[CPP Wrapper] ERROR:\n{result.stderr}")
            return -1.0

        return elapsed
    except subprocess.TimeoutExpired:
        print("[CPP Wrapper] ERROR: C++ run timed out.")
        return -1.0


# ─── Python/Pandas Baseline ───────────────────────────────────────────────────


def run_pandas(input_csv: str, output_csv: str, window: int = 15) -> float:
    """
    Equivalent rolling preprocessing with pandas.

    Returns
    -------
    Wall-clock time in seconds.
    """
    print(f"\n[CPP Wrapper] Running pandas preprocessor …")
    t0 = time.perf_counter()

    df = pd.read_csv(input_csv)

    for col in METRICS:
        if col not in df.columns:
            continue
        roll = df[col].rolling(window=window, min_periods=1)
        df[f"{col}_roll_mean_{window}"] = roll.mean()
        df[f"{col}_roll_std_{window}"]  = roll.std().fillna(0.0)

    df.to_csv(output_csv, index=False)
    elapsed = time.perf_counter() - t0

    rows = len(df)
    print(f"[Pandas] Rows processed : {rows:,}")
    print(f"[Pandas] Elapsed time   : {elapsed*1000:.1f} ms")
    print(f"[Pandas] Throughput     : {int(rows/elapsed):,} rows/sec")
    print(f"[Pandas] Output written : {output_csv}")

    return elapsed


# ─── Benchmark Comparison ─────────────────────────────────────────────────────


def benchmark(
    input_csv: str = "outputs/telemetry_raw.csv",
    cpp_output: str = "outputs/telemetry_cpp.csv",
    pandas_output: str = "outputs/telemetry_pandas.csv",
    window: int = 15,
) -> dict:
    """
    Run both C++ and pandas preprocessors and compare performance.

    Returns
    -------
    dict with keys: cpp_time_s, pandas_time_s, speedup, cpp_ok
    """
    print("\n" + "=" * 55)
    print("  C++ vs Pandas Preprocessing Benchmark")
    print("=" * 55)

    # Compile C++
    cpp_ok = compile_cpp()

    cpp_time = -1.0
    if cpp_ok:
        cpp_time = run_cpp(input_csv, cpp_output, window)

    pandas_time = run_pandas(input_csv, pandas_output, window)

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  BENCHMARK RESULTS")
    print("=" * 55)

    if cpp_time > 0 and pandas_time > 0:
        speedup = pandas_time / cpp_time
        print(f"  C++ time    : {cpp_time*1000:8.1f} ms")
        print(f"  Pandas time : {pandas_time*1000:8.1f} ms")
        print(f"  Speedup     : {speedup:8.2f}×  (C++ is {speedup:.1f}× faster)")
    elif cpp_time <= 0:
        print(f"  C++ time    : FAILED (g++ not available or compile error)")
        print(f"  Pandas time : {pandas_time*1000:8.1f} ms")
        speedup = 0.0
    else:
        speedup = pandas_time / cpp_time
        print(f"  C++ time    : {cpp_time*1000:8.1f} ms")
        print(f"  Pandas time : {pandas_time*1000:8.1f} ms")
        print(f"  Speedup     : {speedup:8.2f}×")

    # ── Validate outputs match ─────────────────────────────────────────────────
    if cpp_ok and cpp_time > 0 and os.path.exists(cpp_output):
        try:
            df_cpp = pd.read_csv(cpp_output)
            df_pd  = pd.read_csv(pandas_output)

            col = f"latency_ms_roll_mean_{window}"
            if col in df_cpp.columns and col in df_pd.columns:
                max_diff = (df_cpp[col] - df_pd[col]).abs().max()
                print(f"\n  Numeric validation (latency roll mean):")
                print(f"  Max absolute diff C++ vs Pandas: {max_diff:.6f}")
                if max_diff < 1e-4:
                    print("  [OK] Results match within tolerance")
                else:
                    print("  [WARN] Results differ beyond tolerance")
        except Exception as e:
            print(f"  Validation error: {e}")

    print("=" * 55 + "\n")

    return {
        "cpp_time_s":    cpp_time,
        "pandas_time_s": pandas_time,
        "speedup":       speedup,
        "cpp_ok":        cpp_ok and cpp_time > 0,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="C++ Preprocessing Wrapper & Benchmark")
    p.add_argument("--input",   default="outputs/telemetry_raw.csv")
    p.add_argument("--output",  default="outputs/telemetry_cpp.csv")
    p.add_argument("--window",  type=int, default=15)
    p.add_argument("--compile-only", action="store_true",
                    help="Only compile the C++ binary, do not run benchmark")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Change working directory to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_dir, ".."))

    if args.compile_only:
        compile_cpp(force=True)
    else:
        benchmark(
            input_csv=args.input,
            cpp_output=args.output,
            pandas_output="outputs/telemetry_pandas.csv",
            window=args.window,
        )
