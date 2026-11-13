#!/usr/bin/env python3
"""
Time the Python labeller per-window cost.

Usage:
    python scripts/time_py_labeler.py -p sp500_gold
    python scripts/time_py_labeler.py -p sp500_bitcoin
    python scripts/time_py_labeler.py -p sp500_gold -p sp500_bitcoin -p sp500_nasdaq
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import argparse
import shutil
import tempfile
import time

import numpy as np

from _cli import bootstrap
bootstrap()

from tslearning.config import CFG
from tslearning.datasets.pairs import describe, load_series
from tslearning.labeling.py_labeler import label_pair
from tslearning.labeling.transfer_set import load_thresholds


def time_pair(pair: str, trials: int = 5):
    d = describe(pair)
    transformed = load_series([d["a"], d["b"]])
    tau, qlow = load_thresholds()
    lags = list(CFG["predicates"]["lags"])

    lo, hi = "2018-01-01", "2024-12-31"
    seq_len = d["seq_len"]
    stride = 10

    # Warm-up run
    tmp = tempfile.mkdtemp()
    result = label_pair(d["slug"], transformed[d["a"]], transformed[d["b"]],
                        d["b_key"], tau, qlow, out_dir=tmp,
                        lo=lo, hi=hi, stride=stride, seq_len=seq_len,
                        base_name=d["a_key"], lags=lags)
    shutil.rmtree(tmp)
    n_windows = result["windows"]

    # Timed runs
    times = []
    for _ in range(trials):
        tmp = tempfile.mkdtemp()
        t0 = time.perf_counter()
        label_pair(d["slug"], transformed[d["a"]], transformed[d["b"]],
                   d["b_key"], tau, qlow, out_dir=tmp,
                   lo=lo, hi=hi, stride=stride, seq_len=seq_len,
                   base_name=d["a_key"], lags=lags)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        shutil.rmtree(tmp)

    mean_s = np.mean(times)
    std_s = np.std(times)
    per_window_ms = mean_s / n_windows * 1000

    print(f"\n{pair}")
    print(f"  Windows: {n_windows}, Lags: {lags}, seq_len: {seq_len}")
    print(f"  Trials: {[f'{t:.4f}s' for t in times]}")
    print(f"  Mean: {mean_s:.4f}s ± {std_s:.4f}s")
    print(f"  Per window: {per_window_ms:.3f} ms")
    return {"pair": pair, "windows": n_windows, "mean_s": mean_s,
            "std_s": std_s, "per_window_ms": per_window_ms}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--pairs", nargs="+", default=["sp500_gold"],
                    help="pairs to time (default: sp500_gold)")
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()

    results = [time_pair(p, args.trials) for p in args.pairs]

    if len(results) > 1:
        total_windows = sum(r["windows"] for r in results)
        weighted_ms = sum(r["per_window_ms"] * r["windows"] for r in results) / total_windows
        print(f"\nWeighted mean across all pairs: {weighted_ms:.3f} ms/window")
        print(f"Total windows: {total_windows}")


if __name__ == "__main__":
    main()
