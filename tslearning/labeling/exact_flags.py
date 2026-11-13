"""
Exact, Python computation of the per-series / per-pair market-state predicates
— trend, volatility regime, co-exceedance (tail dependence).

These match the definitions in preLabeling.vada and use the SAME fitted
thresholds: tau (90th percentile of |return|, per series) and qLow (5th
percentile of return, per series). No reasoner, no classifier — exact and
instant. Use these for the flags; use the classifier only for the correlation
regime.

Definitions (identical to the .vada):
  trend       sum of the 50 daily returns:  >= 0.02 -> trendUp, <= -0.02 -> trendDown, else deadZone
  volatility  % of days with |return| > tau:  >= 30 -> highVol, <= 5 -> lowVol, else neither
  tail        # of days where BOTH series close below their qLow (same day): >= 2 -> tailDependent
"""
from __future__ import annotations
import os
from collections import defaultdict
import numpy as np
import pandas as pd


def label_constants() -> dict:
    """The labelling cut-offs, from configs/config.yaml `labels:` (one source for the
    Python labeller, the exact flags and the generated .vada — see
    reasoner_client.ruleset_for_run). Defaults are the paper's values, tuned for a
    50-day window: a run with another window length should set them in its YAML."""
    from tslearning.config import CFG
    lab = CFG.get("labels") or {}
    tail = lab.get("tail_min", 2)
    if str(tail).lower() == "auto":
        tail = tail_min_auto(int(CFG["windows"]["financial"]["length"]),
                             float(CFG["predicates"]["crash_percentile"]) / 100.0,
                             float(lab.get("tail_alpha", 0.01)))
    return {"trend": float(lab.get("trend", 0.02)),
            "high_vol_pct": float(lab.get("high_vol_pct", 30.0)),
            "low_vol_pct": float(lab.get("low_vol_pct", 5.0)),
            "tail_min": int(tail)}


def tail_min_auto(window: int, p: float = 0.05, alpha: float = 0.01) -> int:
    """Smallest k such that k or more co-crash days in a window of `window` days would
    happen with probability <= alpha if the two series were INDEPENDENT.

    A crash day is a return below the series' qLow (its `p`-th percentile, 5%), so
    under independence each day is a joint crash with probability p**2 and the count is
    Binomial(window, p**2). tailDependent should mean "more joint crashes than chance".
    With p = 0.05 and alpha = 0.01: 2 for windows up to 59 days (the paper's 2 at
    W = 50), 3 for 60-175 days, 4 for 176-330.
    """
    from math import comb
    q = p * p
    tail = 1.0
    for k in range(0, window + 1):
        # tail = P(N >= k); stop at the first k where it is small enough
        if tail <= alpha:
            return max(k, 1)
        tail -= comb(window, k) * q ** k * (1 - q) ** (window - k)
    return window


def trend(window, up: "float | None" = None, down: "float | None" = None):
    """Return (label, sum) for one window. Cut-offs default to config labels.trend."""
    t = label_constants()["trend"]
    up = t if up is None else up
    down = -t if down is None else down
    s = float(np.sum(window))
    if s >= up:
        return "trendUp", s
    if s <= down:
        return "trendDown", s
    return "deadZone", s


def volatility(window, tau: float, high: "float | None" = None,
               low: "float | None" = None):
    """Return (label_or_None, pct-of-big-days) for one window. Cut-offs default to
    config labels.high_vol_pct / low_vol_pct."""
    c = label_constants()
    high = c["high_vol_pct"] if high is None else high
    low = c["low_vol_pct"] if low is None else low
    pct = 100.0 * float(np.mean(np.abs(np.asarray(window, float)) > tau))
    if pct >= high:
        return "highVol", pct
    if pct <= low:
        return "lowVol", pct
    return None, pct


def cocrash_count(win_x, win_y, qlow_x: float, qlow_y: float) -> int:
    """Number of days where BOTH series close below their qLow (same day)."""
    x, y = np.asarray(win_x, float), np.asarray(win_y, float)
    return int(np.sum((x <= qlow_x) & (y <= qlow_y)))


def emit_flag_csvs(sp_windows, bt_windows, timepoints, tau, qlow, out_dir,
                   seq_len: int = 50, tail_min: "int | None" = None) -> dict:
    """
    Write reasoner-schema predicate CSVs (trend / vol / tail) for the real windows.

    tau, qlow : dicts like {"sp500": .., "bitcoin": ..}
    Emits, in `out_dir`, one CSV per predicate that fires:
        trendUp/trendDown/deadZone, highVol/lowVol   -> columns X, (Sum|Pct), Start, End   (per series)
        tailDependent                                -> columns X, Y, Start, End           (per pair)
    Window i covers the interval [timepoints[i], timepoints[i] + seq_len).
    Names match nb_10_part_2 Part B: "sp500_real_{t}" / "btc_real_{t}".
    """
    tail_min = label_constants()["tail_min"] if tail_min is None else tail_min
    os.makedirs(out_dir, exist_ok=True)
    rows = defaultdict(list)
    for i, t in enumerate(timepoints):
        s, e = t, t + seq_len
        x, y = np.asarray(sp_windows[i], float), np.asarray(bt_windows[i], float)
        for win, series, tau_v in [(x, "sp500_real", tau["sp500"]),
                                    (y, "btc_real",   tau["bitcoin"])]:
            tlabel, ssum = trend(win)
            rows[tlabel].append({"X": f"{series}_{t}", "Sum": ssum, "Start": s, "End": e})
            vlabel, vpct = volatility(win, tau_v)
            if vlabel:
                rows[vlabel].append({"X": f"{series}_{t}", "Pct": vpct, "Start": s, "End": e})
        if cocrash_count(x, y, qlow["sp500"], qlow["bitcoin"]) >= tail_min:
            rows["tailDependent"].append(
                {"X": f"sp500_real_{t}", "Y": f"btc_real_{t}", "Start": s, "End": e})

    counts = {}
    for pred, rr in rows.items():
        pd.DataFrame(rr).to_csv(os.path.join(str(out_dir), f"{pred}.csv"), index=False)
        counts[pred] = len(rr)
    return counts
