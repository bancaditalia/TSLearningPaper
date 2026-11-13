"""
The tau / qLow thresholds the volatility and tail predicates compare each day against —
per WINDOW, for the reasoner and the Python labeller alike.

    tau   the vol_percentile-th percentile of |return|   -> a "big move" day   (highVol/lowVol)
    qLow  the crash_percentile-th percentile of return   -> a "crash" day      (tailDependent)

Two ways to set them (config `labels.thresholds`):

  train    (default; the paper) one value per series, fitted on the run's training period
           (transfer_set.fit_thresholds -> thresholds/tau.csv, qlow.csv).
  rolling  one value per WINDOW: the percentile of the series' returns over the
           `labels.rolling_days` trading days BEFORE the window starts (default 500, about
           two years). Uses only the past, so no look-ahead; "big move" then means big
           compared with recent conditions, so a calm decade after a crisis still has
           volatile windows. Falls back to the train value when a series has fewer than
           `labels.rolling_min_days` days of history before the window.

WHY PER WINDOW (fixed 2026-10-21). preLabeling.vada joins `tau(X,T), flow(X,K,V)` on X.
The batches name each window (`sp500_real_42`), but the thresholds file named each SERIES
(`sp500`), so the join never matched: every engine run labelled every window lowVol,
never highVol, never a crash. Each batch now gets its own threshold files with one row
per window name and that window's time block — the program is unchanged.
Synthetic windows (transfer sets) have no calendar date: they always use the train value.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from tslearning.config import CFG

_SUFFIX = re.compile(r"^(?P<key>.+?)_(?:synth|real)_\d+$")


def mode() -> str:
    m = str((CFG.get("labels") or {}).get("thresholds", "train")).lower()
    if m not in ("train", "rolling"):
        raise ValueError(f"labels.thresholds must be 'train' or 'rolling', got {m!r}")
    return m


def series_key(window_name: str) -> str:
    """'sp500_real_42' -> 'sp500', 'btc_synth_3' -> 'bitcoin', 'us_inflation_real_5' ->
    'us_inflation'; a bare series key (verify.py's continuous EDB) is returned as is."""
    from tslearning.labeling.transfer_set import SHORT_NAME
    m = _SUFFIX.match(str(window_name))
    key = m.group("key") if m else str(window_name)
    inverse = {short: long for long, short in SHORT_NAME.items()}
    return inverse.get(key, key)


def train_values() -> "tuple[dict, dict]":
    from tslearning.labeling.transfer_set import load_thresholds
    return load_thresholds()


@lru_cache(maxsize=None)
def _history(key: str) -> pd.Series:
    """The series' full transformed history (returns), indexed by date."""
    from tslearning.datasets.pairs import load_series
    lut = {n.lower(): n for n in CFG["series_freq"]}
    name = lut.get(key, key)
    return load_series([name])[name].dropna().sort_index()


def rolling_values(key: str, dates) -> "tuple[np.ndarray, np.ndarray, int]":
    """(tau, qlow, n_fallback) for windows starting on `dates`: percentiles of the
    `rolling_days` returns strictly before each date; the train value where the history
    is shorter than `rolling_min_days`."""
    lab = CFG.get("labels") or {}
    days = int(lab.get("rolling_days", 500))
    min_days = int(lab.get("rolling_min_days", 250))
    vol_p = float(CFG["predicates"]["vol_percentile"])
    crash_p = float(CFG["predicates"]["crash_percentile"])
    r = _history(key)
    tau_tr, q_tr = train_values()
    pos = r.index.searchsorted(pd.DatetimeIndex(pd.to_datetime(dates)), side="left")
    tau, q, fb = [], [], 0
    vals = r.to_numpy(float)
    for p in pos:                                  # returns before the window's first day
        past = vals[max(0, p - days):p]
        if len(past) < min_days:
            tau.append(tau_tr[key]); q.append(q_tr[key]); fb += 1
        else:
            tau.append(float(np.percentile(np.abs(past), vol_p)))
            q.append(float(np.percentile(past, crash_p)))
    return np.asarray(tau), np.asarray(q), fb


def window_values(keys_dates: dict) -> "tuple[dict, dict]":
    """{window_name: (series_key, start_date | None)} -> ({window: tau}, {window: qlow}).
    A window without a date, or mode 'train', gets its series' train value."""
    tau_tr, q_tr = train_values()
    tau, q = {}, {}
    if mode() == "rolling":
        by_key = {}
        for w, (k, d) in keys_dates.items():
            if d is not None:
                by_key.setdefault(k, []).append((w, d))
        for k, items in by_key.items():
            t, qq, _ = rolling_values(k, [d for _, d in items])
            for (w, _), tv, qv in zip(items, t, qq):
                tau[w], q[w] = float(tv), float(qv)
    for w, (k, _) in keys_dates.items():
        if w not in tau:
            if k not in tau_tr:
                raise KeyError(f"no fitted threshold for series '{k}' (window {w}); "
                               f"fit_thresholds must include it")
            tau[w], q[w] = tau_tr[k], q_tr[k]
    return tau, q


def _thr_paths(batch_csv) -> "tuple[Path, Path]":
    b = Path(batch_csv)
    d = b.parent / "_thr"
    return d / f"tau_{b.stem}.csv", d / f"qlow_{b.stem}.csv"


def write_for_batch(batch_csv, window_dates: "dict | None" = None,
                    overwrite: bool = False) -> "tuple[Path, Path]":
    """Write the per-window threshold files of one batch (preLabeling.vada's T and Q):
    <batch dir>/_thr/tau_<batch>.csv and qlow_<batch>.csv, columns series,tau|qlow,start,
    end — `series` holding the WINDOW name, start/end its time block in the batch.

    `window_dates` ({window_name: first date}) is given by writers that know the
    calendar (real pairs); without it every window gets its series' train value. An
    existing pair of files is kept unless `overwrite` (the batch writer's own files win
    over the constant ones reasoner_client would generate)."""
    from tslearning.config import PATHS
    tau_p, q_p = _thr_paths(batch_csv)
    fitted = PATHS["data_processed"] / "thresholds" / "tau.csv"
    dated = tau_p.with_suffix(".dated")    # marker: written by a calendar-aware writer
    if tau_p.exists() and q_p.exists() and not overwrite and (
            dated.exists() or not fitted.exists()
            or tau_p.stat().st_mtime >= fitted.stat().st_mtime):
        return tau_p, q_p                  # the writer's own, or newer than the fit
    df = pd.read_csv(batch_csv, usecols=["window_name", "start_time", "end_time"])
    blocks = df.groupby("window_name").agg(start=("start_time", "min"),
                                           end=("end_time", "max"))
    window_dates = window_dates or {}
    tau, q = window_values({w: (series_key(w), window_dates.get(w))
                            for w in blocks.index})
    tau_p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"series": blocks.index, "tau": [tau[w] for w in blocks.index],
                  "start": blocks.start.values, "end": blocks.end.values}
                 ).to_csv(tau_p, index=False)
    pd.DataFrame({"series": blocks.index, "qlow": [q[w] for w in blocks.index],
                  "start": blocks.start.values, "end": blocks.end.values}
                 ).to_csv(q_p, index=False)
    if window_dates:
        dated.write_text(f"thresholds: {mode()}\n")
    elif dated.exists():
        dated.unlink()
    return tau_p, q_p
