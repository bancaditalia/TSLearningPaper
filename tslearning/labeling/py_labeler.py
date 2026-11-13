"""
Pure-Python labeler for real pairs — the reasoner bypass.

Reproduces preLabeling.vada's correlation concordance (lines 38-82) exactly in Python, plus
the exact market-state flags (exact_flags.py), so a pair can be labelled with NO reasoner run.

Validated against the reasoner's own output on sp500_gold (171 windows):
    same-sign concordance : 100% exact
    diff-sign concordance : 95% exact (max off-by-2pp)
    regime label agreement: 96.5%

Regime, same as recalibrate.py, is decided by relative dominance of the two concordances:
    score = sameSign% - diffSign% ;  > +deadband -> positive ; < -deadband -> negative ; else none.
Output goes to <out_dir>/<pair>/{predicate}_{pair}.csv in the same schema nb_11 reads, so the
reasoner path and this path are interchangeable downstream.
"""
from __future__ import annotations
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from tslearning.labeling.exact_flags import trend, volatility, cocrash_count


def CFG_labels() -> dict:
    from tslearning.config import CFG
    return CFG.get("labels") or {}

REGIMES = ["isPositiveCorrelation", "isNegativeCorrelation", "isNoCorrelation"]


def compute_concordance(x, y, denom: int = 50):
    """Replicate preLabeling.vada's magnitude+sign concordance for one 50-day window.

    magnitudeConsistent day t: |X| and |Y| both rise, or both fall, from t-1 to t.
    Among those, split by sign of X_t * Y_t. Returns (sameSign%, diffSign%), each /denom*100.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    ax, ay = np.abs(x), np.abs(y)
    same = diff = 0
    for t in range(1, len(x)):
        mag = (ax[t] > ax[t-1] and ay[t] > ay[t-1]) or (ax[t] < ax[t-1] and ay[t] < ay[t-1])
        if not mag:
            continue
        s = x[t] * y[t]
        if s > 0:
            same += 1
        elif s < 0:
            diff += 1
    return same * 100.0 / denom, diff * 100.0 / denom


def _aligned(series_returns: dict, lo, hi):
    idx = None
    for s in series_returns.values():
        i = s.dropna().sort_index().loc[lo:hi].index
        idx = i if idx is None else idx.intersection(i)
    return idx.sort_values()


def _dominance(x, y, seq_len: int) -> float:
    """sameSign% − diffSign% for two aligned windows."""
    same, diff = compute_concordance(x, y, denom=seq_len)
    return same - diff


def default_deadband() -> float:
    """The regime cut, from configs/config.yaml `labels.deadband`.

    The same constant is used by relabel.py for the classifier's training labels and
    passed to preLabeling.vada as ${Deadband}. Callers that pass an explicit value are
    unaffected — nb_prep_real_pairs passes 5.0 explicitly and keeps behaving as it did.
    """
    from tslearning.config import CFG
    return float(CFG.get("labels", {}).get("deadband", 10.0))


def lead_label(x, y, t, lag, seq_len: int = 50, deadband: "float | None" = None,
               net: bool = True, require_direction: bool = True):
    """Concordance of X at [t, t+seq_len) against Y shifted `lag` ahead.

    X leads Y positively if X's move today agrees with Y's move `lag` days later. Same concordance measure and
    same dominance rule as lag 0, so lagged and unlagged labels are comparable.

    Two corrections, both on by default — without them
    overlapping windows manufacture leads that are not there:

    `net=True` scores the NET lead, score(k) − score(0). At lag 1 the two 50-day
    windows share 49 days, so the raw lagged score mostly re-measures lag 0; a pair with
    strong contemporaneous correlation would score high at every small lag with no lead
    involved. Only the extra agreement gained by shifting Y speaks to lead-lag.

    `require_direction=True` demands asymmetry: X→Y must beat Y→X at the same lag.
    A lead is a directional claim; if both directions score alike, what you have is
    contemporaneous correlation smeared across lags, and the label is `noLead`.

    Returns (label, score) where score is the net lead when net=True, or None if the
    shifted window runs off either end.
    """
    deadband = default_deadband() if deadband is None else deadband
    n = min(len(x), len(y))
    if t + lag + seq_len > n or t + seq_len > n:
        return None

    fwd = _dominance(x[t:t + seq_len], y[t + lag:t + lag + seq_len], seq_len)

    if net:
        base = _dominance(x[t:t + seq_len], y[t:t + seq_len], seq_len)
        score = fwd - base
    else:
        score = fwd

    if require_direction and lag > 0:
        # Y leading X at the same lag: Y today vs X `lag` days later.
        rev = _dominance(y[t:t + seq_len], x[t + lag:t + lag + seq_len], seq_len)
        if net:
            rev -= _dominance(y[t:t + seq_len], x[t:t + seq_len], seq_len)
        if abs(rev) >= abs(score):
            return "noLead", score

    label = ("leadsPositively" if score > deadband
             else "leadsNegatively" if score < -deadband
             else "noLead")
    return label, score


def label_pair(pair, sp_returns, other_returns, other_name, tau, qlow, out_dir,
               lo, hi, stride: int = 10, seq_len: int = 50,
               deadband: "float | None" = None,
               tail_min: "int | None" = None, base_name: str = "sp500", lags=None,
               net_lead: bool = True, require_direction: bool = True,
               window_thresholds=None):
    """Label one pair (`base_name` vs `other_name`) fully in Python and write
    per-predicate CSVs.

    tau, qlow: dicts keyed by lowercase series name ("sp500", "gold", ...).
    Returns a summary dict. Names/schema match the reasoner path (sp500_real_{t}).

    `base_name` defaults to "sp500" so existing callers are unaffected; set it to
    label a pair whose first side is something else (e.g. a macro pair like
    US_Inflation x US_Unemployment, where base_name="us_inflation").

    `lags` enables the lead-lag predicates (leadsPositively / leadsNegatively /
    noLead), emitted with a Lag column as `lag_bin` expects. `net_lead` and
    `require_direction` are the two corrections — see lead_label. None (the default)
    skips them, so existing callers are unaffected; pass CFG["predicates"]["lags"]
    or e.g. [1,2,3,4,5] to compute them. These are the rules dropped from the .vada
    for costing ~3x the runtime — here they are a few array slices.

    `window_thresholds` = (tau_a, q_a, tau_b, q_b), one value per window, overrides
    `tau`/`qlow` and labels.thresholds (used by eval/simulation.py).
    """
    deadband = default_deadband() if deadband is None else deadband
    if tail_min is None:
        from tslearning.labeling.exact_flags import label_constants
        tail_min = label_constants()["tail_min"]
    out_dir = Path(out_dir); dst = out_dir / pair; os.makedirs(dst, exist_ok=True)
    idx = _aligned({base_name: sp_returns, other_name: other_returns}, lo, hi)
    sp = sp_returns.loc[idx].to_numpy(float)
    ot = other_returns.loc[idx].to_numpy(float)
    N = len(idx)
    tps = list(range(0, N - seq_len + 1, stride))

    # tau / qLow per window (labels.thresholds): the train value of each series, or the
    # rolling percentile of the returns before the window starts (thresholds.py)
    from tslearning.labeling import thresholds as thr
    starts = [idx[t] for t in tps]
    if window_thresholds is not None:
        # given by the caller, one value per window (simulation.py: the simulated
        # series have no history in data/processed, so the caller computes them)
        tau_a, q_a, tau_b, q_b = (np.asarray(v, float) for v in window_thresholds)
        if not all(len(v) == len(tps) for v in (tau_a, q_a, tau_b, q_b)):
            raise ValueError(f"window_thresholds: need {len(tps)} values per array")
    elif thr.mode() == "rolling" and tps:
        tau_a, q_a, fb_a = thr.rolling_values(base_name, starts)
        tau_b, q_b, fb_b = thr.rolling_values(other_name, starts)
        if fb_a or fb_b:
            print(f"  {pair}: {max(fb_a, fb_b)} of {len(tps)} windows have under "
                  f"{(CFG_labels().get('rolling_min_days', 250))} days of history; "
                  f"they use the train thresholds")
    else:
        tau_a = np.full(len(tps), tau[base_name]); q_a = np.full(len(tps), qlow[base_name])
        tau_b = np.full(len(tps), tau[other_name]); q_b = np.full(len(tps), qlow[other_name])

    rows = defaultdict(list)
    counts = {r: 0 for r in REGIMES}
    lead_counts = {lag: defaultdict(int) for lag in (lags or [])}
    for i_t, t in enumerate(tps):
        s, e = t, t + seq_len
        x, y = sp[t:t+seq_len], ot[t:t+seq_len]

        # --- correlation regime (validated concordance + dominance) ---
        same, dsd = compute_concordance(x, y, denom=seq_len)
        score = same - dsd
        reg = ("isPositiveCorrelation" if score > deadband
               else "isNegativeCorrelation" if score < -deadband
               else "isNoCorrelation")
        rows[reg].append({"X": f"{base_name}_real_{t}", "Y": f"{other_name}_real_{t}",
                          "Start": s, "End": e}); counts[reg] += 1

        # --- exact flags (per series) ---
        for win, series, tv in [(x, base_name, tau_a[i_t]), (y, other_name, tau_b[i_t])]:
            tl, ssum = trend(win)
            rows[tl].append({"X": f"{series}_real_{t}", "Sum": ssum, "Start": s, "End": e})
            vl, vp = volatility(win, tv)
            if vl:
                rows[vl].append({"X": f"{series}_real_{t}", "Pct": vp, "Start": s, "End": e})

        # --- tail dependence (co-crash) ---
        if cocrash_count(x, y, q_a[i_t], q_b[i_t]) >= tail_min:
            rows["tailDependent"].append({"X": f"{base_name}_real_{t}", "Y": f"{other_name}_real_{t}",
                                          "Start": s, "End": e})

        # --- lead-lag: does X at t agree with Y at t+lag? ---
        for lag in (lags or []):
            res = lead_label(sp, ot, t, lag, seq_len=seq_len, deadband=deadband,
                             net=net_lead, require_direction=require_direction)
            if res is None:
                continue
            label, score = res
            rows[label].append({"X": f"{base_name}_real_{t}", "Y": f"{other_name}_real_{t}",
                                "Lag": lag, "Pct": score, "Start": s, "End": e})
            lead_counts[lag][label] += 1

    for pred, rr in rows.items():
        pd.DataFrame(rr).to_csv(dst / f"{pred}_{pair}.csv", index=False)
    # day -> date map, for reporting break dates later
    pd.DataFrame({"day": range(N), "date": idx.astype(str)}).to_csv(dst / f"{pair}_dates.csv", index=False)

    out = {"pair": pair, "aligned_days": N, "windows": len(tps),
           "regime_counts": counts, "predicates_written": sorted(rows), "out_dir": str(dst)}
    if lags:
        out["lead_counts"] = {lag: dict(c) for lag, c in lead_counts.items()}
        # Strongest lag = the one firing a directional verdict most often; §3.1(d)
        # says keep that one rather than reporting all of them.
        out["strongest_lag"] = max(
            lead_counts,
            key=lambda l: lead_counts[l]["leadsPositively"] + lead_counts[l]["leadsNegatively"],
            default=None)
    return out


def retire_pair_dir(out_dir, pair: str, verbose: bool = True) -> "Path | None":
    """Move an existing `<out_dir>/<pair>/` aside before relabelling it.

    Labelling overwrites only the files it writes; anything else left in the folder (old
    lead-lag files, an old macro state) stays behind on the PREVIOUS run's day index —
    which is how T4/T7 got scored on misaligned days (2026-09-07). Moved, not deleted,
    to data/processed/_stale/<pair>_<timestamp>/.
    """
    import shutil
    from datetime import datetime
    from tslearning.config import PATHS

    src = Path(out_dir) / pair
    if not src.exists():
        return None
    dst = PATHS["data_processed"] / "_stale" / \
        f"{Path(out_dir).name}_{pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    if verbose:
        print(f"  previous {pair} labels moved to {dst.relative_to(PATHS['data_processed'])}")
    return dst
