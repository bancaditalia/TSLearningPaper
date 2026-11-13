"""
Macro state predicates — a monthly background state, mapped onto a daily pair's timeline.

This is what template T7 needs, and it is the one place in the
project where mixing time granularities is the *point* rather than a problem.

────────────────────────────────────────────────────────────────────────────────
WHAT THIS IS AND IS NOT
────────────────────────────────────────────────────────────────────────────────

NOT: correlating a daily series against a monthly one. That would force both onto one
clock — averaging equities down to monthly (discarding the daily detail the whole
pipeline rests on) or forward-filling CPI to daily (inventing ~20 fake observations per
real one). `datasets/pairs.check_same_frequency` refuses that, deliberately.

IS: a monthly series marks off stretches of the calendar — "inflation was rising in these
months" — and the daily pair is measured as usual *inside* those stretches. Nothing about
how correlation is computed changes. The macro fact occupies the same slot in a rule that
`highVol(X)` does in T1:

    T1:  highVol(sp500), trendDown(sp500)   condition, daily, from returns
    T7:  market(sp500,C), inflationHigh(C)   condition, monthly, from CPI, linked to the
                                             pair through the economy sp500 belongs to

Two intervals of different granularity can be intersected directly, because both are just
intervals. That is the whole argument for interval semantics, and no other template in the
library exercises it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tslearning.config import CFG, PATHS

# ── What "rising" means, and why the naive answer is wrong ────────────────────
# US_Inflation is already TRANSFORMED: logdiff of CPI, i.e. the monthly inflation RATE.
# So "value > 0" means "prices went up", which is true in 91% of months — a condition
# that covers 91% of the timeline cannot discriminate anything (the same vacuity that
# made T2's 1.00 meaningless). Three usable definitions instead:
#
#   "high"         rate above a percentile of the TRAIN period. The tightening-regime
#                  reading, and the one T7 wants: "while inflation was elevated".
#                  Balanced by construction, and threshold-fitted like tau/qLow.
#   "accelerating" the rate is increasing (first difference of the rate > 0). "Inflation
#                  rising" in the strict economic sense. Fires ~half the time.
#   "sign"         rate > 0. Kept only for completeness; near-vacuous for CPI, though
#                  meaningful for a series that genuinely changes sign (e.g. GDP growth).
DEFAULT_MODE = "high"
DEFAULT_PERCENTILE = 75.0
DEFAULT_DEADBAND = 0.0


def macro_state_intervals(series_name: str, lo=None, hi=None,
                          deadband: float = DEFAULT_DEADBAND,
                          smooth_months: int = 3, mode: str = DEFAULT_MODE,
                          percentile: float = DEFAULT_PERCENTILE,
                          train=None) -> dict:
    """Monthly high/low (or rising/falling) intervals for a macro series, as CALENDAR ranges.

    `mode` selects the definition — see the note at the top of this module; "high" is the
    default because a sign test on an already-differenced series is near-vacuous.

    `smooth_months` is a centred rolling mean applied first. Monthly macro data is noisy
    month to month, and without it the state flickers, producing dozens of one-month
    episodes instead of the multi-month regimes the economics is about (a tightening
    episode, not a single CPI print).

    The percentile for mode="high" is fitted on the TRAIN period only, like tau/qLow.

    Returns {"rising": [...], "falling": [...]} of (start_date, end_date) — the names are
    kept generic across modes so downstream code needs no special casing.
    """
    from tslearning.datasets.data_utils import load_transformed_macro

    full = load_transformed_macro()[series_name].dropna().sort_index()
    s = full.loc[lo:hi] if (lo is not None or hi is not None) else full
    if smooth_months > 1:
        s = s.rolling(smooth_months, center=True, min_periods=1).mean()

    if mode == "high":
        tr_lo, tr_hi = train or CFG["split"]["train"]
        ref = s.loc[tr_lo:tr_hi]
        ref = ref if len(ref) else s
        thr = float(np.percentile(ref.values, percentile))
        state = pd.Series(np.where(s.values > thr, "rising", "falling"), index=s.index)
    elif mode == "accelerating":
        ds = s.diff()
        state = pd.Series(np.where(ds.values > deadband, "rising",
                          np.where(ds.values < -deadband, "falling", "flat")),
                          index=s.index)
    elif mode == "sign":
        state = pd.Series(np.where(s.values > deadband, "rising",
                          np.where(s.values < -deadband, "falling", "flat")),
                          index=s.index)
    else:
        raise ValueError(f"mode must be high|accelerating|sign, got {mode!r}")

    out = {"rising": [], "falling": []}
    if state.empty:
        return {**out, "series": s}

    # coalesce consecutive months with the same state into one calendar interval
    cur, start = state.iloc[0], state.index[0]
    idx = list(state.index)
    for i in range(1, len(state) + 1):
        end_of_run = (i == len(state)) or (state.iloc[i] != cur)
        if end_of_run:
            # the state holds until the NEXT observation begins (monthly stamps are the
            # first of the month, so a run ending at month m covers through m's end)
            stop = idx[i] if i < len(idx) else idx[-1] + pd.offsets.MonthEnd(1)
            if cur in out:
                out[cur].append((start, stop))
            if i < len(state):
                cur, start = state.iloc[i], idx[i]
    return {**out, "series": s}


def to_day_indices(date_intervals, dates_map: dict) -> list:
    """Convert calendar date ranges to day indices on a pair's trading-day timeline.

    `dates_map` is the pair's {day_index: 'YYYY-MM-DD'} (from `<pair>_dates.csv`). A
    monthly interval covering, say, 2022-03-01..2022-06-01 becomes the day indices of the
    pair's trading days inside that span — so a 3-month macro state lands on ~63 daily
    facts without either series being resampled.
    """
    if not dates_map:
        return []
    ts = pd.Series({int(d): pd.Timestamp(v) for d, v in dates_map.items()}).sort_index()

    out = []
    for lo, hi in date_intervals:
        inside = ts[(ts >= pd.Timestamp(lo)) & (ts < pd.Timestamp(hi))]
        if inside.empty:
            continue
        days = sorted(inside.index)
        # split into contiguous runs of day indices (there should be exactly one, but a
        # gap in the pair's aligned calendar would otherwise be papered over)
        run_start = prev = days[0]
        for d in days[1:]:
            if d != prev + 1:
                out.append((run_start, prev + 1))
                run_start = d
            prev = d
        out.append((run_start, prev + 1))
    return out


def split_macro_name(series_name: str) -> "tuple[str, str]":
    """'US_Inflation' -> ('us', 'inflation'); 'DE_Inflation' -> ('de', 'inflation').

    The macro predicate is indexed by the ECONOMY (the country code) and named after the
    indicator, so `inflationHigh(C)` holds for C = "us", "de", ... and a rule reaches it
    through market(X, C). Naming the predicate after one country (the old
    `us_inflationHigh("us_inflation")`) left nothing in the rule linking it to the pair.
    """
    country, _, indicator = series_name.partition("_")
    if not indicator:
        raise ValueError(f"expected '<COUNTRY>_<Indicator>', got {series_name!r}")
    return country.lower(), indicator[0].lower() + indicator[1:]


def emit_macro_state_csvs(series_name: str, pair: str, out_dir, dates_map: dict,
                          lo=None, hi=None, deadband: float = DEFAULT_DEADBAND,
                          smooth_months: int = 3, mode: str = DEFAULT_MODE,
                          percentile: float = DEFAULT_PERCENTILE,
                          verbose: bool = True) -> dict:
    """Write `<indicator>High/<indicator>Low` CSVs on the pair's day timeline.

    Rows are keyed by the economy (`X` = "us_real_<day>", read back as "us"), so one file
    can hold several countries: rows for other countries already in the file are kept,
    rows for this one are replaced. Output schema matches every other unary predicate
    (X, Start, End), so the rule layer reads it with no special casing.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    st = macro_state_intervals(series_name, lo, hi, deadband, smooth_months,
                               mode=mode, percentile=percentile)
    country, indicator = split_macro_name(series_name)
    # name the predicate after the DEFINITION, so the rule text says what it means
    hi_name, lo_name = {"high": ("High", "Low"),
                        "accelerating": ("Accelerating", "Decelerating"),
                        "sign": ("Rising", "Falling")}[mode]
    counts = {}
    for state, pred in (("rising", f"{indicator}{hi_name}"),
                        ("falling", f"{indicator}{lo_name}")):
        days = to_day_indices(st[state], dates_map)
        rows = pd.DataFrame([{"X": f"{country}_real_{s}", "Start": s, "End": e}
                             for s, e in days], columns=["X", "Start", "End"])
        path = out_dir / f"{pred}_{pair}.csv"
        if path.exists():
            old = pd.read_csv(path)
            keep = old[~old["X"].astype(str).str.startswith(f"{country}_real_")]
            rows = pd.concat([keep, rows], ignore_index=True)
        rows.to_csv(path, index=False)
        counts[pred] = {"country": country, "intervals": len(days),
                        "days": sum(e - s for s, e in days)}
        if verbose:
            print(f"    {pred}({country}){'':14s} {len(days):3d} intervals  "
                  f"{counts[pred]['days']:5d} days")
    return counts


def run(pair: str = "sp500_gold", series_name: str = "US_Inflation",
        source: str = "py", deadband: float = DEFAULT_DEADBAND,
        smooth_months: int = 3, mode: str = DEFAULT_MODE,
        percentile: float = DEFAULT_PERCENTILE, verbose: bool = True) -> dict:
    """Add a macro state to an already-labelled pair, in place.

    Reads the pair's `<pair>_dates.csv` to learn its trading-day calendar, then writes the
    macro predicate CSVs into the same folder so `load_pair_facts` picks them up.
    """
    from tslearning.rules.generate import SOURCES

    d = PATHS["data_processed"] / SOURCES[source] / pair
    dates_csv = d / f"{pair}_dates.csv"
    if not dates_csv.exists():
        raise FileNotFoundError(
            f"{dates_csv} missing — label the pair first with prep_real_pairs.py")

    df = pd.read_csv(dates_csv)
    dates_map = dict(zip(df["day"], df["date"]))
    lo, hi = df["date"].min(), df["date"].max()

    if verbose:
        print(f"  macro state {series_name} -> {pair} timeline ({lo}..{hi})")
    counts = emit_macro_state_csvs(series_name, pair, d, dates_map, lo, hi,
                                   deadband, smooth_months, mode=mode,
                                   percentile=percentile, verbose=verbose)
    return {"pair": pair, "series": series_name, "out_dir": str(d), "counts": counts}


def run_configured(pair: str, source: str = "py", verbose: bool = True) -> dict:
    """Recompute every macro state listed in `real_pairs.macro_states` for one pair.

    Called right after a financial pair is labelled, because the macro state is mapped
    onto the pair's own calendar and goes stale the moment that calendar changes.
    """
    from tslearning.datasets.pairs import describe

    if describe(pair)["freq"] != "financial":
        return {}
    out = {}
    for series in (CFG.get("real_pairs") or {}).get("macro_states") or []:
        out[series] = run(pair=pair, series_name=series, source=source, verbose=verbose)
    return out
