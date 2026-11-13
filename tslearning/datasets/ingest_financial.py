"""
Financial data ingestion — the library half of `scripts/data_prep.py -d financial`.

What it does, in order:
  1. download (or read from cache) the yfinance price series          -> data/raw/{name}_raw.csv
  2. transform to log-returns (VIX: log-level)                         -> transformed.pkl
  3. stylized-fact fingerprints per series                             -> fingerprints.pkl
  4. cut into overlapping windows (length/step from config.yaml)       -> windowed.pkl
  5. build the date-aligned SP500 x Bitcoin window pairs               -> aligned_windows_real.pkl

Everything the notebook plots is deliberately left out — figures are a notebook
concern, not a pipeline one.

Heavy / optional dependencies (yfinance, scipy, statsmodels) are imported inside
the functions that need them, so `--help` and unit tests work without them.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from tslearning.config import CFG, PATHS

# ── The catalog ────────────────────────────────────
# Download period: config `data.financial_start` / `financial_end` (default 2002-2024).
# A series' own "start" is a FLOOR: Bitcoin stays from 2017, as the paper's data is.
TICKERS = {
    "SP500":   {"ticker": "^GSPC"},
    "Bitcoin": {"ticker": "BTC-USD", "start": "2017-01-01"},
    "VIX":     {"ticker": "^VIX"},
    "Gold":    {"ticker": "GC=F"},
    "NASDAQ":  {"ticker": "^IXIC"},
    # US 10-year Treasury note futures, continuous front contract (CBOT ZN), like Gold's
    # GC=F: prices, so log-returns; same roll caveat. Added 2026-10-20 for the
    # stock-bond pair (T6 regime reversal, T7 macro conditioning).
    "UST10Y":  {"ticker": "ZN=F"},
    # iShares 20+ Year Treasury Bond ETF: the bond series most used in stock-bond
    # safe-haven studies; trades from mid-2002. Adjusted close (auto_adjust=True), so the
    # monthly distributions are in the returns — and Yahoo REVISES the adjusted history
    # after every new distribution: keep the cached file, extend with --extend only.
    "TLT":     {"ticker": "TLT"},
    # US 10-year ZERO-COUPON Treasury, from the Fed's fitted yield curve (Gurkaynak, Sack &
    # Wright 2007; data/raw/feds200628.csv, downloaded by hand from
    # federalreserve.gov/econres/feds/the-us-treasury-yield-curve-1961-to-the-present.htm).
    # Not a traded price: the daily return of a constant-maturity zero is derived from the
    # yield (see gsw_returns). No contract rolls, no distribution adjustments.
    "UST10Z":  {"source": "gsw", "column": "SVENY10", "maturity": 10},
}

GSW_FILE = "feds200628.csv"


def period(name: str) -> "tuple[str, str]":
    """(start, end) to download for one series: the config period, never before the
    series' own floor."""
    d = CFG.get("data") or {}
    start = d.get("financial_start", "2002-01-01")
    floor = TICKERS[name].get("start")
    if floor and pd.Timestamp(floor) > pd.Timestamp(start):
        start = floor
    return start, d.get("financial_end", "2024-12-31")

# Which transform each series gets. Returns for prices, log-level for VIX (it is
# already a volatility index, not a price).
RETURN_SERIES = ["SP500", "NASDAQ", "Bitcoin", "Gold", "UST10Y", "TLT", "UST10Z"]
LEVEL_SERIES = ["VIX"]

DEFAULT_SERIES = list(TICKERS)


# ══════════════════════════════════════════════════════════════════════════════
#  1. Download / cache
# ══════════════════════════════════════════════════════════════════════════════

def gsw_returns(yields_pct: pd.Series, maturity: float) -> pd.Series:
    """Daily log return of a constant-maturity zero-coupon bond from its yield.

    y in percent, continuously compounded (as in the GSW file). Holding an n-year zero
    from one day to the next:
        r_t = -n * (y_t - y_{t-1}) + y_{t-1} * dt,     dt = calendar days / 365
    the price effect of the yield change (duration n) plus the day's accrual. Yields up
    -> negative return, as for a traded bond price.
    """
    y = yields_pct.dropna().sort_index() / 100.0
    dt = y.index.to_series().diff().dt.days / 365.0
    r = -maturity * y.diff() + y.shift(1) * dt
    return r.dropna()


def _gsw(name: str, raw_dir: Path, start: str, end: str) -> pd.DataFrame:
    """A GSW series as a raw frame with a total-return index in "Close", so the usual
    transform (log returns of Close) gives back gsw_returns exactly."""
    import numpy as np
    import gzip
    # the committed snapshot is gzipped; a fresh hand download is the plain CSV
    candidates = [raw_dir / GSW_FILE, raw_dir / (GSW_FILE + ".gz")]
    f = next((c for c in candidates if c.exists()), None)
    if f is None:
        raise FileNotFoundError(
            f"{candidates[0]} (or .gz) not found: download feds200628.csv from "
            f"https://www.federalreserve.gov/econres/feds/"
            f"the-us-treasury-yield-curve-1961-to-the-present.htm into data/raw/")
    if all(c.exists() for c in candidates):
        print(f"[GSW] both {GSW_FILE} and {GSW_FILE}.gz in {raw_dir} — using the plain "
              f"CSV; remove one to be sure which snapshot the results come from")
    opener = gzip.open if f.suffix == ".gz" else open
    with opener(f, "rt") as fh:
        head = next(i for i, l in enumerate(fh) if l.startswith("Date"))
    d = pd.read_csv(f, skiprows=head, index_col=0, parse_dates=True)
    spec = TICKERS[name]
    y = d[spec["column"]].loc[start:end]
    r = gsw_returns(y, spec["maturity"])
    idx = np.exp(np.concatenate([[0.0], r.cumsum().to_numpy()]))
    dates = [y.dropna().sort_index().index[0]] + list(r.index)
    return pd.DataFrame({"Close": idx}, index=pd.DatetimeIndex(dates, name="Date"))


def _yahoo(name: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf  # network dependency, imported only when needed
    return yf.download(TICKERS[name]["ticker"], start=start, end=end,
                       auto_adjust=True, progress=False)


def _data_lines(path: Path) -> "tuple[list, dict]":
    """(header lines, {date: data line}) of a raw CSV as yfinance writes it (3 header
    lines: Price / Ticker / Date)."""
    lines = path.read_text().splitlines()
    head = [l for l in lines[:3]]
    body = {}
    for l in lines[3:]:
        if l.strip():
            body[l.split(",", 1)[0]] = l
    return head, body


def extend_raw(name: str, raw_dir: Path, tol: float = 1e-9, verbose: bool = True) -> str:
    """Add the days before data/raw/<name>_raw.csv starts, back to the config start,
    WITHOUT touching the rows already there.

    The longer history is downloaded into data/raw/_extend/<name>_raw.csv first and its
    overlap with the current file is compared value by value. Only if every overlapping
    value agrees (relative difference <= tol) are the earlier rows put in front of the
    current file, whose own lines are kept byte for byte (a backup is kept in
    data/raw/_backup/). Otherwise nothing changes and the differences are reported —
    Yahoo sometimes revises past prices, and the paper's numbers rest on these files.
    Returns a one-line summary.
    """
    import datetime as _dt
    import numpy as np

    cache = raw_dir / f"{name}_raw.csv"
    start, end = period(name)
    head, cur = _data_lines(cache)
    first = min(cur)
    if pd.Timestamp(first) <= pd.Timestamp(start) + pd.Timedelta(days=10):
        return f"{name}: already from {first}, nothing to add"

    tmp = raw_dir / "_extend" / f"{name}_raw.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    _yahoo(name, start, end).to_csv(tmp)
    head_new, new = _data_lines(tmp)
    if head_new != head:
        return f"{name}: NOT extended — the new download has other columns ({tmp})"

    overlap = sorted(set(cur) & set(new))
    if not overlap:
        return f"{name}: NOT extended — no overlapping days to compare ({tmp})"
    to_num = lambda line: np.array([float(x) if x else np.nan
                                    for x in line.split(",")[1:]], float)
    worst, n_diff = 0.0, 0
    for d in overlap:
        a, b = to_num(cur[d]), to_num(new[d])
        rel = np.nanmax(np.abs(a - b) / np.maximum(np.abs(a), 1e-12)) if len(a) else 0.0
        if rel > tol:
            n_diff += 1
            worst = max(worst, float(rel))
    if n_diff:
        return (f"{name}: NOT extended — {n_diff} of {len(overlap)} overlapping days differ "
                f"(largest relative difference {worst:.2e}). Current file kept; the new "
                f"download is in {tmp} for inspection.")

    earlier = [new[d] for d in sorted(new) if d < first]
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = raw_dir / "_backup" / f"{name}_raw_{stamp}.csv"
    bak.parent.mkdir(parents=True, exist_ok=True)
    bak.write_text(cache.read_text())
    cache.write_text("\n".join(head + earlier + [cur[d] for d in sorted(cur)]) + "\n")
    return (f"{name}: extended by {len(earlier)} days ({min(new)} -> {first}); "
            f"{len(overlap)} overlapping days identical; backup {bak.name}")


def download_prices(series: "list[str] | None" = None, force: bool = False,
                    raw_dir: "Path | None" = None, verbose: bool = True,
                    extend: bool = False) -> dict:
    """Return {name: raw price DataFrame}, reading data/raw/{name}_raw.csv when it
    exists unless `force=True`. Only touches the network for missing/forced series.

    `extend=True`: a cached file that starts after the config start is extended
    backwards with `extend_raw` (existing rows never change). Without it, such a file is
    used as is and a note says so."""
    raw_dir = Path(raw_dir or PATHS["data_raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    names = series or DEFAULT_SERIES

    raw = {}
    for name in names:
        start, end = period(name)
        if TICKERS[name].get("source") == "gsw":       # local file, nothing to download
            df = _gsw(name, raw_dir, start, end)
            if verbose:
                print(f"[GSW]        {name:8s}: {len(df):,} rows from {GSW_FILE}")
            raw[name] = df
            continue
        cache = raw_dir / f"{name}_raw.csv"
        if cache.exists() and not force:
            if extend:
                msg = extend_raw(name, raw_dir, verbose=verbose)
                if verbose:
                    print(f"[EXTEND]     {msg}")
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            if verbose:
                print(f"[CACHE]      {name:8s}: {len(df):,} rows")
            first = pd.to_datetime(df.index, errors="coerce").dropna().min()
            if not extend and first > pd.Timestamp(start) + pd.Timedelta(days=10):
                print(f"note: {name} on disk starts {first.date()}, the config asks for "
                      f"{start}; run data_prep.py --extend to add the earlier days "
                      f"(the days already on disk are kept unchanged)")
        else:
            df = _yahoo(name, start, end)
            df.to_csv(cache)
            if verbose:
                print(f"[DOWNLOADED] {name:8s}: {len(df):,} rows")
        raw[name] = df
    return raw


# ══════════════════════════════════════════════════════════════════════════════
#  2. Transform
# ══════════════════════════════════════════════════════════════════════════════

def extract_close(df: pd.DataFrame) -> pd.Series:
    """Flatten yfinance's MultiIndex columns to a plain numeric Close series."""
    col = df["Close"]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return pd.to_numeric(col, errors="coerce").dropna().astype(float)


def transform_prices(raw: dict, verbose: bool = True) -> "tuple[dict, pd.DataFrame | None]":
    """Return (transformed, sp500_ohlcv_log_returns).

    transformed maps series name -> stationary series (log-returns, or log-level
    for VIX).
    """
    from tslearning.datasets.data_utils import log_returns, log_level

    close = {name: extract_close(df) for name, df in raw.items()}

    transformed = {}
    for name, s in close.items():
        transformed[name] = log_level(s) if name in LEVEL_SERIES else log_returns(s)
        transformed[name].index = pd.to_datetime(transformed[name].index)

    # SP500 OHLCV log-returns (used by nb_03 / nb_04)
    ohlcv = None
    if "SP500" in raw:
        df_sp = raw["SP500"]
        if isinstance(df_sp.columns, pd.MultiIndex):
            df_sp = df_sp.droplevel(1, axis=1)
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df_sp.columns]
        ohlcv = (df_sp[cols].apply(pd.to_numeric, errors="coerce").dropna()
                 .apply(log_returns))

    if verbose:
        for name, s in transformed.items():
            print(f"{name:12s}: {len(s):,} obs  "
                  f"[{s.index[0].date()} -> {s.index[-1].date()}]")
    return transformed, ohlcv


# ══════════════════════════════════════════════════════════════════════════════
#  3. Fingerprints  +  4. Windows
# ══════════════════════════════════════════════════════════════════════════════

def compute_fingerprints(transformed: dict, verbose: bool = True) -> dict:
    """Stylized-fact fingerprint per series."""
    from tslearning.datasets.data_utils import compute_fingerprint

    out = {}
    for name, s in transformed.items():
        fp = compute_fingerprint(s, name)
        out[name] = fp
        if verbose:
            print(f"{name:12s} n={fp['n_obs']:>6,}  std={fp['std']:.4f}  "
                  f"kurt={fp['excess_kurtosis']:+.2f}  lev={fp['leverage_corr']:+.3f}")
    return out


def build_windows(transformed: dict, verbose: bool = True) -> dict:
    """Window every series with its configured (length, step).

    split="fraction" (the default) is deliberate: ALL windows, full history —
    this is the synthesizer's training data. The calendar split is applied later,
    only to the classifier's real input..
    """
    from tslearning.datasets.data_utils import synth_windows, window_for

    windowed = {}
    for name, s in transformed.items():
        try:
            length, step = window_for(name)
        except KeyError:
            if verbose:
                print(f"{name:12s} not in config.series_freq -> skipped")
            continue
        tr, te = synth_windows(s, length, step)   # date split: no test-period leakage
        if tr.shape[0] == 0:
            if verbose:
                print(f"{name:12s} window={length} no training windows before the test period "
                      f"({len(s)} obs from {s.dropna().index.min():%Y-%m-%d}) -> skipped")
            continue
        windowed[name] = {"train": tr, "test": te, "window_len": length, "step": step}
        if verbose:
            print(f"{name:12s} window={length:3d}  train={tr.shape[0]:,}  test={te.shape[0]:,}")
    return windowed


# ══════════════════════════════════════════════════════════════════════════════
#  5. Aligned real pairs
# ══════════════════════════════════════════════════════════════════════════════

def build_aligned_pairs(transformed: dict, base: str = "SP500", other: str = "Bitcoin",
                        seq_len: "int | None" = None, verbose: bool = True) -> dict:
    """Date-align two return series, window both on the shared calendar, and return
    the bundle nb_10_part_2 / nb_11 consume as `aligned_windows_real.pkl`.

    Window i of both series starts on the same calendar date, so "timepoint" is a
    single index into the common-date list. Keys are kept exactly as nb_00 wrote
    them ("sp500_real", "bitcoin_real_paired", ...) so downstream code is unchanged
    even when base/other differ.
    """
    from tslearning.datasets.data_utils import window_for
    seq_len = seq_len or window_for(base)[0]

    a = transformed[base].dropna().sort_index()
    b = transformed[other].dropna().sort_index()

    common = a.index.intersection(b.index).sort_values()
    a_common = a.loc[common].to_numpy(float)
    b_common = b.loc[common].to_numpy(float)

    def _slide(values):
        return np.array([values[i:i + seq_len]
                         for i in range(len(values) - seq_len + 1)])

    a_win = _slide(a_common)
    b_win = _slide(b_common)

    # `other` on its own full calendar too (Bitcoin trades weekends), offset so
    # that timepoint 0 is still the first common date.
    b_full_values = b.to_numpy(float)
    offset = int(b.index.searchsorted(common[0]))
    b_win_full = _slide(b_full_values)

    bundle = {
        "sp500_real":          {"windows": a_win, "timepoints": list(range(len(a_win)))},
        "bitcoin_real_paired": {"windows": b_win, "timepoints": list(range(len(b_win)))},
        "bitcoin_real_full":   {"windows": b_win_full,
                                "timepoints": list(range(-offset, len(b_win_full) - offset))},
        "timepoint_to_date":   {i: d.strftime("%Y-%m-%d") for i, d in enumerate(common)},
        "pair":                (base, other),
    }
    if verbose:
        print(f"aligned {base} x {other}: {len(common)} common dates "
              f"[{common[0].date()} -> {common[-1].date()}]  "
              f"windows {a_win.shape} / {b_win.shape}")
    return bundle


# ══════════════════════════════════════════════════════════════════════════════
#  Stage runner
# ══════════════════════════════════════════════════════════════════════════════

def run(series: "list[str] | None" = None, force: bool = False, extend: bool = False,
        aligned_pair: "tuple[str, str]" = ("SP500", "Bitcoin"),
        skip_fingerprints: bool = False, verbose: bool = True) -> dict:
    """Run the whole financial ingestion stage and write the pkl artifacts.

    Returns a dict of what was produced, for the caller to inspect or assert on.
    """
    out_dir = PATHS["data_processed"]
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = download_prices(series, force=force, verbose=verbose, extend=extend)
    transformed, ohlcv = transform_prices(raw, verbose=verbose)
    windowed = build_windows(transformed, verbose=verbose)
    fingerprints = {} if skip_fingerprints else compute_fingerprints(transformed, verbose=verbose)

    written = []

    def _dump(obj, filename):
        with open(out_dir / filename, "wb") as f:
            pickle.dump(obj, f)
        written.append(filename)

    # MERGE, never replace: a run over some series must not delete the others
    from tslearning.datasets.data_utils import merge_into_pickle
    run_series = set(transformed)
    transformed = merge_into_pickle(out_dir / "transformed.pkl", transformed)
    windowed = merge_into_pickle(out_dir / "windowed.pkl", windowed)
    written += ["transformed.pkl", "windowed.pkl"]
    if not skip_fingerprints:
        fingerprints = merge_into_pickle(out_dir / "fingerprints.pkl", fingerprints)
        written.append("fingerprints.pkl")
    if ohlcv is not None:
        ohlcv.to_pickle(out_dir / "sp500_ohlcv_lr.pkl")
        written.append("sp500_ohlcv_lr.pkl")

    # rebuilt from the MERGED series whenever this run touched either side, so a run
    # refreshing SP500 alone can't leave a stale SP500 x Bitcoin pair behind
    aligned = None
    base, other = aligned_pair
    stale = not (out_dir / "aligned_windows_real.pkl").exists() or \
        bool({base, other} & run_series)
    if base in transformed and other in transformed and stale:
        aligned = build_aligned_pairs(transformed, base, other, verbose=verbose)
        _dump(aligned, "aligned_windows_real.pkl")
    elif verbose:
        why = ("unchanged by this run" if base in transformed and other in transformed
               else "not both available")
        print(f"aligned pair ({base}, {other}) {why} -> "
              "aligned_windows_real.pkl not rewritten")

    if verbose:
        print("\nwrote -> " + ", ".join(written))
    return {"transformed": transformed, "windowed": windowed,
            "fingerprints": fingerprints, "aligned": aligned, "written": written}
