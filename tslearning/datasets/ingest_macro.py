"""
Macro data ingestion — the library half of `scripts/data_prep.py -d macro`.

Sources:
  * FRED (needs FRED_API_KEY in the environment) — US GDP, CPI, Fed funds,
    unemployment, consumer credit, M2.
  * Eurostat (no key) — DE/FR/IT unemployment, HICP inflation, GDP.

Outputs (data/processed/): transformed_macro.pkl, windowed_macro.pkl,
fingerprints_macro.pkl. Raw downloads are cached in data/raw/.
"""
from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path

import pandas as pd

from tslearning.config import CFG, PATHS

# ── Catalogs and 13 ──────────────────────────────
FRED_CATALOG = {
    "US_GDP":          {"id": "GDPC1",    "transform": "logdiff", "freq": "Q"},
    "US_Inflation":    {"id": "CPIAUCSL", "transform": "logdiff", "freq": "M"},
    "US_Rate":         {"id": "FEDFUNDS", "transform": "diff",    "freq": "M"},
    "US_Unemployment": {"id": "UNRATE",   "transform": "diff",    "freq": "M"},
    "US_Credit":       {"id": "TOTALSL",  "transform": "logdiff", "freq": "M"},
    "US_Money":        {"id": "M2SL",     "transform": "logdiff", "freq": "M"},
}

EU_COUNTRIES = ["DE", "FR", "IT"]

EUROSTAT_CATALOG = {
    "Unemployment": {
        "code": "une_rt_m", "transform": "diff",
        "filters": {"s_adj": "SA", "age": "TOTAL", "sex": "T", "unit": "PC_ACT"},
    },
    "Inflation": {
        "code": "prc_hicp_midx", "transform": "logdiff",
        "filters": {"unit": "I15", "coicop": "CP00"},
    },
    "GDP": {
        "code": "namq_10_gdp", "transform": "logdiff",
        "filters": {"s_adj": "SCA", "unit": "CLV10_MEUR", "na_item": "B1GQ"},
    },
}


def _use_certifi() -> None:
    """nb_00b cell 1: Eurostat/FRED TLS fails on some macOS Pythons without this."""
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  FRED
# ══════════════════════════════════════════════════════════════════════════════

def download_fred(series: "list[str] | None" = None, force: bool = False,
                  raw_dir: "Path | None" = None, verbose: bool = True) -> dict:
    """Return {name: raw pd.Series}, cached at data/raw/{name}_raw.csv."""
    raw_dir = Path(raw_dir or PATHS["data_raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    names = series or list(FRED_CATALOG)

    client = None
    raw = {}
    for name in names:
        cfg = FRED_CATALOG[name]
        cache = raw_dir / f"{name}_raw.csv"
        if cache.exists() and not force:
            s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
            if verbose:
                print(f"[CACHE]      {name:16s} ({cfg['id']:9s}): {len(s):>5,} obs")
        else:
            if client is None:
                _use_certifi()
                from fredapi import Fred
                key = os.environ.get("FRED_API_KEY")
                if not key:
                    raise RuntimeError(
                        "FRED_API_KEY not set. export FRED_API_KEY='...' and retry, "
                        "or run with cached data/raw/*_raw.csv present.")
                client = Fred(api_key=key)
            s = client.get_series(cfg["id"])
            s.to_csv(cache)
            if verbose:
                print(f"[DOWNLOADED] {name:16s} ({cfg['id']:9s}): {len(s):>5,} obs")
        raw[name] = s
    return raw


# ══════════════════════════════════════════════════════════════════════════════
#  Eurostat
# ══════════════════════════════════════════════════════════════════════════════

def download_eurostat(indicators: "list[str] | None" = None, force: bool = False,
                      raw_dir: "Path | None" = None, verbose: bool = True) -> dict:
    """Return {indicator: wide DataFrame}, cached at data/raw/EU_{indicator}_raw.csv."""
    raw_dir = Path(raw_dir or PATHS["data_raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    names = indicators or list(EUROSTAT_CATALOG)

    raw = {}
    for ind in names:
        cache = raw_dir / f"EU_{ind}_raw.csv"
        if cache.exists() and not force:
            df = pd.read_csv(cache)
            if verbose:
                print(f"[CACHE]      EU {ind:13s}: shape {df.shape}")
        else:
            _use_certifi()
            import eurostat
            df = eurostat.get_data_df(EUROSTAT_CATALOG[ind]["code"])
            df.to_csv(cache, index=False)
            if verbose:
                print(f"[DOWNLOADED] EU {ind:13s}: shape {df.shape}")
        raw[ind] = df
    return raw


def parse_eurostat(df: pd.DataFrame, geo: str, filters: dict) -> pd.Series:
    """Extract one date-indexed Series for one country from a wide Eurostat frame.

    Verbatim from nb_00b cell 19 — the dimension column names vary by dataset, and
    the geo column is sometimes 'geo', sometimes 'geo\\TIME_PERIOD'.
    """
    df = df.copy()

    geo_col = next((c for c in df.columns if c.split("\\")[0] == "geo"), None)
    if geo_col is None:
        raise KeyError(f"No 'geo' column found. Columns: {list(df.columns)[:10]}")

    mask = df[geo_col] == geo
    for dim, val in filters.items():
        if dim not in df.columns:
            raise KeyError(
                f"Filter dim '{dim}' not in columns. "
                f"Available dims: {[c for c in df.columns if df[c].dtype == object]}")
        mask &= (df[dim] == val)

    sub = df[mask]
    if len(sub) == 0:
        raise ValueError(f"No row matches geo={geo} with filters {filters}")
    if len(sub) > 1:
        warnings.warn(f"{len(sub)} rows match geo={geo}; using the first.")
    row = sub.iloc[0]

    time_cols = [c for c in df.columns if str(c)[:4].isdigit()]
    s = pd.to_numeric(row[time_cols], errors="coerce").dropna()
    s.index = pd.to_datetime(
        [str(c).replace("Q1", "-01").replace("Q2", "-04")
              .replace("Q3", "-07").replace("Q4", "-10") for c in s.index],
        errors="coerce")
    s = s[s.index.notna()].sort_index()
    s.name = geo
    return s.astype(float)


# ══════════════════════════════════════════════════════════════════════════════
#  Transform / window / fingerprint
# ══════════════════════════════════════════════════════════════════════════════

def transform_macro(fred_raw: dict, eurostat_raw: dict,
                    countries: "list[str] | None" = None,
                    verbose: bool = True) -> dict:
    """Apply each series' configured transform; return {name: stationary Series}."""
    from tslearning.datasets.data_utils import macro_transform

    countries = countries or EU_COUNTRIES
    out = {}

    for name, cfg in FRED_CATALOG.items():
        if name not in fred_raw:
            continue
        t = macro_transform(fred_raw[name], cfg["transform"])
        t.index = pd.to_datetime(t.index)
        out[name] = t
        if verbose:
            print(f"{name:16s} {cfg['transform']:8s} n={len(t):>5,}  "
                  f"[{t.index[0].date()} -> {t.index[-1].date()}]")

    for ind, cfg in EUROSTAT_CATALOG.items():
        if ind not in eurostat_raw:
            continue
        for geo in countries:
            key = f"{geo}_{ind}"
            try:
                t = macro_transform(parse_eurostat(eurostat_raw[ind], geo, cfg["filters"]),
                                    cfg["transform"])
                t.index = pd.to_datetime(t.index)
                out[key] = t
                if verbose:
                    print(f"{key:16s} {cfg['transform']:8s} n={len(t):>5,}  "
                          f"[{t.index[0].date()} -> {t.index[-1].date()}]")
            except Exception as e:  # a missing dimension value shouldn't kill the run
                if verbose:
                    print(f"{key:16s} SKIPPED: {type(e).__name__}: {e}")
    return out


def compute_fingerprints(transformed: dict, verbose: bool = True) -> dict:
    """Fingerprints with macro=True (lag counts capped for short samples)."""
    from tslearning.datasets.data_utils import compute_fingerprint

    out = {}
    for name, s in transformed.items():
        fp = compute_fingerprint(s, name, macro=True)
        out[name] = fp
        if verbose:
            print(f"{name:16s} n={fp['n_obs']:>5,}  mu={fp['mean']:+.4f}  "
                  f"sigma={fp['std']:.4f}  kurt={fp['excess_kurtosis']:+.2f}")
    return out


def build_windows(transformed: dict, verbose: bool = True) -> dict:
    """Window each macro series with its configured (length, step). Series with no
    window config, or too short for one window, are skipped."""
    from tslearning.datasets.data_utils import synth_windows, window_for

    windowed = {}
    for name, s in transformed.items():
        try:
            length, step = window_for(name)
        except KeyError:
            if verbose:
                print(f"{name:16s} no window config -> skipped")
            continue
        tr, te = synth_windows(s, length, step)   # date split: no test-period leakage
        if tr.shape[0] == 0:
            if verbose:
                print(f"{name:16s} window={length:>3d} no training windows before the test period "
                      f"({len(s)} obs from {s.dropna().index.min():%Y-%m-%d}) -> skipped")
            continue
        windowed[name] = {"train": tr, "test": te, "window_len": length, "step": step}
        if verbose:
            print(f"{name:16s} window={length:>3d}  train={tr.shape[0]:>4,}  test={te.shape[0]:>4,}")
    return windowed


# ══════════════════════════════════════════════════════════════════════════════
#  Stage runner
# ══════════════════════════════════════════════════════════════════════════════

def run(series: "list[str] | None" = None, force: bool = False,
        countries: "list[str] | None" = None, skip_eu: bool = False,
        skip_fingerprints: bool = False, verbose: bool = True) -> dict:
    """Run the whole macro ingestion stage and write the pkl artifacts.

    `series` filters the FRED names only (the EU set is driven by `countries`).
    """
    out_dir = PATHS["data_processed"]
    out_dir.mkdir(parents=True, exist_ok=True)

    fred_raw = download_fred(series, force=force, verbose=verbose)
    eurostat_raw = {} if skip_eu else download_eurostat(force=force, verbose=verbose)

    transformed = transform_macro(fred_raw, eurostat_raw, countries, verbose=verbose)
    windowed = build_windows(transformed, verbose=verbose)
    fingerprints = {} if skip_fingerprints else compute_fingerprints(transformed, verbose=verbose)

    written = []

    def _dump(obj, filename):
        with open(out_dir / filename, "wb") as f:
            pickle.dump(obj, f)
        written.append(filename)

    # MERGE, never replace: a run over some series/countries must not delete the others
    from tslearning.datasets.data_utils import merge_into_pickle
    transformed = merge_into_pickle(out_dir / "transformed_macro.pkl", transformed)
    windowed = merge_into_pickle(out_dir / "windowed_macro.pkl", windowed)
    written += ["transformed_macro.pkl", "windowed_macro.pkl"]
    if not skip_fingerprints:
        fingerprints = merge_into_pickle(out_dir / "fingerprints_macro.pkl", fingerprints)
        written.append("fingerprints_macro.pkl")

    if verbose:
        print("\nwrote -> " + ", ".join(written))
    return {"transformed": transformed, "windowed": windowed,
            "fingerprints": fingerprints, "written": written}
