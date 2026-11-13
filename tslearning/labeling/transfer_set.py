"""
Transfer-set construction — the library half of `scripts/build_transfer_set.py`
and of the batch-building step of `scripts/prep_real_pairs.py`.

Vocabulary:
  * "transfer set" (Bucila et al. 2006) = the labelled pair set the student
    distills from. Here: window pairs the reasoner will annotate.
  * "disjoint-block layout" = each pair is written into its own time block, so the
    reasoner cannot form temporal joins ACROSS pairs and therefore labels each
    pair independently. Every batch CSV in this project uses it.

Two generators:
  plain      independent DDPM samples paired up. Honest as a transfer set, but
             nearly every pair comes out "no correlation",
             which is what collapsed the early classifier versions.
  engineered controlled pos / neg / none correlation shapes with volatility, trend
             and co-crash events injected on top, so highVol / trendDown /
             tailDependent actually fire. Records the injected ground truth in
             engineered_manifest.csv for a known-answer check.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from tslearning.config import CFG, PATHS

# ── The transfer-set short name ────────────────────────────────────────────────
# Windows inside the batch CSVs are named "{short}_synth_{i}", and the window arrays
# are saved as "engineered_{short}.npy". nb_10 established "btc" for Bitcoin, so the
# files on disk and every labelled window carry that, while the pair spec and the
# series registry say "bitcoin". Resolve in ONE place; two independent conventions for
# one series is how `engineered_bitcoin.npy` came to be looked up for a file that has
# always been called `engineered_btc.npy`.
SHORT_NAME = {"bitcoin": "btc"}



def _tail_k() -> int:
    """Joint-crash days injected into a "tail" pair: tail_min or tail_min + 1, so the
    reasoner's tailDependent (>= tail_min) fires on it. 2-3 days at the paper's W=50."""
    from tslearning.labeling.exact_flags import label_constants
    return label_constants()["tail_min"]

def engineered_name(series: str) -> str:
    """Series name -> the short name used in window ids and engineered_*.npy."""
    s = str(series).lower()
    return SHORT_NAME.get(s, s)


def engineered_path(series: str, data_dir=None) -> Path:
    """Locate engineered_<series>.npy, accepting either naming convention.

    Raises with the candidates tried and what is actually present, because the failure
    mode otherwise is a bare FileNotFoundError naming a file that was never written.
    """
    data_dir = Path(data_dir or PATHS["data_processed"])
    tried = []
    for name in dict.fromkeys([engineered_name(series), str(series).lower()]):
        p = data_dir / f"engineered_{name}.npy"
        tried.append(p.name)
        if p.exists():
            return p
    have = sorted(q.name for q in data_dir.glob("engineered_*.npy"))
    raise FileNotFoundError(
        f"no engineered window array for '{series}'. Tried {tried}; "
        f"present: {have or '(none)'}. Run `build_transfer_set.py -p <pair>` to "
        f"create them, or add a SHORT_NAME entry in labeling/transfer_set.py.")

BATCH_COLUMNS = ["window_name", "observation_id", "value", "start_time", "end_time"]


# ══════════════════════════════════════════════════════════════════════════════
#  Batch CSV writer (the reasoner's input format)
# ══════════════════════════════════════════════════════════════════════════════

def build_pairs_csv(arr_a, arr_b, csv_path, start_global: int = 0,
                    name_a: str = "sp500_synth", name_b: str = "btc_synth") -> pd.DataFrame:
    """Write one reasoner-input CSV in disjoint-block layout.

    arr_a, arr_b : (N, L) arrays, same shape. Pair i is (name_a_i, name_b_i) and
    occupies time block [i*L, (i+1)*L), so the reasoner treats pairs independently.
    Identical output to nb_10 cell 18.
    """
    arr_a, arr_b = np.asarray(arr_a), np.asarray(arr_b)
    assert arr_a.shape == arr_b.shape, f"shape mismatch: {arr_a.shape} vs {arr_b.shape}"
    L = arr_a.shape[1]

    rows, obs_id = [], 0
    for series_name, arr in [(name_a, arr_a), (name_b, arr_b)]:
        for w_idx, window in enumerate(arr):
            window_name = f"{series_name}_{start_global + w_idx}"
            base = w_idx * L
            for t, value in enumerate(window):
                rows.append({"window_name": window_name, "observation_id": obs_id,
                             "value": float(value), "start_time": base + t,
                             "end_time": base + t + 1})
                obs_id += 1

    df = pd.DataFrame(rows, columns=BATCH_COLUMNS)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def write_batches(arr_a, arr_b, out_dir, batch_size: int = 25,
                  prefix: str = "batch", name_a: str = "sp500_synth",
                  name_b: str = "btc_synth", verbose: bool = True) -> int:
    """Split a pair set into batch CSVs named {prefix}_{NNN}.csv. Returns the count."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_batches = len(arr_a) // batch_size
    for b in range(n_batches):
        s = b * batch_size
        build_pairs_csv(arr_a[s:s + batch_size], arr_b[s:s + batch_size],
                        out_dir / f"{prefix}_{b:03d}.csv", start_global=s,
                        name_a=name_a, name_b=name_b)
    if verbose:
        print(f"wrote {n_batches} batch CSVs ({batch_size} pairs each) -> {out_dir}")
    return n_batches


# ══════════════════════════════════════════════════════════════════════════════
#  Fitted thresholds (tau, qLow)
# ══════════════════════════════════════════════════════════════════════════════

def fit_thresholds(series: "list[str] | None" = None, out_dir=None,
                   verbose: bool = True) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Fit and save the per-series thresholds the reasoner and exact_flags share.

    tau  = `vol_percentile`th percentile of |return|  (default 90) -> big-move day
    qLow = `crash_percentile`th percentile of return  (default 5)  -> crash day

    Fitted on the TRAIN period only (config split.train) — these are fitted values,
    not config. Written as tau.csv / qlow.csv with the
    wide [start, end) validity columns the .vada program expects.
    Same computation as nb_10 cells 25-26 and nb_prep_real_pairs cell 2.

    Works for financial and macro series alike: `load_series` reads each from
    whichever pkl holds it.
    """
    from tslearning.datasets.pairs import load_series

    series = series or ["SP500", "Bitcoin"]
    out_dir = Path(out_dir or (PATHS["data_processed"] / "thresholds"))
    out_dir.mkdir(parents=True, exist_ok=True)

    vol_pct = CFG["predicates"]["vol_percentile"]
    crash_pct = CFG["predicates"]["crash_percentile"]
    lo, hi = CFG["split"]["train"]

    RANGE_START, RANGE_END = 0, 1_000_000
    transformed = load_series(series)

    tau_rows, qlow_rows = [], []
    for name in series:
        s = transformed[name].dropna().sort_index().loc[lo:hi]
        r = s.values.astype(float)
        tau = float(np.percentile(np.abs(r), vol_pct))
        qlow = float(np.percentile(r, crash_pct))
        sid = name.lower()
        tau_rows.append({"series": sid, "tau": tau, "start": RANGE_START, "end": RANGE_END})
        qlow_rows.append({"series": sid, "qlow": qlow, "start": RANGE_START, "end": RANGE_END})
        if verbose:
            print(f"{name:10s} train {s.index.min().date()}..{s.index.max().date()}  "
                  f"n={len(r):>5}  tau={tau:.6f}  qLow={qlow:.6f}")

    tau_df = pd.DataFrame(tau_rows, columns=["series", "tau", "start", "end"])
    qlow_df = pd.DataFrame(qlow_rows, columns=["series", "qlow", "start", "end"])

    # MERGE, never truncate. The threshold files are shared by every pair and every
    # stage (the .vada reads them wholesale), so fitting two macro series must not
    # delete the financial rows an in-flight run depends on. Freshly fitted values
    # win; untouched series are preserved.
    tau_df = _merge_thresholds(out_dir / "tau.csv", tau_df, "tau")
    qlow_df = _merge_thresholds(out_dir / "qlow.csv", qlow_df, "qlow")

    tau_df.to_csv(out_dir / "tau.csv", index=False)
    qlow_df.to_csv(out_dir / "qlow.csv", index=False)
    if verbose:
        print(f"wrote -> {out_dir/'tau.csv'}, {out_dir/'qlow.csv'} "
              f"({len(tau_df)} series on file)")
    return tau_df, qlow_df


def _merge_thresholds(path, fresh: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Combine newly fitted threshold rows with whatever is already on disk."""
    path = Path(path)
    if not path.exists():
        return fresh
    old = pd.read_csv(path)
    keep = old[~old["series"].isin(set(fresh["series"]))]
    return (pd.concat([keep, fresh], ignore_index=True)
            .sort_values("series")
            .reset_index(drop=True)[["series", value_col, "start", "end"]])


def load_thresholds(thr_dir=None) -> "tuple[dict, dict]":
    """Read tau.csv / qlow.csv back as {series: value} dicts."""
    thr_dir = Path(thr_dir or (PATHS["data_processed"] / "thresholds"))
    tau = pd.read_csv(thr_dir / "tau.csv").set_index("series")["tau"].to_dict()
    qlow = pd.read_csv(thr_dir / "qlow.csv").set_index("series")["qlow"].to_dict()
    return tau, qlow


# ══════════════════════════════════════════════════════════════════════════════
#  Generator: plain (independent DDPM pairs)
# ══════════════════════════════════════════════════════════════════════════════

def generate_plain(arr_a: np.ndarray, arr_b: np.ndarray, n_pairs: "int | None" = None,
                   seed: "int | None" = None) -> "tuple[np.ndarray, np.ndarray, pd.DataFrame]":
    """Pair up two independent synthetic sets, truncated to the same length.

    This is nb_10-clean's transfer set. Expect an overwhelming "no correlation"
    majority — that is the honest data-free-KD framing, and also why `engineered`
    exists.
    """
    rng = np.random.default_rng(seed if seed is not None else CFG["seed"])
    n = min(len(arr_a), len(arr_b))
    if n_pairs:
        n = min(n, n_pairs)
    idx = rng.permutation(n)
    X = np.asarray(arr_a[:n], np.float32)[idx]
    Y = np.asarray(arr_b[:n], np.float32)[idx]
    manifest = pd.DataFrame({"pair_id": range(n), "regime": "unknown",
                             "source": "plain"})
    return X, Y, manifest


# ══════════════════════════════════════════════════════════════════════════════
#  Generator: engineered
# ══════════════════════════════════════════════════════════════════════════════

def _side(shape, std, drift):
    """Rescale a shape to a target std and add a per-day drift."""
    s = (shape - shape.mean()) / (shape.std() + 1e-8)
    return s * std + drift


def generate_engineered(pool: np.ndarray, n_pairs: int, tau: dict, qlow: dict,
                        series_a: str = "sp500", series_b: str = "bitcoin",
                        high_vol_frac: float = 0.4, tail_frac: float = 0.25,
                        seq_len: int = 50, seed: "int | None" = None,
                        neg_noise: float = 0.5
                        ) -> "tuple[np.ndarray, np.ndarray, pd.DataFrame]":
    """Build balanced pos / neg / none pairs with injected vol, trend and co-crashes.

    Reuses real DDPM windows as the shape pool, then:
      * correlation regime: y = +shape (pos) | -shape (neg) | unrelated shape (none)
      * volatility:  target std = 1.2*tau (high) or 0.4*tau (low), per series
      * trend:       +/-5% or flat, spread over the window
      * tail:        2-3 days forced below 1.5*qLow in BOTH series

    neg_noise (0 to 1): controls how noisy the negative-correlation pairs are.
    At 0, they are perfect mirrors (y = -x, the original behaviour). At 1 the
    anticorrelation is a noisy blend of -shape and an independent shape:
        y = (1 - alpha) * (-shape) + alpha * independent_shape
    where alpha ~ U(0, neg_noise). This produces negatives that are still
    anticorrelated (the concordance measure still classifies them as negative)
    but not perfectly inverted — closer to how real flight-to-safety looks.

    The returned manifest is the INJECTED ground truth — a known-answer check that
    the reasoner (and later the classifier) recovers what was put in.
    """
    rng = np.random.default_rng(seed if seed is not None else CFG["seed"])
    pool = np.asarray(pool, np.float32)
    tx, ty = tau[series_a], tau[series_b]
    qx, qy = qlow[series_a], qlow[series_b]

    def pick():
        return pool[rng.integers(len(pool))]

    X, Y, man = [], [], []
    for _ in range(n_pairs):
        hv_x = rng.random() < high_vol_frac
        hv_y = rng.random() < high_vol_frac
        std_x = tx * 1.2 if hv_x else tx * 0.4
        std_y = ty * 1.2 if hv_y else ty * 0.4
        dx = rng.choice([0.05, -0.05, 0.0])
        dy = rng.choice([0.05, -0.05, 0.0])

        shp = pick()
        x = _side(shp, std_x, dx / seq_len)
        regime = rng.choice(["pos", "neg", "none"])
        if regime == "pos":
            y = _side(shp, std_y, dy / seq_len)
        elif regime == "neg":
            # Noisy anticorrelation: blend -shape with an independent shape
            # so the classifier sees varied negatives, not just perfect mirrors.
            alpha = rng.uniform(0, neg_noise)
            blended = (1 - alpha) * (-shp) + alpha * pick()
            y = _side(blended, std_y, dy / seq_len)
        else:
            y = _side(pick(), std_y, dy / seq_len)

        tail = rng.random() < tail_frac
        if tail:
            days = rng.choice(seq_len, size=int(rng.integers(_tail_k(), _tail_k() + 2)), replace=False)
            x[days], y[days] = qx * 1.5, qy * 1.5

        X.append(x.astype(np.float32))
        Y.append(y.astype(np.float32))
        man.append({"regime": regime,
                    "vol_x": "high" if hv_x else "low",
                    "vol_y": "high" if hv_y else "low",
                    "trend_x": "up" if dx > 0 else "down" if dx < 0 else "flat",
                    "trend_y": "up" if dy > 0 else "down" if dy < 0 else "flat",
                    "tail": bool(tail)})

    order = rng.permutation(n_pairs)
    manifest = pd.DataFrame(man).iloc[order].reset_index(drop=True)
    manifest.insert(0, "pair_id", range(n_pairs))  # pair_id i <-> {series}_synth_i
    return np.array(X)[order], np.array(Y)[order], manifest


# ══════════════════════════════════════════════════════════════════════════════
#  Real aligned pairs -> batches
# ══════════════════════════════════════════════════════════════════════════════

def build_real_batches(pair: str, series_returns: dict, other: str, lo: str, hi: str,
                       stride: int, batch_size: int, out_dir, seq_len: int = 50,
                       base: str = "sp500", verbose: bool = True) -> dict:
    """Align two real return series, slide subsampled windows, write batch CSVs.

    Window starts are the REAL day index into the common-date list, so the
    predicate intervals the reasoner emits are already on the real timeline (and
    {pair}_dates.csv maps day -> calendar date for reporting break dates later).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = None
    for s in series_returns.values():
        i = s.dropna().sort_index().loc[lo:hi].index
        idx = i if idx is None else idx.intersection(i)
    idx = idx.sort_values()

    a = series_returns[base].loc[idx].values.astype(np.float32)
    b = series_returns[other].loc[idx].values.astype(np.float32)
    N = len(idx)
    tps = list(range(0, N - seq_len + 1, stride))

    n_batches = 0
    for start in range(0, len(tps), batch_size):
        chunk = tps[start:start + batch_size]
        rows, obs = [], 0
        for p, t in enumerate(chunk):
            block = p * seq_len
            for nm, arr in [(f"{base}_real_{t}", a[t:t + seq_len]),
                            (f"{other}_real_{t}", b[t:t + seq_len])]:
                for k, v in enumerate(arr):
                    rows.append({"window_name": nm, "observation_id": obs,
                                 "value": float(v), "start_time": block + k,
                                 "end_time": block + k + 1})
                    obs += 1
        bpath = out_dir / f"{pair}_batch_{n_batches:03d}.csv"
        pd.DataFrame(rows, columns=BATCH_COLUMNS).to_csv(bpath, index=False)
        # per-window tau/qLow for the reasoner (thresholds.py): this writer knows each
        # window's first calendar day, so `labels.thresholds: rolling` works here
        from tslearning.labeling import thresholds as thr
        thr.write_for_batch(bpath, overwrite=True, window_dates={
            nm: idx[t] for t in chunk
            for nm in (f"{base}_real_{t}", f"{other}_real_{t}")})
        n_batches += 1

    pd.DataFrame({"day": range(N), "date": idx.astype(str)}).to_csv(
        out_dir / f"{pair}_dates.csv", index=False)

    if verbose:
        print(f"{pair}: {N} aligned days -> {len(tps)} windows (stride {stride}) "
              f"-> {n_batches} batches")
    return {"pair": pair, "aligned_days": N, "windows": len(tps),
            "batches": n_batches, "out_dir": str(out_dir)}


# ══════════════════════════════════════════════════════════════════════════════
#  Stage runner
# ══════════════════════════════════════════════════════════════════════════════

def run(pair: str = "sp500_bitcoin", mode: str = "engineered",
        n_pairs: "int | None" = None, batch_size: int = 25,
        fit_thr: bool = True, out_dir=None, verbose: bool = True) -> dict:
    """Build the synthetic transfer set for one pair and write its batch CSVs.

    `pair` is "<a>_<b>" with lowercase series names; the synthetic .npy files are
    looked up per side via the synth registry, so the same code serves
    sp500_bitcoin, sp500_nasdaq, ...
    """
    from tslearning.datasets.pairs import resolve_pair
    from tslearning.synth import registry

    data_dir = PATHS["data_processed"]
    series_a, series_b, pair = resolve_pair(pair)
    # one folder per pair: the flat data/processed/ default let pairs overwrite each
    # other's batch_*.csv / manifest, and overwrote v7's training windows
    out_dir = Path(out_dir or (data_dir / f"transfer_{pair}"))
    a_name, b_name = engineered_name(series_a), engineered_name(series_b)

    if fit_thr:
        fit_thresholds([series_a, series_b], verbose=verbose)
    tau, qlow = load_thresholds()

    path_a = registry.sample_path(series_a)
    path_b = registry.sample_path(series_b)
    for p in (path_a, path_b):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — run `synthesize.py -d {p.stem}` first")
    arr_a, arr_b = np.load(path_a), np.load(path_b)
    seq_len = arr_a.shape[1]

    # .npy files go alongside the batch CSVs so that multi-pair training can
    # find each pair's windows in its own directory.
    npy_dir = out_dir
    npy_dir.mkdir(parents=True, exist_ok=True)

    if mode == "engineered":
        pool = np.concatenate([arr_a, arr_b], axis=0)
        n = n_pairs or 525          # nb_10-alt default: ~21 batches of 25
        X, Y, manifest = generate_engineered(pool, n, tau, qlow,
                                             series_a=series_a.lower(),
                                             series_b=series_b.lower(),
                                             seq_len=seq_len)
        np.save(npy_dir / f"engineered_{a_name}.npy", X)
        np.save(npy_dir / f"engineered_{b_name}.npy", Y)
        manifest.to_csv(npy_dir / "engineered_manifest.csv", index=False)
        if verbose:
            print(f"engineered pairs: {len(X)} | {manifest['regime'].value_counts().to_dict()} "
                  f"| tail {int(manifest['tail'].sum())} "
                  f"| highVol_x {int((manifest['vol_x']=='high').sum())}")
    elif mode == "plain":
        X, Y, manifest = generate_plain(arr_a, arr_b, n_pairs)
        if verbose:
            print(f"plain pairs: {len(X)} (independent DDPM samples)")
    else:
        raise ValueError(f"unknown mode {mode!r} (expected 'engineered' or 'plain')")

    n_batches = write_batches(X, Y, out_dir, batch_size=batch_size,
                              name_a=f"{a_name}_synth", name_b=f"{b_name}_synth",
                              verbose=verbose)

    return {"pair": pair, "mode": mode, "n_pairs": len(X), "n_batches": n_batches,
            "batch_size": batch_size, "seq_len": seq_len,
            "manifest": manifest, "out_dir": str(out_dir)}
