"""
Real-data inference — the library half of `scripts/infer.py`.

This is the stage the whole distillation exists for: annotating the FULL real
history, at every lag, which is intractable with the reasoner and instant with the
student.

Three things come out of it:
  1. predictions_lag{k}.pkl — the regime bundle, now including the softmax
     PROBABILITIES. The notebook saved only the argmax, which is why HMM smoothing had nothing to decode; keeping probs here is what unblocks it.
  2. temporal/{regime}.csv — regime facts in the reasoner's own schema.
  3. temporal/{flag}.csv — the exact market-state flags, computed in Python, NOT
     predicted. In a mined rule the condition (market state) is therefore exact
     ground truth and only the conclusion (regime) carries classifier noise.

A note on lags: lag k pairs series A at day t with series B at day t+k, which is the
lead-lag / price-discovery question ("who moves first"). The reasoner needed ~3x the
runtime for the lag ruleset, so it was dropped there; here it costs one extra forward
pass per lag.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from tslearning.config import CFG, PATHS


def load_aligned(path=None) -> dict:
    """Read aligned_windows_real.pkl (written by datasets/ingest_financial.py)."""
    path = Path(path or (PATHS["data_processed"] / "aligned_windows_real.pkl"))
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run data_prep.py -d financial first")
    with open(path, "rb") as f:
        return pickle.load(f)


def _lagged(windows_a, windows_b, timepoints, lag: int):
    """Pair window i of A with window i+lag of B. lag=0 is the aligned case."""
    n = min(len(windows_a), len(windows_b) - lag, len(timepoints))
    if n <= 0:
        raise ValueError(f"lag {lag} leaves no pairs")
    return (np.asarray(windows_a[:n]),
            np.asarray(windows_b[lag:lag + n]),
            list(timepoints[:n]))


def predict_timeline(model, windows_a, windows_b, timepoints, lag: int = 0,
                     verbose: bool = True) -> dict:
    """Run the student over the real timeline at one lag.

    Returns {"regime", "probs", "timepoints", "soft_cols", "lag"}.
    """
    from tslearning.distill.classifier import (build_timeline_input, predict_regime,
                                               regime_labels)

    a, b, tps = _lagged(windows_a, windows_b, timepoints, lag)
    X = build_timeline_input(a, b, standardize=True)
    regime, probs = predict_regime(model, X, return_probs=True)
    soft_cols = regime_labels()

    if verbose:
        dist = {soft_cols[c]: int((regime == c).sum()) for c in range(len(soft_cols))}
        print(f"lag {lag}: {len(regime)} timepoints | regime dist {dist}")
        print("  mean prob per class:",
              {c: round(float(v), 3) for c, v in zip(soft_cols, probs.mean(0))})
    return {"regime": regime, "probs": probs, "timepoints": tps,
            "soft_cols": soft_cols, "lag": lag}


def emit_flags(aligned, pair: str, out_dir, seq_len: int, verbose: bool = True) -> dict:
    """Write the exact market-state flag CSVs for the pair's real windows.

    Only the SP500 x Bitcoin bundle is covered here, because that is the pair
    aligned_windows_real.pkl holds and exact_flags.emit_flag_csvs names its rows for.
    For any other pair the equivalent flags are produced by the real-pair path
    (`prep_real_pairs.py`, which routes through py_labeler).
    """
    from tslearning.labeling.exact_flags import emit_flag_csvs
    from tslearning.labeling.transfer_set import load_thresholds

    if pair != "sp500_bitcoin":
        if verbose:
            print(f"flags: skipped for {pair} — use prep_real_pairs.py for non-Bitcoin "
                  "pairs (it emits the same predicates on the real timeline)")
        return {}

    tau, qlow = load_thresholds()
    sp = aligned["sp500_real"]["windows"]
    bt = aligned["bitcoin_real_paired"]["windows"]
    tps = list(aligned["sp500_real"]["timepoints"])
    counts = emit_flag_csvs(sp, bt, tps, tau, qlow, out_dir=out_dir, seq_len=seq_len)
    if verbose:
        print("exact flags written:")
        for k, v in sorted(counts.items()):
            print(f"  {k:16s} {v:5d}")
    return counts


# ══════════════════════════════════════════════════════════════════════════════
#  Per-pair inference in the real-pairs layout  (the path the rules can consume)
# ══════════════════════════════════════════════════════════════════════════════

REGIME_PREDICATES = ("isPositiveCorrelation", "isNegativeCorrelation", "isNoCorrelation")

# The classifier PREDICTS only the correlation regime. Every other predicate in a pair's
# folder — volatility, trend, tail dependence, lead-lag, macro state — is an exact
# per-window computation with no network involved, so the classifier source takes those
# files verbatim from the Python labelling of the same pair. This is what the paper
# already claims ("the condition is exact; only the conclusion carries model noise"),
# and copying rather than recomputing guarantees the two sources differ ONLY in the
# regime predicates, which is the comparison you want.
# Default weights: configs/config.yaml `classifier.weights` (distill.classifier.default_weights).


def _py_pair_dir(pair: str) -> Path:
    return PATHS["data_processed"] / "real_pairs_regime_py" / pair


def load_pair_timeline(pair: str, py_dir=None):
    """The pair's aligned daily returns, on EXACTLY the day indexing the Python labelling
    used, read back from its `{pair}_dates.csv`.

    Re-deriving the alignment independently is how day indices silently drift by one or
    two and every downstream comparison becomes garbage. The dates file IS the alignment;
    use it.
    """
    from tslearning.datasets.pairs import describe, load_series

    d = describe(pair)
    py_dir = Path(py_dir or _py_pair_dir(d["slug"]))
    dates_csv = py_dir / f"{d['slug']}_dates.csv"
    if not dates_csv.exists():
        raise FileNotFoundError(
            f"{dates_csv} missing — label the pair first: "
            f"python scripts/prep_real_pairs.py -p {d['slug']} --engine python --stride 1")
    idx = pd.to_datetime(pd.read_csv(dates_csv)["date"])

    series = load_series([d["a"], d["b"]])
    a = series[d["a"]].reindex(idx).to_numpy(np.float32)
    b = series[d["b"]].reindex(idx).to_numpy(np.float32)
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError(f"{pair}: NaNs after reindexing on the labelled calendar — the "
                         "transformed series on disk no longer match the dates file. "
                         "Re-run data_prep.py and prep_real_pairs.py together.")
    return {"desc": d, "idx": idx, "a": a, "b": b, "py_dir": py_dir}


def run_pair(pair: str, weights=None, out_root=None, stride: int = 1,
             copy_exact: bool = True, verbose: bool = True) -> dict:
    """Annotate one pair's full real timeline with the student and write the result in
    the same per-pair layout as `real_pairs_regime_py/<pair>/`, so `generate_rules.py
    --source classifier` reads it with no special casing.

    Writes to `real_pairs_regime_classifier/<pair>/`:
      {regime}_{pair}.csv          the THREE predicted regime predicates
      predictions_lag0.pkl         softmax probabilities per window (for HMM smoothing)
      everything else              copied from the Python labelling (exact flags, dates)
    """
    import shutil

    from tslearning.distill.classifier import default_weights, load_classifier

    weights = Path(weights) if weights else default_weights()
    tl = load_pair_timeline(pair)
    d, idx, a, b = tl["desc"], tl["idx"], tl["a"], tl["b"]
    slug, seq_len = d["slug"], d["seq_len"]
    out_dir = Path(out_root or (PATHS["data_processed"] / "real_pairs_regime_classifier")) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    N = len(idx)
    tps = list(range(0, N - seq_len + 1, stride))
    wa = np.stack([a[t:t + seq_len] for t in tps])
    wb = np.stack([b[t:t + seq_len] for t in tps])

    model = load_classifier(weights)
    out = predict_timeline(model, wa, wb, tps, lag=0, verbose=verbose)
    regime, probs, labels = out["regime"], out["probs"], out["soft_cols"]

    rows = {r: [] for r in labels}
    for i, t in enumerate(tps):
        rows[labels[int(regime[i])]].append(
            {"X": f"{d['a_key']}_real_{t}", "Y": f"{d['b_key']}_real_{t}",
             "Start": t, "End": t + seq_len})
    counts = {}
    for pred, rr in rows.items():
        pd.DataFrame(rr, columns=["X", "Y", "Start", "End"]).to_csv(
            out_dir / f"{pred}_{slug}.csv", index=False)
        counts[pred] = len(rr)

    with open(out_dir / "predictions_lag0.pkl", "wb") as f:
        pickle.dump({"timepoints": tps, "regime": regime, "probs": probs,
                     "soft_cols": labels, "lag": 0, "pair": slug,
                     "weights": str(weights), "seq_len": seq_len}, f)

    copied = []
    if copy_exact:
        for src in sorted(tl["py_dir"].glob("*.csv")):
            if any(src.name.startswith(f"{r}_") for r in REGIME_PREDICATES):
                continue          # the one thing the classifier is FOR
            shutil.copy2(src, out_dir / src.name)
            copied.append(src.name)

    if verbose:
        print(f"{slug}: {N} days -> {len(tps)} windows (stride {stride}) | weights {weights.name}")
        print(f"  predicted regimes: {counts}")
        print(f"  copied {len(copied)} exact-predicate files from the Python labelling")
        print(f"  -> {out_dir}")
    return {"pair": slug, "out_dir": out_dir, "counts": counts, "n_windows": len(tps),
            "weights": str(weights), "copied": copied}


# ══════════════════════════════════════════════════════════════════════════════
#  Stage runner  (legacy single-pair bundle — kept for nb_10_part_2 compatibility)
# ══════════════════════════════════════════════════════════════════════════════

def run(pair: str = "sp500_bitcoin", weights=None, lags=(0,), out_dir=None,
        write_flags: bool = True, seq_len: "int | None" = None,
        verbose: bool = True) -> dict:
    """Predict the regime over the real timeline and emit the predicate CSVs."""
    from tslearning.datasets.data_utils import window_for
    from tslearning.distill.classifier import load_classifier
    from tslearning.segment.predicate_io import write_regime_csvs

    data_dir = PATHS["data_processed"]
    out_dir = Path(out_dir or (data_dir / "temporal"))
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_len = seq_len or window_for("SP500")[0]

    aligned = load_aligned()
    a = np.asarray(aligned["sp500_real"]["windows"], np.float32)
    b = np.asarray(aligned["bitcoin_real_paired"]["windows"], np.float32)
    timepoints = list(aligned["sp500_real"]["timepoints"])

    model = load_classifier(weights)

    bundles, written = {}, []
    for lag in lags:
        out = predict_timeline(model, a, b, timepoints, lag=lag, verbose=verbose)
        path = data_dir / f"predictions_lag{lag}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"timepoints": out["timepoints"], "regime": out["regime"],
                         "probs": out["probs"], "soft_cols": out["soft_cols"],
                         "lag": lag, "pair": pair}, f)
        written.append(path.name)
        bundles[lag] = out

        if lag == 0:
            if verbose:
                print("regime CSVs (reasoner schema):")
            write_regime_csvs(out["regime"], out["timepoints"], out_dir,
                              seq_len=seq_len, labels=out["soft_cols"], verbose=verbose)

    flags = emit_flags(aligned, pair, out_dir, seq_len, verbose=verbose) if write_flags else {}

    if verbose:
        print(f"wrote -> {', '.join(written)} and predicate CSVs in {out_dir}")
    return {"pair": pair, "bundles": bundles, "flags": flags,
            "out_dir": out_dir, "written": written}


def stale_copies(pair: str, out_root=None) -> "list[str]":
    """Files of the classifier folder that no longer match the Python labelling.

    infer.py copies every non-regime file (flags, lead-lag, macro state, dates) from
    real_pairs_regime_py/<pair>/ ONCE. Rerunning step 6 afterwards leaves the classifier
    folder with the old copies — rules mined from it then mix outdated flags, or a
    different calendar, with the predicted regimes. Returns the names that differ or are
    missing ([] = in sync, or no classifier folder at all).
    """
    import filecmp

    py_dir = _py_pair_dir(pair)
    clf_dir = Path(out_root or (PATHS["data_processed"] / "real_pairs_regime_classifier")) / pair
    if not clf_dir.exists() or not py_dir.exists():
        return []
    out = []
    for src in sorted(py_dir.glob("*.csv")):
        if any(src.name.startswith(f"{r}_") for r in REGIME_PREDICATES):
            continue
        dst = clf_dir / src.name
        if not dst.exists() or not filecmp.cmp(src, dst, shallow=False):
            out.append(src.name)
    return out


def warn_if_stale(pair: str, context: str = "") -> bool:
    """Print a WARNING if the pair's classifier folder is out of step with step 6."""
    names = stale_copies(pair)
    if names:
        shown = ", ".join(names[:4]) + (f" (+{len(names) - 4} more)" if len(names) > 4 else "")
        print(f"WARNING{(' ' + context) if context else ''}: the classifier folder for {pair} "
              f"is out of date with the Python labelling ({shown}). Rerun "
              f"`python scripts/infer.py -p {pair}` before mining rules from it.")
    return bool(names)
