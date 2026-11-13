"""
Segmentation — from per-window predicate facts to clean regime INTERVALS.

The problem this stage solves: windows overlap (step 1, length 50), so ~50 windows
cover every day and they frequently disagree. Something has to turn that into one
regime per day, then into intervals.

Three smoothing methods, selected by `segmentation.method` in config.yaml:
  none     the raw argmax per window. Jumpy; useful as the ablation floor.
  majority rolling majority vote over a window of timepoints (nb_11's current
           hack). Two arbitrary knobs, no meaning, and the output moves when they
           move — kept because the paper wants it as an ablation row.
  hmm      Viterbi decoding with one interpretable parameter: stickiness p, where
           1/(1-p) is the expected regime duration in days (p=0.98 -> ~50 days ~=
           one window). Overrides isolated one-day contradictions but follows a
           sustained change, and yields exactly one regime per day by construction,
           so the "exclusive vs everything" conflict-resolution step disappears. Uses hmmlearn if present, otherwise a ~30-line
           Viterbi implemented here.

Interval convention throughout: [start, end) half-open, in day indices on the real
timeline; {pair}_dates.csv maps day -> calendar date.
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from tslearning.config import CFG, PATHS

REGIME_LABELS = list(CFG["classifier"]["softmax_group"])
FLAG_LABELS = list(CFG["classifier"]["multilabel"])
PRED_KIND = CFG["predicates"]["kind"]
ALL_PREDICATES = REGIME_LABELS + FLAG_LABELS


# ══════════════════════════════════════════════════════════════════════════════
#  Smoothing
# ══════════════════════════════════════════════════════════════════════════════

def smooth_none(winner: np.ndarray) -> np.ndarray:
    return np.asarray(winner, int)


def smooth_majority(winner: np.ndarray, window: int = 50) -> np.ndarray:
    """Rolling majority vote over the winner sequence."""
    from scipy.stats import mode

    winner = np.asarray(winner, int)
    out = []
    for i in range(len(winner)):
        lo = max(0, i - window // 2)
        hi = min(len(winner), i + window // 2)
        out.append(int(mode(winner[lo:hi], keepdims=False).mode))
    return np.array(out, int)


def smooth_hmm(probs: np.ndarray, stickiness: "float | None" = None) -> np.ndarray:
    """Viterbi decode of the most probable regime sequence.

    probs : (T, K) per-timepoint class probabilities. Ideally CALIBRATED — a raw
    softmax is overconfident, which makes the emission term dominate the transition
    term (temperature scaling would fix this; this function is
    C2 and works either way).

    Transition matrix: stay with probability p, switch to each of the other K-1
    states with (1-p)/(K-1).
    """
    probs = np.asarray(probs, float)
    if probs.ndim != 2:
        raise ValueError(f"probs must be (T, K), got {probs.shape}")
    T, K = probs.shape
    p = CFG["segmentation"]["hmm_stickiness"] if stickiness is None else stickiness

    trans = np.full((K, K), (1.0 - p) / max(K - 1, 1))
    np.fill_diagonal(trans, p)

    eps = 1e-12
    log_e = np.log(probs + eps)
    log_t = np.log(trans + eps)

    delta = np.log(np.full(K, 1.0 / K)) + log_e[0]
    psi = np.zeros((T, K), dtype=int)
    for t in range(1, T):
        scores = delta[:, None] + log_t          # (from, to)
        psi[t] = scores.argmax(0)
        delta = scores.max(0) + log_e[t]

    path = np.zeros(T, dtype=int)
    path[-1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]
    return path


def smooth(winner=None, probs=None, method: "str | None" = None,
           majority_window: int = 50, stickiness: "float | None" = None,
           verbose: bool = True) -> np.ndarray:
    """Dispatch on `method` (default: config segmentation.method).

    `probs` is required for hmm; `winner` suffices for none/majority. When only
    `probs` is given, winner is its argmax.
    """
    method = (method or CFG["segmentation"]["method"]).lower()
    if winner is None:
        if probs is None:
            raise ValueError("need winner or probs")
        winner = np.asarray(probs).argmax(1)

    if method == "none":
        out = smooth_none(winner)
    elif method == "majority":
        out = smooth_majority(winner, majority_window)
    elif method == "hmm":
        if probs is None:
            # Degrade to a hard one-hot emission rather than silently doing something
            # else: Viterbi still enforces persistence, it just cannot weigh
            # confidence. Calibrated probabilities are strictly better.
            if verbose:
                print("hmm: no probabilities given -> using one-hot emissions "
                      "(pass probs from the classifier for a proper decode)")
            K = int(np.max(winner)) + 1
            probs = np.eye(K)[np.asarray(winner, int)] * 0.9 + 0.1 / K
        out = smooth_hmm(probs, stickiness)
    else:
        raise ValueError(f"unknown smoothing method {method!r}")

    if verbose:
        changes = int((out[1:] != out[:-1]).sum())
        print(f"smoothing={method}: {len(out)} timepoints, {changes} regime changes, "
              f"avg run {len(out)/max(changes,1):.1f} timepoints")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Emit predicate CSVs (the reasoner's own schema, so everything downstream is uniform)
# ══════════════════════════════════════════════════════════════════════════════

def write_regime_csvs(regime: np.ndarray, timepoints, out_dir, seq_len: int = 50,
                      labels=None, name_a: str = "sp500_real", name_b: str = "btc_real",
                      verbose: bool = True) -> dict:
    """One CSV per regime class, columns X,Y,Start,End — the same schema the
    reasoner writes, which is what lets nb_11 read either source identically."""
    labels = labels or REGIME_LABELS
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = defaultdict(list)
    for i, t in enumerate(timepoints):
        rows[labels[int(regime[i])]].append(
            {"X": f"{name_a}_{t}", "Y": f"{name_b}_{t}", "Start": t, "End": t + seq_len})

    counts = {}
    for name, rr in rows.items():
        pd.DataFrame(rr).to_csv(out_dir / f"{name}.csv", index=False)
        counts[name] = len(rr)
        if verbose:
            print(f"  {name:26s} {len(rr):5d} -> {out_dir / (name + '.csv')}")
    return counts


def load_predicate_raw(name: str, temporal_dir) -> pd.DataFrame:
    """Read one {predicate}.csv. A predicate that never fired has no file; return an
    empty frame with the right columns rather than raising."""
    kind = PRED_KIND[name]
    cols = ["X", "Y", "Start", "End"] if kind == "pair_bin" else ["X", "Start", "End"]
    path = Path(temporal_dir) / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame(columns=cols)
    return pd.read_csv(path)


# ══════════════════════════════════════════════════════════════════════════════
#  Coalescing into intervals
# ══════════════════════════════════════════════════════════════════════════════

def series_of(name) -> str:
    """'sp500_real_1234' -> 'sp500'; 'btc_real_12' -> 'bitcoin' (btc normalised)."""
    name = str(name)
    for sep in ("_real_", "_synth_"):
        if sep in name:
            name = name.split(sep)[0]
    return "bitcoin" if name == "btc" else name


def coalesce_presence(df, kind: str) -> list:
    """Merge overlapping/adjacent [Start,End) intervals per key.

    pair_bin -> key is (series_a, series_b); unary -> key is the series name.
    Returns [(key, start, end), ...]. (nb_11 cells 11-12.)
    """
    if df is None or len(df) == 0:
        return []
    bykey = defaultdict(list)
    for _, r in df.iterrows():
        key = ((series_of(r["X"]), series_of(r["Y"])) if kind == "pair_bin"
               else series_of(r["X"]))
        bykey[key].append((int(r["Start"]), int(r["End"])))

    out = []
    for key, ivs in bykey.items():
        ivs.sort()
        cs, ce = ivs[0]
        for s, e in ivs[1:]:
            if s <= ce:               # overlapping or touching -> extend
                ce = max(ce, e)
            else:
                out.append((key, cs, ce))
                cs, ce = s, e
        out.append((key, cs, ce))
    return out


def coalesce_regime(regime: np.ndarray, timepoints, labels=None) -> list:
    """Merge consecutive same-label timepoints into intervals.

    Applies to a SMOOTHED sequence, which is already one label per timepoint — so
    no conflict resolution is needed, unlike the raw overlapping-window case.
    """
    labels = labels or REGIME_LABELS
    out = []
    for i, t in enumerate(timepoints):
        lab = labels[int(regime[i])]
        if out and out[-1][0] == lab and int(t) == out[-1][2]:
            out[-1] = (lab, out[-1][1], int(t) + 1)
        else:
            out.append((lab, int(t), int(t) + 1))
    return out


def build_predicate_intervals(source_dir, pair: "str | None" = None,
                              predicates=None, verbose: bool = True) -> dict:
    """Coalesce every predicate CSV in a folder into interval facts.

    Handles both layouts this project produced:
      temporal/{predicate}.csv                  (single pair, nb_10_part_2 Part B)
      <regime_dir>/<pair>/{predicate}_{pair}.csv (multi pair, py_labeler / recalibrate)

    Returns {predicate: [(key, start, end), ...]}.
    """
    source_dir = Path(source_dir)
    predicates = predicates or ALL_PREDICATES

    out = {}
    for pred in predicates:
        candidates = [source_dir / f"{pred}.csv"]
        if pair:
            candidates.insert(0, source_dir / f"{pred}_{pair}.csv")
        path = next((c for c in candidates if c.exists()), None)
        out[pred] = ([] if path is None
                     else coalesce_presence(pd.read_csv(path), PRED_KIND[pred]))

    if verbose:
        print("coalesced interval facts per predicate:")
        for pred in predicates:
            print(f"  {pred:26s} {len(out.get(pred, [])):>5} intervals")
    return out


def intervals_to_frame(predicate_intervals: dict) -> pd.DataFrame:
    """Flatten to a tidy frame: predicate, key_a, key_b, start, end, length."""
    rows = []
    for pred, ivs in predicate_intervals.items():
        for key, s, e in ivs:
            a, b = key if isinstance(key, tuple) else (key, "")
            rows.append({"predicate": pred, "key_a": a, "key_b": b,
                         "start": int(s), "end": int(e), "length": int(e) - int(s)})
    cols = ["predicate", "key_a", "key_b", "start", "end", "length"]
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values(["predicate", "start"]).reset_index(drop=True)


def attach_dates(df: pd.DataFrame, dates_csv) -> pd.DataFrame:
    """Add calendar start_date / end_date columns using a {pair}_dates.csv map.

    Rules are mined in day indices but must be REPORTED in calendar days, and break dates only mean something as dates.
    """
    dates_csv = Path(dates_csv)
    if not dates_csv.exists():
        return df
    m = pd.read_csv(dates_csv).set_index("day")["date"].to_dict()
    out = df.copy()
    out["start_date"] = out["start"].map(m)
    out["end_date"] = (out["end"] - 1).map(m)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Stage runner
# ══════════════════════════════════════════════════════════════════════════════

def run(pair: str = "sp500_bitcoin", source: "str | None" = None,
        smoothing: "str | None" = None, majority_window: int = 50,
        stickiness: "float | None" = None, out_dir=None, seq_len: "int | None" = None,
        verbose: bool = True) -> dict:
    """Segment one pair: smooth (if a classifier bundle is available), coalesce,
    write the interval table the rule stage consumes.

    `source` is the folder holding the predicate CSVs. Default resolution order:
      1. real_pairs_regime_py/<pair>      (python labeler)
      2. real_pairs_regime_reasoner/<pair>(reasoner, recalibrated)
      3. temporal/                        (classifier output, single pair)
    """
    data_dir = PATHS["data_processed"]
    seq_len = seq_len or 50

    if source is None:
        for cand in (data_dir / "real_pairs_regime_py" / pair,
                     data_dir / "real_pairs_regime_reasoner" / pair,
                     data_dir / "temporal"):
            if cand.exists():
                source = cand
                break
        else:
            raise FileNotFoundError(
                "no predicate CSVs found. Run label.py / prep_real_pairs.py / infer.py "
                f"first, or pass --source. Looked under {data_dir}")
    source = Path(source)
    if verbose:
        print(f"segmenting {pair} from {source}")

    # If a classifier prediction bundle exists, (re)smooth the regime and rewrite the
    # regime CSVs before coalescing. Otherwise the regime CSVs on disk are already
    # one-label-per-window (python labeler / recalibrated reasoner) and are used as is.
    bundle = data_dir / "predictions_lag0.pkl"
    smoothed_info = None
    if source.name == "temporal" and bundle.exists():
        import pickle
        with open(bundle, "rb") as f:
            loaded = pickle.load(f)
        timepoints = list(loaded["timepoints"])
        probs = loaded.get("probs")
        winner = np.asarray(loaded["regime"], int)
        sm = smooth(winner=winner, probs=probs, method=smoothing,
                    majority_window=majority_window, stickiness=stickiness,
                    verbose=verbose)
        counts = write_regime_csvs(sm, timepoints, source, seq_len=seq_len,
                                  labels=loaded.get("soft_cols", REGIME_LABELS),
                                  verbose=verbose)
        smoothed_info = {"method": smoothing or CFG["segmentation"]["method"],
                         "counts": counts,
                         "changes": int((sm[1:] != sm[:-1]).sum())}
    elif verbose and smoothing:
        print(f"note: --smoothing {smoothing} ignored — {source.name} holds one label "
              "per window already (no per-timepoint probabilities to decode)")

    predicate_intervals = build_predicate_intervals(source, pair=pair, verbose=verbose)
    frame = intervals_to_frame(predicate_intervals)

    for dates in (source / f"{pair}_dates.csv", source.parent / f"{pair}_dates.csv"):
        if dates.exists():
            frame = attach_dates(frame, dates)
            break

    out_dir = Path(out_dir or (data_dir / "intervals"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pair}_intervals.csv"
    frame.to_csv(out_path, index=False)
    if verbose:
        print(f"wrote -> {out_path}  ({len(frame)} intervals)")

    return {"pair": pair, "source": str(source), "smoothing": smoothed_info,
            "predicate_intervals": predicate_intervals, "frame": frame,
            "out_path": out_path}
