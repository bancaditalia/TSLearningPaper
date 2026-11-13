"""
Read the reasoner's per-predicate CSVs and turn them into training targets.

Ported from the loader + target-builder cells duplicated in
nb_10_classifiers-alt.ipynb (cells 40-43) and nb_10_part_2.ipynb (cell 1). The
notebooks are NOT modified. Both the distillation trainer and the fidelity
evaluation need this, which is why it lives here rather than inside either.

The reasoner writes one CSV per predicate per batch: {predicate}_{tag}.csv, whose
SHAPE depends on the predicate. configs/config.yaml's `predicates.kind` registry is
what tells us how to read each one:

    pair_bin        X,Y            a fact about the pair            (isPositiveCorrelation)
    pair_cont:COL   X,Y,COL        a number about the pair          (percentagePositive)
    unary_bin       X              a fact about one series           (trendUp, highVol)
    unary_cont:COL  X,COL          a number about one series        (sumFlow, bigMovePct)
    lag_bin         X,Y,Lag        a fact about the pair at a lag    (leadsPositively)
    lag_cont:COL    X,Y,Lag,COL    a number about the pair at a lag  (positivePctLag)

Adding a predicate to the reasoner therefore needs no code change here — only a
`kind` entry, and a place in one of the classifier buckets
(softmax_group / multilabel / aux).
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from tslearning.config import CFG

PRED_KIND = CFG["predicates"]["kind"]
LAGS = CFG["predicates"]["lags"]
SOFTMAX_GROUP = list(CFG["classifier"]["softmax_group"])
MULTILABEL = list(CFG["classifier"]["multilabel"])
AUX = list(CFG["classifier"]["aux"])


# ── Pair-key helpers ───────────────────────────────────────────────────────────
# The reasoner may emit (a,b) or (b,a) for a symmetric predicate, so pairs are
# keyed by an unordered frozenset and only canonicalised for display/ordering.

def pkey(a, b):
    return frozenset((a, b))


def is_cross(a: str, b: str, side_a: str = "sp500", side_b: str = "btc") -> bool:
    """True for a genuine cross-series pair, false for self- and same-series pairs."""
    return (a.startswith(side_a) and b.startswith(side_b)) or \
           (a.startswith(side_b) and b.startswith(side_a))


def canon(fs, side_a: str = "sp500"):
    a, b = tuple(fs)
    return (a, b) if a.startswith(side_a) else (b, a)


# ── Loading ────────────────────────────────────────────────────────────────────

def load_predicate_csv(predicate: str, results_dir) -> pd.DataFrame:
    """Concatenate every per-batch {predicate}_*.csv in `results_dir`.

    Returns an empty frame if the predicate never fired in this run — a predicate
    with no file is normal, not an error.
    """
    paths = sorted(glob.glob(os.path.join(str(results_dir), f"{predicate}_*.csv")))
    if not paths:
        return pd.DataFrame()
    return pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)


def _store(predicate: str, results_dir):
    """Index one predicate's facts for O(1) lookup, shaped by its `kind`."""
    kind = PRED_KIND[predicate]
    d = load_predicate_csv(predicate, results_dir)
    if d.empty:
        return set() if "bin" in kind else {}

    if kind == "pair_bin":
        return {pkey(a, b) for a, b in zip(d["X"], d["Y"])}
    if kind.startswith("pair_cont"):
        c = kind.split(":")[1]
        return {pkey(a, b): float(v) for a, b, v in zip(d["X"], d["Y"], d[c])}
    if kind == "unary_bin":
        return set(d["X"])
    if kind.startswith("unary_cont"):
        c = kind.split(":")[1]
        return {x: float(v) for x, v in zip(d["X"], d[c])}
    if kind == "lag_bin":
        return {(pkey(a, b), int(l)) for a, b, l in zip(d["X"], d["Y"], d["Lag"])}
    c = kind.split(":")[1]
    return {(pkey(a, b), int(l)): float(v)
            for a, b, l, v in zip(d["X"], d["Y"], d["Lag"], d[c])}


def columns_for(predicate: str) -> "list[str]":
    """Column names this predicate contributes to a target matrix."""
    kind = PRED_KIND[predicate]
    if kind.startswith(("pair_bin", "pair_cont")):
        return [predicate]
    if kind.startswith(("unary_bin", "unary_cont")):
        return [f"{predicate}_a", f"{predicate}_b"]
    return [f"{predicate}_lag{L}" for L in LAGS]


def values_for(predicate: str, a: str, b: str, store) -> "list[float]":
    """This predicate's values for one pair, aligned with `columns_for`."""
    kind, key = PRED_KIND[predicate], pkey(a, b)
    if kind == "pair_bin":
        return [1.0 if key in store else 0.0]
    if kind.startswith("pair_cont"):
        return [store.get(key, 0.0)]
    if kind == "unary_bin":
        return [1.0 if a in store else 0.0, 1.0 if b in store else 0.0]
    if kind.startswith("unary_cont"):
        return [store.get(a, 0.0), store.get(b, 0.0)]
    if kind == "lag_bin":
        return [1.0 if (key, L) in store else 0.0 for L in LAGS]
    return [store.get((key, L), 0.0) for L in LAGS]


# ── Target matrices ────────────────────────────────────────────────────────────

def build_targets(results_dir, side_a: str = "sp500", side_b: str = "btc") -> dict:
    """Build the aligned pair list and the three target matrices.

    Returns a dict with:
      pair_names  [(a, b), ...] ordered by window index
      y_soft      (N, 3)  one-hot regime target for the softmax head
      y_multi     (N, M)  independent flag targets (kept for the 3-head variant)
      y_aux       (N, D)  auxiliary regression targets
      *_cols      the column names, and aux_is_binary for the aux ones
    """
    results_dir = Path(results_dir)

    pairs = set()
    for pred in SOFTMAX_GROUP + [p for p in AUX if PRED_KIND[p].startswith("pair")]:
        d = load_predicate_csv(pred, results_dir)
        if d.empty:
            continue
        for a, b in zip(d["X"], d["Y"]):
            if a != b and is_cross(a, b, side_a, side_b):
                pairs.add(pkey(a, b))

    def _window_index(p):
        try:
            return int(p[0].split("_")[-1])
        except ValueError:
            return 0

    pair_names = sorted((canon(fs, side_a) for fs in pairs), key=_window_index)
    stores = {p: _store(p, results_dir) for p in set(SOFTMAX_GROUP + MULTILABEL + AUX)}

    ys, ym, ya = [], [], []
    for (a, b) in pair_names:
        key = pkey(a, b)
        ys.append([1.0 if key in stores[p] else 0.0 for p in SOFTMAX_GROUP])
        ym.append([v for p in MULTILABEL for v in values_for(p, a, b, stores[p])])
        ya.append([v for p in AUX for v in values_for(p, a, b, stores[p])])

    return {
        "pair_names": pair_names,
        "y_soft": np.array(ys, dtype=np.float32).reshape(len(pair_names), -1),
        "y_multi": np.array(ym, dtype=np.float32).reshape(len(pair_names), -1),
        "y_aux": np.array(ya, dtype=np.float32).reshape(len(pair_names), -1),
        "soft_cols": list(SOFTMAX_GROUP),
        "multi_cols": [c for p in MULTILABEL for c in columns_for(p)],
        "aux_cols": [c for p in AUX for c in columns_for(p)],
        "aux_is_binary": [("bin" in PRED_KIND[p]) for p in AUX for _ in columns_for(p)],
    }


def normalize_aux(y_aux: np.ndarray, aux_cols, aux_is_binary) -> np.ndarray:
    """Scale continuous aux targets before an MSE head: percentages -> [0,1],
    everything else z-scored."""
    out = np.asarray(y_aux, np.float32).copy()
    for i, binary in enumerate(aux_is_binary):
        if binary:
            continue
        col = aux_cols[i]
        if ("Percentage" in col) or ("Pct" in col) or col.startswith("percentagePositive"):
            out[:, i] = y_aux[:, i] / 100.0
        else:
            mu, sd = y_aux[:, i].mean(), y_aux[:, i].std() + 1e-8
            out[:, i] = (y_aux[:, i] - mu) / sd
    return out
