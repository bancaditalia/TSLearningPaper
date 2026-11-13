"""
Recalibrate + consolidate a real pair's reasoner output — WITHOUT re-running the reasoner.

Why: preLabeling.vada decides the correlation regime with a fixed *absolute* cut on the
compound magnitude+sign concordance (isPositive if same-sign% > 55, isNegative if diff-sign%
> 45). On real daily data that concordance tops out around ~46, so the cut never fires and
every window is "no correlation". The concordance percentages ARE emitted though, so we can
re-derive the regime from them by *relative dominance*:

    score = sameSign%  -  diffSign%
    score >  deadband  -> isPositiveCorrelation   (net co-movement in the same direction)
    score < -deadband  -> isNegativeCorrelation   (net co-movement in opposite directions)
    |score| <= deadband -> isNoCorrelation        (no clear tilt)

The absolute level is low on both sides; what matters is which direction wins. `deadband` is
how big a gap (in percentage points) we demand before committing to a regime.

This also remaps every predicate's [Start, End) from the disjoint-block position used in the
reasoner batches back to the REAL day index parsed from the window name (`sp500_real_{t}`),
so the coalescing step in nb_11 works on a real timeline.

To make the regime reasoner-native later, change lines 86-87 of preLabeling.vada to compare
the two percentages instead of the fixed 55/45, then re-run — this module is the fast
stand-in so the rule work isn't blocked on that.
"""
from __future__ import annotations
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

REGIMES = ["isPositiveCorrelation", "isNegativeCorrelation", "isNoCorrelation"]
_DERIVED = set(REGIMES) | {
    "magnitudeSameSignConsistencyPercentage",
    "magnitudeDiffSignConsistencyPercentage",
}


def _timepoint(name) -> int:
    """`sp500_real_1230` -> 1230 (the real window-start day index)."""
    return int(str(name).split("_")[-1])


def _concat(in_dir, pred, pair, seq_len):
    """Concatenate a predicate's per-batch CSVs; remap block-time -> real day."""
    frames = [pd.read_csv(f) for f in sorted(glob.glob(str(in_dir / f"{pred}_{pair}_*.csv")))]
    frames = [f for f in frames if not f.empty]           # avoids the empty-concat warning
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "X" in df.columns and "Start" in df.columns:
        df["Start"] = df["X"].map(_timepoint)
        df["End"] = df["Start"] + seq_len
    return df


def recalibrate_pair(pair, in_dir, out_dir, deadband: float = 5.0, seq_len: int = 50):
    """
    Recompute the correlation regime for `pair` from the emitted concordance percentages and
    write a clean, real-timeline predicate set for nb_11.

    Returns (regime_df, summary_dict). `regime_df` has columns X, Y, Start, End, same, diff,
    score, regime. Writes `{predicate}_{pair}.csv` (regime + all flags) into out_dir/pair/.
    """
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    key = ["X", "Y", "Start", "End"]
    dst = out_dir / pair
    os.makedirs(dst, exist_ok=True)

    same = _concat(in_dir, "magnitudeSameSignConsistencyPercentage", pair, seq_len)
    diff = _concat(in_dir, "magnitudeDiffSignConsistencyPercentage", pair, seq_len)
    anchor = _concat(in_dir, "percentagePositive", pair, seq_len)  # present for every window
    if anchor.empty:
        raise FileNotFoundError(f"no percentagePositive_{pair}_*.csv in {in_dir}")

    m = anchor[key].drop_duplicates()
    m = m.merge(same[key + ["Pct"]].rename(columns={"Pct": "same"}), on=key, how="left")
    m = m.merge(diff[key + ["Pct"]].rename(columns={"Pct": "diff"}), on=key, how="left")
    m[["same", "diff"]] = m[["same", "diff"]].fillna(0.0)
    m["score"] = m["same"] - m["diff"]
    m["regime"] = np.where(m["score"] > deadband, "isPositiveCorrelation",
                  np.where(m["score"] < -deadband, "isNegativeCorrelation", "isNoCorrelation"))

    for reg in REGIMES:
        m.loc[m["regime"] == reg, key].to_csv(dst / f"{reg}_{pair}.csv", index=False)

    # carry over the remaining predicates (flags, tail dependence, …) on the real timeline
    others = sorted({
        re.match(rf"(.+)_{pair}_\d+\.csv$", os.path.basename(p)).group(1)
        for p in glob.glob(str(in_dir / f"*_{pair}_*.csv"))
    } - _DERIVED)
    for pred in others:
        d = _concat(in_dir, pred, pair, seq_len)
        if not d.empty:
            d.to_csv(dst / f"{pred}_{pair}.csv", index=False)

    summary = {
        "regime_counts": m["regime"].value_counts().to_dict(),
        "score_quartiles": [round(float(q), 1) for q in np.percentile(m["score"], [0, 25, 50, 75, 100])],
        "n_windows": int(len(m)),
        "flags_carried": others,
        "out_dir": str(dst),
    }
    return m, summary
