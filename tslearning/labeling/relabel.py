"""
Re-derive the correlation regime from percentages the reasoner ALREADY emitted.

────────────────────────────────────────────────────────────────────────────────
WHY THIS EXISTS
────────────────────────────────────────────────────────────────────────────────

`preLabeling.vada` decides the regime with two ABSOLUTE cuts, each testing one
percentage in isolation (lines 86-87):

    isPositiveCorrelation :- magnitudeSameSignConsistencyPercentage(X,Y,P), P > 55.0.
    isNegativeCorrelation :- magnitudeDiffSignConsistencyPercentage(X,Y,P), P > 45.0.

Those bars are reachable on ENGINEERED SYNTHETIC pairs, where injected correlation
pushes the concordance high. On REAL daily returns they are not: the magnitude gate
discards most days before anything is counted, so both percentages stay small. Measured
on the real test set (71 windows, 2022-2024):

    max same% = 56   (the bar is 55 — exactly one window clears it)
    max diff% = 32   (the bar is 45 — UNREACHABLE, isNegativeCorrelation cannot fire)

Result: 67 / 4 / 0 across the three classes. Any three-class metric computed against
that is uninformative — a classifier that ignores its input and always answers
"isNoCorrelation" scores the same 0.944 accuracy as a trained one.

The fix is to compare the two percentages against EACH OTHER instead of against fixed
bars — relative dominance:

    score = same% - diff%
    score >  +deadband  ->  isPositiveCorrelation
    score <  -deadband  ->  isNegativeCorrelation
    |score| <= deadband ->  isNoCorrelation

The absolute level is low on both sides; which side wins is what carries the signal.

────────────────────────────────────────────────────────────────────────────────
WHAT THIS MODULE IS FOR, AND WHAT IT IS NOT
────────────────────────────────────────────────────────────────────────────────

It re-derives the regime from percentages that are already on disk. It runs NO
reasoning — the expensive part (computing the concordance) already happened, and the
part being replaced is a threshold applied afterwards. Relabelling 21 batches costs
about a second; re-running them through the engine would cost hours.

It is a STAND-IN, not the destination. The permanent fix is to change lines 86-87 of
`preLabeling.vada` to compare the percentages, so the reasoner is natively
non-degenerate and one definition holds end to end. This module exists so the
consequences of that change can be measured BEFORE committing to it — in particular
whether retraining the classifier on dominance labels resolves the real-data collapse.

Related: `recalibrate.py` does the same rescoring for the REAL-PAIR layout
(`{pred}_{pair}_{NNN}.csv`, windows named `sp500_real_{t}`) and additionally remaps
block time to the real timeline. This module handles the flat transfer-set / test-set
layout (`{pred}_{tag}.csv`, windows named `sp500_synth_{i}`) and changes nothing but
the regime.
"""
from __future__ import annotations

import glob
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REGIMES = ["isPositiveCorrelation", "isNegativeCorrelation", "isNoCorrelation"]
SAME = "magnitudeSameSignConsistencyPercentage"
DIFF = "magnitudeDiffSignConsistencyPercentage"
KEY = ["X", "Y", "Start", "End"]

# ONE source of truth: configs/config.yaml `labels.deadband`. The same constant is read
# by py_labeler (the facts the rules use) and passed to preLabeling.vada as ${Deadband}.
# Three independent defaults is what let the reasoner and the Python path drift apart.
from tslearning.config import CFG                                    # noqa: E402

DEFAULT_DEADBAND = float(CFG.get("labels", {}).get("deadband", 10.0))


def _tags(results_dir: Path) -> "list[str]":
    """Batch tags present, from the percentage files (every window has those)."""
    out = []
    for p in sorted(glob.glob(str(results_dir / f"{SAME}_*.csv"))):
        m = re.match(rf"^{re.escape(SAME)}_(.+)\.csv$", Path(p).name)
        if m:
            out.append(m.group(1))
    return out


def _read(results_dir: Path, pred: str, tag: str) -> pd.DataFrame:
    p = results_dir / f"{pred}_{tag}.csv"
    if not p.exists():
        return pd.DataFrame(columns=KEY + ["Pct"])
    d = pd.read_csv(p)
    return d if not d.empty else pd.DataFrame(columns=KEY + ["Pct"])


def dominance_frame(results_dir, tag: str,
                    deadband: float = DEFAULT_DEADBAND) -> pd.DataFrame:
    """One batch's windows with same%, diff%, score and the new regime.

    A pair missing from a percentage file means that predicate did not fire for it,
    which is a genuine zero — not missing data — so it is filled with 0.
    """
    results_dir = Path(results_dir)
    same = _read(results_dir, SAME, tag)
    diff = _read(results_dir, DIFF, tag)
    if same.empty and diff.empty:
        return pd.DataFrame(columns=KEY + ["same", "diff", "score", "regime"])

    base = pd.concat([same[KEY], diff[KEY]], ignore_index=True).drop_duplicates()
    m = base.merge(same[KEY + ["Pct"]].rename(columns={"Pct": "same"}), on=KEY, how="left")
    m = m.merge(diff[KEY + ["Pct"]].rename(columns={"Pct": "diff"}), on=KEY, how="left")
    m[["same", "diff"]] = m[["same", "diff"]].fillna(0.0)
    m["score"] = m["same"] - m["diff"]
    m["regime"] = np.where(m.score > deadband, "isPositiveCorrelation",
                   np.where(m.score < -deadband, "isNegativeCorrelation",
                            "isNoCorrelation"))
    return m


def run(results_dir, out_dir=None, deadband: float = DEFAULT_DEADBAND,
        copy_others: bool = True, all_tags: bool = False,
        verbose: bool = True) -> dict:
    """Write a relabelled copy of `results_dir`.

    Everything except the three regime CSVs is copied through unchanged, so the output
    is a drop-in replacement that `predicate_targets.build_targets` (and therefore
    distill.py / eval.py via --results) reads with no code change.

    NON-NUMERIC TAGS ARE SKIPPED by default (`all_tags=False`). `batched_results`
    contains a stray `*_test.csv` set — a debug run that reuses window indices 0..N
    from batch 000 with DIFFERENT values. Because every loader here globs
    `{predicate}_*.csv`, those rows silently join the training targets and contradict
    batch 000: window 0 is simultaneously isNegativeCorrelation (same 16 / diff 60) and
    isNoCorrelation (same 34 / diff 24). Keeping them produces windows carrying two
    regimes at once. Pass all_tags=True to mirror the input exactly instead.

    Returns a summary with the class counts before and after.
    """
    results_dir = Path(results_dir)
    out_dir = Path(out_dir or f"{results_dir}_dominance")
    out_dir.mkdir(parents=True, exist_ok=True)

    tags = _tags(results_dir)
    if not tags:
        raise FileNotFoundError(
            f"no {SAME}_*.csv in {results_dir} — this needs a reasoner run whose "
            f"output includes the concordance percentages")

    skipped = []
    if not all_tags:
        keep = [t for t in tags if t.isdigit()]
        skipped = [t for t in tags if not t.isdigit()]
        if keep:                       # never skip everything
            tags = keep

    # count the OLD labels first, so the comparison is on the same windows
    before = {r: 0 for r in REGIMES}
    for r in REGIMES:
        for p in glob.glob(str(results_dir / f"{r}_*.csv")):
            d = pd.read_csv(p)
            if not d.empty:
                before[r] += int((d.X != d.Y).sum())

    def _skipped_tag(name: str) -> bool:
        return any(name.endswith(f"_{t}.csv") or name == f"{t}.json" for t in skipped)

    if copy_others:
        for p in results_dir.iterdir():
            if not p.is_file():
                continue
            if any(p.name.startswith(f"{r}_") for r in REGIMES):
                continue               # regimes are regenerated below
            if _skipped_tag(p.name):
                continue               # do not carry the contradictory tag through
            if p.name == "timing.json":
                continue               # reasoner timing belongs to the original run only
            shutil.copy2(p, out_dir / p.name)

    after = {r: 0 for r in REGIMES}
    for tag in tags:
        m = dominance_frame(results_dir, tag, deadband)
        for r in REGIMES:
            sub = m[m.regime == r][KEY]
            sub.to_csv(out_dir / f"{r}_{tag}.csv", index=False)
            after[r] += int((sub.X != sub.Y).sum())

    if verbose:
        print(f"relabelled {len(tags)} batches  ->  {out_dir}")
        print(f"  deadband {deadband:g}")
        w = max(len(r) for r in REGIMES)
        for r in REGIMES:
            print(f"  {r:<{w}}  {before[r]:>6} -> {after[r]:>6}")
        if skipped:
            print(f"  SKIPPED non-numeric tag(s): {', '.join(skipped)} — these reuse "
                  f"window indices from batch 000 with different values and would put "
                  f"two regimes on one window (--all-tags to keep them)")
    return {"out_dir": out_dir, "tags": tags, "skipped": skipped,
            "before": before, "after": after, "deadband": deadband}
