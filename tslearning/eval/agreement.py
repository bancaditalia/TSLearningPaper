"""
Label agreement — does the Python labeler produce the same facts as the reasoner?

This is the label-agreement number (regime agreement with the reasoner on real windows).
It belongs in the paper, so it belongs in a script.

────────────────────────────────────────────────────────────────────────────────
WHAT IT CHECKS, AND HOW IT DIFFERS FROM RULE VERIFICATION
────────────────────────────────────────────────────────────────────────────────

`verify_rules.py --engine both` checks the RULE layer: given identical predicate facts,
do the Python scorer and the reasoner derive the same rule confidence? It holds the
facts fixed.

This checks the PREDICATE layer, one level below: given identical raw returns, do
`py_labeler` and `preLabeling.vada` derive the same regime? It is the link the rule
verification explicitly does not cover, and stating both is what makes the verification
claim honest rather than overreaching.

Three outcomes are reported separately, because they mean different things:

  agree        the two paths assigned the same regime
  disagree     they assigned different regimes — a real discrepancy, listed individually
  unlabelled   the ENGINE assigned nothing at all

That last one is not a rounding error. A window on which no day is magnitude-consistent
emits no `...ConsistencyPercentage` fact — absent, not zero — so every regime rule fails
to join and the window silently vanishes. Python fills the missing side with 0.0 and
labels it. Measured at 13% of the synthetic transfer set and 0% of the real test set, so
it hides on real data and bites on synthetic.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from tslearning.config import PATHS

REGIMES = ["isPositiveCorrelation", "isNegativeCorrelation", "isNoCorrelation"]


def engine_labels(results_dir) -> dict:
    """{(X,Y) unordered: regime} from the reasoner's regime CSVs in one folder.

    The folder must hold ONE pair. Every loader in this project globs
    `{predicate}_*.csv`, so two pairs in one folder merge into each other silently.
    """
    out = {}
    for r in REGIMES:
        for f in sorted(glob.glob(str(Path(results_dir) / f"{r}_*.csv"))):
            d = pd.read_csv(f)
            if d.empty:
                continue
            for x, y in zip(d["X"], d["Y"]):
                if x != y:
                    out[tuple(sorted((x, y)))] = r
    return out


def python_labels(ts: dict, deadband: float) -> dict:
    """{(X,Y) unordered: regime} from py_labeler's concordance on the same windows."""
    from tslearning.labeling.py_labeler import compute_concordance

    out, scores = {}, {}
    for i, t in enumerate(ts["timepoints"]):
        same, diff = compute_concordance(ts["windows_a"][i], ts["windows_b"][i],
                                         denom=ts["seq_len"])
        s = same - diff
        key = tuple(sorted((f"{ts['base']}_real_{t}", f"{ts['other']}_real_{t}")))
        out[key] = (REGIMES[0] if s > deadband else
                    REGIMES[1] if s < -deadband else REGIMES[2])
        scores[key] = s
    return out, scores


FLAGS = ["highVol", "lowVol", "tailDependent"]


def flag_agreement(ts: dict, results_dir) -> dict:
    """{predicate: (n_agree, n_windows, mismatches)} for the volatility and tail flags.

    Same windows, same per-window tau / qLow (thresholds.window_values: train or rolling,
    exactly what the batch files gave the engine). highVol / lowVol are judged per
    series window, tailDependent per window pair. Added 2026-10-21 with the per-window
    thresholds fix: before it, the engine never applied tau or qLow.
    """
    from tslearning.labeling import thresholds as thr
    from tslearning.labeling.exact_flags import cocrash_count, label_constants, volatility

    base, other = ts["base"], ts["other"]
    names = {}
    for i, t in enumerate(ts["timepoints"]):
        for side, key in (("a", base), ("b", other)):
            names[f"{key}_real_{t}"] = (key, ts["dates"][t], i, side)
    tau, q = thr.window_values({w: (k, d) for w, (k, d, _, _) in names.items()})
    tail_min = label_constants()["tail_min"]

    py = {f: set() for f in FLAGS}
    for w, (k, d, i, side) in names.items():
        win = ts["windows_a"][i] if side == "a" else ts["windows_b"][i]
        lab, _ = volatility(win, tau[w])
        if lab:
            py[lab].add(w)
    for i, t in enumerate(ts["timepoints"]):
        wa, wb = f"{base}_real_{t}", f"{other}_real_{t}"
        if cocrash_count(ts["windows_a"][i], ts["windows_b"][i], q[wa], q[wb]) >= tail_min:
            py["tailDependent"].add(tuple(sorted((wa, wb))))

    eng = {f: set() for f in FLAGS}
    for f in FLAGS:
        for path in sorted(glob.glob(str(Path(results_dir) / f"{f}_*.csv"))):
            d = pd.read_csv(path)
            if d.empty:
                continue
            if f == "tailDependent":
                eng[f] |= {tuple(sorted((x, y))) for x, y in zip(d["X"], d["Y"]) if x != y}
            else:
                eng[f] |= set(d["X"])
    out = {}
    universe = {"highVol": set(names), "lowVol": set(names),
                "tailDependent": {tuple(sorted((f"{base}_real_{t}", f"{other}_real_{t}")))
                                  for t in ts["timepoints"]}}
    for f in FLAGS:
        u = universe[f]
        e, p_ = eng[f] & u, py[f] & u
        mism = sorted(e ^ p_)
        out[f] = (len(u) - len(mism), len(u), mism, len(e), len(p_))
    return out


def run(pair: str = "sp500_gold", step: int = 10, results_dir=None,
        deadband: "float | None" = None, save=None, verbose: bool = True) -> dict:
    """Compare the two labelling paths on one pair's test-period windows."""
    from tslearning.eval.fidelity import build_real_pair_test_set
    from tslearning.labeling.py_labeler import default_deadband

    deadband = default_deadband() if deadband is None else float(deadband)
    ts = build_real_pair_test_set(pair, step=step, verbose=verbose)
    slug = ts["pair"]
    results_dir = Path(results_dir or (PATHS["data_processed"] / f"real_test_{slug}"))
    hint = (f"  python scripts/eval.py --metric fidelity --dataset real "
            f"--pair {pair} --step {step} --label-first")
    if not results_dir.exists():
        raise FileNotFoundError(
            f"{results_dir} not found — label this pair first:\n{hint}")

    eng = engine_labels(results_dir)
    # Distinguish "never labelled" from "labelling still in progress". The reasoner
    # writes each CSV's header up front, so a folder full of header-only files looks
    # identical to a finished run that labelled nothing — and would be reported as
    # 100% disagreement rather than as an unfinished job.
    if not eng:
        n_batches = len({Path(f).stem.rsplit("_", 1)[-1]
                         for f in glob.glob(str(results_dir / "*_*.csv"))})
        raise FileNotFoundError(
            f"{results_dir} holds no regime facts yet ({n_batches} batch tag(s) "
            f"present, all empty). Either the reasoner has not finished, or it "
            f"failed. Wait for it, then re-run this; to (re)start the labelling:\n"
            f"{hint}")
    py, scores = python_labels(ts, deadband)

    agree, dis, missing = 0, [], []
    for key, p in py.items():
        e = eng.get(key)
        if e is None:
            missing.append((key, p, scores[key]))
        elif e == p:
            agree += 1
        else:
            dis.append((key, e, p, scores[key]))

    n = len(py)
    scored = agree + len(dis)

    # A PARTIAL run is the dangerous case: the fully-empty folder is caught above, but
    # 2 of 6 batches done looks like a finished run whose engine dropped two thirds of
    # the windows — i.e. exactly like the missing-percentage bug this tool exists to
    # detect. Compare against how many windows the engine should have covered.
    partial = len(missing) > 0 and scored > 0
    if partial:
        covered_tags = {Path(f).stem.rsplit("_", 1)[-1]
                        for f in glob.glob(str(results_dir / "isNoCorrelation_*.csv"))}
        expected_tags = (n + 24) // 25
        if len(covered_tags) < expected_tags:
            raise RuntimeError(
                f"{results_dir} looks INCOMPLETE: {len(covered_tags)} batch tag(s) "
                f"present but {expected_tags} expected for {n} windows at step "
                f"{step}, and {len(missing)} window(s) have no engine label.\n"
                f"That is almost certainly a labelling run still in progress (or one "
                f"that stopped early) rather than a real disagreement. Wait for it to "
                f"finish, then re-run.\n"
                f"If the run really did finish, pass --results explicitly to score the "
                f"partial folder anyway.")
    row = {"pair": slug, "step": step, "deadband": deadband, "n_windows": n,
           "n_scored": scored, "n_agree": agree, "n_disagree": len(dis),
           "n_unlabelled_by_engine": len(missing),
           "agreement": (agree / scored) if scored else float("nan")}
    for r in REGIMES:
        row[f"engine_{r}"] = sum(v == r for v in eng.values())
        row[f"python_{r}"] = sum(v == r for v in py.values())
    flags = flag_agreement(ts, results_dir)
    for f, (ok, tot, _m, ne, np_) in flags.items():
        row[f"{f}_agree"], row[f"{f}_windows"] = ok, tot
        row[f"{f}_engine"], row[f"{f}_python"] = ne, np_

    if verbose:
        print(f"\n===== label agreement · {slug} · step {step} · deadband "
              f"{deadband:g} =====")
        print(f"{'regime':<24}{'engine':>8}{'python':>8}")
        for r in REGIMES:
            print(f"{r:<24}{row[f'engine_{r}']:>8}{row[f'python_{r}']:>8}")
        print(f"\nagree {agree}/{scored}"
              + (f" = {row['agreement']:.3f}" if scored else "")
              + f"   disagree {len(dis)}   unlabelled by engine {len(missing)}")
        for key, e, p, s in dis[:15]:
            print(f"  MISMATCH {key[0]} — engine {e} / python {p} (score {s:+.0f})")
        if len(dis) > 15:
            print(f"  ... and {len(dis) - 15} more")
        if missing:
            print(f"  {len(missing)} window(s) got NO engine label — a percentage fact "
                  f"was absent rather than zero, so no regime rule could join")
        if scored and not dis and not missing:
            print("  the .vada dominance rules reproduce the Python labels exactly")
        print(f"\n{'flag':<16}{'engine':>8}{'python':>8}   agree")
        for f, (ok, tot, mism, ne, np_) in flags.items():
            print(f"{f:<16}{ne:>8}{np_:>8}   {ok}/{tot}"
                  + (f"   e.g. {', '.join(map(str, mism[:3]))}" if mism else ""))

    if save:
        out = Path(save)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(out, index=False)
        if verbose:
            print(f"wrote -> {out}")

    return {"row": row, "disagreements": dis, "unlabelled": missing,
            "engine": eng, "python": py}
