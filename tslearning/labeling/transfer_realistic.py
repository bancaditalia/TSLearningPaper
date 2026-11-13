"""
Realistic transfer set — synthetic pairs that look like REAL ones.

Why a new generator. Measured on 2026-10-14 against real 50-day windows:
  * real correlation is MILD: on gold, 90% of real windows have a score
    (same% - diff%) within -14..+20; real negatives sit just past the -10 threshold.
    `transfer_set.generate_engineered` builds y as a copy / mirror of x, so synthetic
    positives and negatives are extreme (median negative -52).
  * the per-series DDPM windows have NO fat tails (excess kurtosis ~0, real 0.8-2.8).

What this generator does (the original one is untouched, so the existing transfer sets
stay reproducible):
  1. CORRELATION STRENGTH per pair: y = c * x_shape + sqrt(1 - c^2) * independent_shape,
     with c drawn per pair (positive, negative or near zero), instead of copy / mirror.
  2. REAL-LIKE TAILS: each window is mapped onto the distribution of real standardized
     returns of its own series (training years only): the window keeps its day-to-day
     ORDER (ranks) from the DDPM, the VALUES come from the real distribution. Done after
     building the pair, because blending two shapes thins the tails.
  3. HOW STRONG, balanced classes (1/3 positive, 1/3 negative, 1/3 none). Three modes:
       strength="boundary" (DEFAULT, data-free, chosen 2026-10-14) most pairs just past the
                          threshold, fewer and fewer further out: |score| = deadband +
                          Exponential(boundary_scale=15), capped at max_score; none uniform
                          within the deadband. Training examples go where the decision is
                          hard. No real pair information is used. On the bitcoin set: median
                          |score| ~20-22, about half within 10-20, 8-14% above 40.
       strength="even"    (data-free, rejected) U(+deadband, +max_score): ~60% of pairs end
                          up strong (> 40), so mild cases stay rare — the original problem.
       strength="matched" (2026-10-14, superseded) the target score is drawn from the
                          real 2015-2021 scores of that class, pooled over all pairs, with a
                          share `extreme_frac` strong (|score| > 40). This uses real PAIR
                          information (co-movement statistics), so it is NOT data-free; and
                          pooling made positives far too strong (nasdaq), so the classifier
                          missed real mild positives. Kept only to reproduce that run.
     A candidate pair is kept only if its score lands where the target says.
  4. Volatility, trend and co-crash injections exactly as in generate_engineered.

Labels are computed in Python (`py_labeler.compute_concordance`), which reproduces the
reasoner's same% / diff% exactly (525/525 windows on every existing transfer set,
2026-10-14). They are written in the reasoner's CSV format, so training reads them with
no code change; spot-check a batch on the engine for the paper.

Real information used, by level:
  * strength="boundary" / "even": only PER-SERIES information from the training years — the return
    distribution of each series (tail mapping), as the DDPM and the volatility thresholds
    already use. Nothing about how two series move TOGETHER: data-free in that sense.
  * strength="matched": additionally real PAIR-level score statistics — not data-free.
Real 2015-2021 pair windows are always used for the CHECK (comparison.csv), never to
build the set in "even" mode. No real window ever enters the transfer set.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tslearning.config import CFG, PATHS

SAME = "magnitudeSameSignConsistencyPercentage"
DIFF = "magnitudeDiffSignConsistencyPercentage"
REGIMES = ("isPositiveCorrelation", "isNegativeCorrelation", "isNoCorrelation")
EXTREME = 40.0                         # |score| above this counts as "extreme"


# ══════════════════════════════════════════════════════════════════════════════
#  Real reference (training years only)
# ══════════════════════════════════════════════════════════════════════════════


def _tail_k() -> int:
    """Joint-crash days injected into a "tail" pair: tail_min or tail_min + 1, so the
    reasoner's tailDependent (>= tail_min) fires on it. 2-3 days at the paper's W=50."""
    from tslearning.labeling.exact_flags import label_constants
    return label_constants()["tail_min"]

def _z(w):
    w = np.asarray(w, float)
    return (w - w.mean()) / (w.std() + 1e-12)


def _score(x, y, seq_len: int = 50) -> float:
    from tslearning.labeling.py_labeler import compute_concordance
    s, d = compute_concordance(x, y, denom=seq_len)
    return s - d


def _label(score: float, deadband: float) -> str:
    return ("isPositiveCorrelation" if score > deadband else
            "isNegativeCorrelation" if score < -deadband else "isNoCorrelation")


def real_reference(pairs=None, seq_len: int = 50, stride: int = 1) -> dict:
    """Real 2015-2021 windows of `pairs`: per-class score samples, and per-series pools
    of window-standardized returns (for the tail mapping)."""
    from tslearning.datasets.pairs import describe, load_series
    from tslearning.labeling.py_labeler import _aligned, default_deadband

    pairs = list(pairs or (CFG.get("paper") or {}).get("pairs") or [])
    lo, hi = CFG["split"]["train"]
    db = default_deadband()
    scores = {r: [] for r in REGIMES}
    pools, kurt = {}, {}
    for pair in pairs:
        d = describe(pair)
        ser = load_series([d["a"], d["b"]])
        idx = _aligned({d["a"]: ser[d["a"]], d["b"]: ser[d["b"]]}, lo, hi)
        a = ser[d["a"]].loc[idx].to_numpy(float)
        b = ser[d["b"]].loc[idx].to_numpy(float)
        for t in range(0, len(a) - seq_len + 1, stride):
            sc = _score(a[t:t + seq_len], b[t:t + seq_len], seq_len)
            scores[_label(sc, db)].append(sc)
        for name, r in ((d["a"], a), (d["b"], b)):
            key = name.lower()
            if key in pools:
                continue
            W = np.array([_z(r[t:t + seq_len])
                          for t in range(0, len(r) - seq_len + 1, seq_len)])
            pools[key] = W.ravel()
            kurt[key] = float(np.mean([((w ** 4).mean() - 3) for w in W]))
    return {"scores": {k: np.asarray(v) for k, v in scores.items()}, "pools": pools,
            "kurtosis": kurt, "deadband": db, "pairs": pairs, "period": [lo, hi]}


# ══════════════════════════════════════════════════════════════════════════════
#  Generator
# ══════════════════════════════════════════════════════════════════════════════

def _tail_map(w, pool, rng):
    """Keep the window's ranks, take its values from the real standardized pool."""
    vals = np.sort(rng.choice(pool, size=len(w), replace=False))
    out = np.empty(len(w))
    out[np.argsort(w, kind="stable")] = vals
    return _z(out)


def generate_realistic(pool_a, pool_b, n_pairs: int, tau: dict, qlow: dict, ref: dict,
                       series_a: str = "sp500", series_b: str = "gold",
                       extreme_frac: float = 0.2, high_vol_frac: float = 0.4,
                       strength: str = "boundary", max_score: float = 90.0,
                       boundary_scale: float = 15.0,
                       tail_frac: float = 0.25, seq_len: int = 50, tol: float = 4.0,
                       max_tries: int = 4000, seed: "int | None" = None):
    """(X, Y, manifest). See the module docstring."""
    rng = np.random.default_rng(seed if seed is not None else CFG["seed"])
    db = ref["deadband"]
    pa, pb = ref["pools"][series_a], ref["pools"][series_b]
    tx, ty = tau[series_a], tau[series_b]
    qx, qy = qlow[series_a], qlow[series_b]
    targets = [REGIMES[i % 3] for i in range(n_pairs)]
    rng.shuffle(targets)

    X, Y, man = [], [], []
    relaxed = 0
    if strength not in ("even", "boundary", "matched"):
        raise ValueError(f"strength must be 'even', 'boundary' or 'matched', got {strength!r}")
    for target in targets:
        if strength == "boundary":
            # most pairs just past the threshold, fewer and fewer further out:
            # |score| = deadband + Exponential(boundary_scale), capped at max_score
            extreme = False
            sign = {"isPositiveCorrelation": 1, "isNegativeCorrelation": -1}.get(target, 0)
            if sign:
                want = sign * float(min(db + rng.exponential(boundary_scale), max_score))
            else:
                want = float(rng.uniform(-db, db))
        elif strength == "even":
            extreme = False
            lo_, hi_ = ((db, max_score) if target == "isPositiveCorrelation" else
                        (-max_score, -db) if target == "isNegativeCorrelation" else
                        (-db, db))
            want = float(rng.uniform(lo_, hi_))
        else:
            extreme = target != "isNoCorrelation" and rng.random() < extreme_frac
            real = ref["scores"][target]
            want = None if extreme or len(real) == 0 else float(rng.choice(real))
        for attempt in range(max_tries):
            tol_now = tol * (1 + attempt // 1000)          # widen slowly if needed
            sign = {"isPositiveCorrelation": 1, "isNegativeCorrelation": -1}.get(target, 0)
            c = (sign * rng.uniform(0.6, 1.0) if extreme else
                 sign * rng.uniform(0.0, 1.0) if sign else rng.uniform(-0.3, 0.3))
            xs = _z(pool_a[rng.integers(len(pool_a))])
            ind = _z(pool_b[rng.integers(len(pool_b))])
            ys = c * xs + np.sqrt(max(0.0, 1 - c * c)) * ind
            xz, yz = _tail_map(xs, pa, rng), _tail_map(ys, pb, rng)

            hv_x, hv_y = rng.random() < high_vol_frac, rng.random() < high_vol_frac
            sx, sy = (1.2 if hv_x else 0.4) * tx, (1.2 if hv_y else 0.4) * ty
            dx, dy = rng.choice([0.05, -0.05, 0.0]), rng.choice([0.05, -0.05, 0.0])
            x = xz * sx + dx / seq_len
            y = yz * sy + dy / seq_len
            tail = rng.random() < tail_frac
            if tail:
                days = rng.choice(seq_len, size=int(rng.integers(_tail_k(), _tail_k() + 2)), replace=False)
                x[days], y[days] = qx * 1.5, qy * 1.5

            sc = _score(x, y, seq_len)
            if _label(sc, db) != target:
                continue
            if extreme and abs(sc) <= EXTREME:
                continue
            if want is not None and abs(sc - want) > tol_now:
                continue
            relaxed += attempt >= 1000
            break
        else:
            raise RuntimeError(f"could not generate a {target} pair near score {want} "
                               f"in {max_tries} tries")
        X.append(x.astype(np.float32))
        Y.append(y.astype(np.float32))
        man.append({"target": target, "extreme": bool(extreme), "c": round(float(c), 3),
                    "score": sc, "target_score": want,
                    "vol_x": "high" if hv_x else "low", "vol_y": "high" if hv_y else "low",
                    "trend_x": "up" if dx > 0 else "down" if dx < 0 else "flat",
                    "trend_y": "up" if dy > 0 else "down" if dy < 0 else "flat",
                    "tail": bool(tail)})
    manifest = pd.DataFrame(man)
    manifest.insert(0, "pair_id", range(n_pairs))
    manifest.attrs["relaxed"] = relaxed
    return np.array(X), np.array(Y), manifest


# ══════════════════════════════════════════════════════════════════════════════
#  Python labels in the reasoner's CSV format
# ══════════════════════════════════════════════════════════════════════════════

def write_python_labels(X, Y, out_dir, name_a: str, name_b: str, batch_size: int = 25,
                        seq_len: int = 50) -> dict:
    """{regime}_{NNN}.csv and same%/diff% CSVs per batch, as the reasoner writes them.
    Window i spans [i*seq_len, (i+1)*seq_len), as in build_pairs_csv."""
    from tslearning.labeling.py_labeler import compute_concordance, default_deadband

    out_dir = Path(out_dir)
    db = default_deadband()
    counts = {r: 0 for r in REGIMES}
    for b in range(len(X) // batch_size):
        rows = {r: [] for r in REGIMES}
        same, diff = [], []
        for i in range(b * batch_size, (b + 1) * batch_size):
            a, o = f"{name_a}_synth_{i}", f"{name_b}_synth_{i}"
            s, d = compute_concordance(X[i], Y[i], denom=seq_len)
            st, en = i * seq_len, (i + 1) * seq_len
            if s > 0:
                same.append({"X": a, "Y": o, "Pct": s, "Start": st, "End": en})
            if d > 0:
                diff.append({"X": a, "Y": o, "Pct": d, "Start": st, "End": en})
            r = _label(s - d, db)
            rows[r].append({"X": a, "Y": o, "Start": st, "End": en})
            counts[r] += 1
        for r in REGIMES:
            pd.DataFrame(rows[r], columns=["X", "Y", "Start", "End"]).to_csv(
                out_dir / f"{r}_{b:03d}.csv", index=False)
        pd.DataFrame(same, columns=["X", "Y", "Pct", "Start", "End"]).to_csv(
            out_dir / f"{SAME}_{b:03d}.csv", index=False)
        pd.DataFrame(diff, columns=["X", "Y", "Pct", "Start", "End"]).to_csv(
            out_dir / f"{DIFF}_{b:03d}.csv", index=False)
    return counts


# ══════════════════════════════════════════════════════════════════════════════
#  Comparison table (the check BEFORE any training)
# ══════════════════════════════════════════════════════════════════════════════

def describe_set(X, Y, deadband: float, seq_len: int = 50) -> dict:
    sc = np.array([_score(x, y, seq_len) for x, y in zip(X, Y)])
    neg = sc[sc < -deadband]
    k = lambda A: float(np.mean([((_z(w) ** 4).mean() - 3) for w in A]))
    return {"n": len(sc), "kurt_a": k(X), "kurt_b": k(Y),
            "q05": float(np.percentile(sc, 5)), "q25": float(np.percentile(sc, 25)),
            "q50": float(np.percentile(sc, 50)), "q75": float(np.percentile(sc, 75)),
            "q95": float(np.percentile(sc, 95)),
            "share_mild": float(np.mean(np.abs(sc) <= 20)),
            "pos": float(np.mean(sc > deadband)), "neg": float(np.mean(sc < -deadband)),
            "none": float(np.mean(np.abs(sc) <= deadband)),
            "neg_median": float(np.median(neg)) if len(neg) else None}


def run(pair: str = "sp500_gold", n_pairs: int = 525, extreme_frac: float = 0.2,
        out_dir=None, batch_size: int = 25, seed: "int | None" = None,
        ref=None, strength: str = "boundary", verbose: bool = True) -> dict:
    """Build transfer_<pair>_realistic/: windows, batch CSVs, Python labels, manifest,
    and comparison.csv (real 2015-2021 vs this set vs the current transfer set)."""
    from tslearning.datasets.pairs import resolve_pair
    from tslearning.labeling.transfer_set import (engineered_name, fit_thresholds,
                                                  load_thresholds, write_batches)
    from tslearning.synth import registry

    series_a, series_b, pair = resolve_pair(pair)
    # tau / qLow of THIS run's training period, fitted first: a new run has no thresholds
    # yet (prep_real_pairs fits them too, later), and the generator scales by them
    fit_thresholds([series_a, series_b], verbose=verbose)
    suffix = {"even": "_even", "boundary": "_boundary"}.get(strength, "_realistic")
    out_dir = Path(out_dir or (PATHS["data_processed"] / f"transfer_{pair}{suffix}"))
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty — pass a new --out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    a_name, b_name = engineered_name(series_a), engineered_name(series_b)

    tau, qlow = load_thresholds()
    arr_a = np.load(registry.sample_path(series_a))
    arr_b = np.load(registry.sample_path(series_b))
    seq_len = arr_a.shape[1]
    ref = ref or real_reference(seq_len=seq_len)   # real windows of the SAME length
    X, Y, man = generate_realistic(arr_a, arr_b, n_pairs, tau, qlow, ref,
                                   series_a=series_a.lower(), series_b=series_b.lower(),
                                   extreme_frac=extreme_frac, seq_len=seq_len, seed=seed,
                                   strength=strength)

    np.save(out_dir / f"engineered_{a_name}.npy", X)
    np.save(out_dir / f"engineered_{b_name}.npy", Y)
    man.to_csv(out_dir / "engineered_manifest.csv", index=False)
    write_batches(X, Y, out_dir, batch_size=batch_size, name_a=f"{a_name}_synth",
                  name_b=f"{b_name}_synth", verbose=False)
    counts = write_python_labels(X, Y, out_dir, a_name, b_name, batch_size, seq_len)

    # the check: real 2015-2021 (this pair) vs the new set vs the current set
    from tslearning.datasets.pairs import load_series
    from tslearning.labeling.py_labeler import _aligned
    lo, hi = ref["period"]
    ser = load_series([series_a, series_b])
    idx = _aligned({series_a: ser[series_a], series_b: ser[series_b]}, lo, hi)
    ra, rb = ser[series_a].loc[idx].to_numpy(float), ser[series_b].loc[idx].to_numpy(float)
    RA = np.array([ra[t:t + seq_len] for t in range(0, len(ra) - seq_len + 1)])
    RB = np.array([rb[t:t + seq_len] for t in range(0, len(rb) - seq_len + 1)])
    rows = {f"real {lo[:4]}-{hi[:4]}": describe_set(RA, RB, ref["deadband"], seq_len),
            "new (realistic)": describe_set(X, Y, ref["deadband"], seq_len)}
    cur = PATHS["data_processed"] / f"transfer_{pair}"
    if (cur / f"engineered_{a_name}.npy").exists():
        rows["current transfer set"] = describe_set(
            np.load(cur / f"engineered_{a_name}.npy"),
            np.load(cur / f"engineered_{b_name}.npy"), ref["deadband"], seq_len)
    comp = pd.DataFrame(rows).T
    comp.to_csv(out_dir / "comparison.csv")
    (out_dir / "generator.json").write_text(json.dumps({
        "generator": "transfer_realistic.generate_realistic", "pair": pair,
        "n_pairs": n_pairs, "strength": strength,
        "extreme_frac": extreme_frac if strength == "matched" else None,
        "seed": CFG["seed"] if seed is None else seed, "reference_pairs": ref["pairs"],
        "reference_period": ref["period"],
        "real_info_used": ("per-series return distributions only (tail mapping)"
                           if strength in ("even", "boundary") else
                           "per-series return distributions + real pair score statistics"),
        "labels": "python (compute_concordance)",
        "label_counts": counts, "relaxed_tolerance": int(man.attrs.get("relaxed", 0)),
    }, indent=2))
    if verbose:
        print(f"{pair}: {len(X)} pairs -> {out_dir} | labels {counts}")
        print(comp.round(2).to_string())
    return {"out_dir": out_dir, "comparison": comp, "counts": counts}
