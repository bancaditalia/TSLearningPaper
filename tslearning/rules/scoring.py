"""
Rule scoring — R1.1 (temporal confidence) and R1.2 (train vs test on frozen constants).

This is the step that turns an emitted rule into a result.

────────────────────────────────────────────────────────────────────────────────
WHAT CONFIDENCE MEANS HERE
────────────────────────────────────────────────────────────────────────────────

Split every rule into a CONDITION (market-state atoms — exact, computed from data) and a
CONCLUSION (the regime/lead atom — carries labelling noise). The rule claims "whenever the
condition holds, the conclusion holds too", so:

    confidence = |days where condition AND conclusion hold| / |days where condition holds|

Only condition-days count. If the stress condition holds on 300 of 2500 days and the
conclusion also holds on 240 of those, confidence is 0.80 — the other 2200 days are
irrelevant to this rule, and including them would let a rule look good simply by being
vacuously true most of the time.

Three rule shapes need three scorers:

    conditional  T1-T4  condition atoms + conclusion within ⟨w⟩
    persistence  T5     fraction of regime episodes lasting at least ⟨d⟩
    transition   T6     fraction of matched transitions satisfying ⟨d_from⟩, ⟨a⟩, ⟨b⟩

────────────────────────────────────────────────────────────────────────────────
WHY TRAIN vs TEST IS THE ONLY NUMBER THAT COUNTS
────────────────────────────────────────────────────────────────────────────────

The constants (w, d, a, b) were FITTED on the training period, so training confidence is
flattering by construction — the 25th-percentile rule for ⟨d⟩, for instance, pins
in-sample confidence at ~0.75 whatever the data does. Freeze the rule and re-evaluate on
unseen years: test ≈ train means the rule generalises and is a finding; test collapsing
means either an artifact or a genuine structural change (which is R2's business).

`lift` is reported alongside: confidence minus the conclusion's unconditional rate over
the same period. A confidence of 0.8 is unimpressive if the conclusion holds 78% of all
days anyway.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tslearning.config import CFG
from tslearning.rules.primitives import PairFacts, coalesce, days, overlap, satisfied, shift

# below this, confidence is computed on nothing (§3.3). In ticks; config rules.
MIN_CONDITION_DAYS = int((CFG.get("rules") or {}).get("min_condition_days", 30))


# ══════════════════════════════════════════════════════════════════════════════
#  Rebuilding a rule's parts against a (possibly different) set of facts
# ══════════════════════════════════════════════════════════════════════════════

def _atom(facts: PairFacts, spec) -> list:
    """(pred, role) -> intervals.

    role   'x' / 'y'   a unary predicate on that side of the pair
           'xy'        a pair predicate on both sides
           'xc' / 'yc' the static link market(X, C) / market(Y, C)
           'cx' / 'cy' a predicate on the ECONOMY of X / Y — the T7 case:
                       market(X, C), inflationHigh(C). C is resolved through market;
                       a series with no economy gives no intervals.
           'xx' / 'yy' a pair predicate on one series and itself (LLM rules) — no facts
           'macro'     a predicate with no link to the pair (a ground constant, e.g. an
                       LLM writing inflationHigh("us")). No key filter.
    """
    pred, role = spec[0], spec[1]
    if role == "x":
        return facts.iv(pred, facts.x)
    if role == "y":
        return facts.iv(pred, facts.y)
    if role in ("xc", "yc"):
        return facts.iv(pred, facts.x if role == "xc" else facts.y)
    if role in ("cx", "cy"):
        c = facts.market_of(facts.x if role == "cx" else facts.y)
        return facts.iv(pred, c) if c else []
    if role == "macro":
        return facts.iv(pred)
    if role in ("xx", "yy"):
        return []        # a pair predicate on a series and itself: no such facts exist
    return facts.iv(pred, facts.x, facts.y)


def _lag_atom(facts: PairFacts, spec, lag: int) -> list:
    if len(spec) > 1 and spec[1] in ("xx", "yy"):
        return []        # a series leading itself: no such facts exist
    return facts.lead_iv(spec[0], lag, facts.x, facts.y)


def _resolve_atoms(parts, facts, constants, side: str) -> list:
    """Build interval list for one side of a conditional rule.

    Handles single atoms, multi-atom overlaps (`conclusion_multi`), and lagged
    atoms — with the lag applied to whichever side `parts["lag_side"]` names
    (default: "conclusion", for forward templates; "condition" for T4b).
    """
    lag = constants.get("k")
    lag_side = parts.get("lag_side", "conclusion")  # which side the lag belongs to
    apply_lag = lag is not None and lag_side == side

    # Multi-atom overlap (T1b, T3b backward conclusions)
    multi_key = f"{side}_multi"
    if multi_key in parts:
        atoms = [_lag_atom(facts, s, lag) if apply_lag else _atom(facts, s)
                 for s in parts[multi_key]]
        return overlap(*atoms) if len(atoms) > 1 else atoms[0]

    # Single atom — list (condition) or tuple (conclusion)
    spec = parts[side]
    if isinstance(spec, list):
        if apply_lag and len(spec) == 1:
            return _lag_atom(facts, spec[0], lag)
        atoms = [_lag_atom(facts, s, lag) if apply_lag else _atom(facts, s)
                 for s in spec]
        return overlap(*atoms) if len(atoms) > 1 else atoms[0]
    # tuple
    if apply_lag:
        return _lag_atom(facts, spec, lag)
    return _atom(facts, spec)


def rebuild(parts: dict, facts: PairFacts, constants: dict):
    """Recompute (condition, conclusion) interval lists for these facts.

    The conclusion is returned UNSHIFTED; the caller applies the temporal operator, so
    the same function serves scoring at any w.

    Supports:
      - multi-atom conditions/conclusions (overlap of several predicates)
      - lag on either side (forward T4: lag on conclusion; backward T4b: lag on condition)
    """
    kind = parts["kind"]
    if kind == "conditional":
        condition = _resolve_atoms(parts, facts, constants, "condition")
        conclusion = _resolve_atoms(parts, facts, constants, "conclusion")
        return condition, conclusion
    if kind == "persistence":
        ivs = _atom(facts, parts["pred"])
        return ivs, ivs
    if kind == "transition":
        return _atom(facts, parts["from"]), _atom(facts, parts["to"])
    raise ValueError(f"unknown parts kind: {kind!r}")


# ══════════════════════════════════════════════════════════════════════════════
#  The three scorers
# ══════════════════════════════════════════════════════════════════════════════

def score_conditional(condition, conclusion, w: int, direction: str,
                      T0: int, T1: int, w_lo: int = 0, op: str = "diamond") -> dict:
    """Temporal confidence for T1-T4 (and LLM rules). `op` is the conclusion's operator:
    "diamond" (<±>, at some point in the window) or "box" ([±], on every day of it)."""
    cond_days = days(coalesce(condition))
    span = T1 - T0
    if cond_days == 0:
        return {"confidence": None, "condition_days": 0, "hit_days": 0,
                "baseline": None, "lift": None, "eligible": False,
                "reason": "no condition days"}
    widened = satisfied(conclusion, w, direction, w_lo, op)
    hit = days(overlap(coalesce(condition), widened))
    conf = hit / cond_days
    clipped = coalesce([(max(s, T0), min(e, T1)) for s, e in widened
                        if min(e, T1) > max(s, T0)])
    base = days(clipped) / span if span else None
    return {"confidence": conf, "condition_days": cond_days, "hit_days": hit,
            "baseline": base, "lift": (conf - base) if base is not None else None,
            "eligible": cond_days >= MIN_CONDITION_DAYS,
            "reason": None if cond_days >= MIN_CONDITION_DAYS
                      else f"only {cond_days} condition days (< {MIN_CONDITION_DAYS})"}


def _block_draws(per_day, block: int, n_boot: int, rng) -> list:
    """Moving-block bootstrap over a 0/1 per-day hit series (Künsch 1989).

    Used when there are too few EPISODES to resample. Blocks of `block` consecutive days
    are drawn with replacement, which preserves dependence within a window while still
    producing variation. `block` should be at least the window length: two days closer
    together than that share most of their underlying observations.

    Weaker than episode resampling, and it must be labelled as such — days inside one
    drawdown are not independent replications of it, they are the same event sampled
    repeatedly. It gives an interval; it does not conjure a second event.
    """
    n = len(per_day)
    if n < block * 2:
        return []
    starts = n - block + 1
    k = max(1, n // block)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, starts, k)
        s = sum(per_day[i:i + block].sum() for i in idx)
        out.append(s / (k * block))
    return out


def bootstrap_lift(rule, facts: PairFacts, n_boot: int = 2000, seed: int = 42,
                   alpha: float = 0.05, min_episodes: int = 5,
                   block: "int | None" = None) -> dict:
    """Bootstrap confidence interval on a conditional rule's LIFT.

    ────────────────────────────────────────────────────────────────────────────
    WHAT IS RESAMPLED, AND WHY IT MATTERS MORE HERE THAN USUAL
    ────────────────────────────────────────────────────────────────────────────

    **Condition EPISODES, not days.** A stress episode lasting 60 days is one event
    observed 60 times, not 60 independent observations. Resampling days would treat it as
    the latter and produce an interval several times too narrow — which is exactly the
    error that makes a small-sample finding look solid.

    That distinction is not academic here: T1 on sp500-gold rests on **five episodes**.
    The honest sample size is 5, and no amount of condition-days changes that.

    The BASELINE is held fixed rather than resampled. It is estimated over the whole
    scored period (hundreds of days), so its sampling error is small next to the
    conditional estimate's; the uncertainty being reported is the uncertainty in the
    conditional part, which is where it lives.

    A CI that spans zero does NOT mean "the effect is zero". It means the sample cannot
    distinguish this effect from none — a different and more honest claim.
    """
    parts = rule.parts or {}
    if parts.get("kind") != "conditional":
        return {"reason": f"only conditional rules are bootstrapped, not "
                          f"{parts.get('kind')!r}"}

    condition, conclusion = rebuild(parts, facts, rule.constants)
    episodes = coalesce(condition)
    w = int(rule.constants.get("w", 0))
    direction = parts.get("direction", "+")
    if not episodes:
        return {"reason": "no condition episodes"}

    w_lo = int(rule.constants.get("w_lo", 0))
    op = parts.get("op", "diamond")
    point = score_conditional(episodes, conclusion, w, direction, facts.T0, facts.T1, w_lo, op)
    if point.get("baseline") is None:
        return {"reason": "no baseline"}
    base = point["baseline"]
    widened = satisfied(conclusion, w, direction, w_lo, op)

    rng = np.random.default_rng(seed)
    n = len(episodes)
    lifts, method = [], "episode"

    if n >= min_episodes:
        per = [(days(overlap([ep], widened)), ep[1] - ep[0]) for ep in episodes]
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            hit = sum(per[i][0] for i in idx)
            tot = sum(per[i][1] for i in idx)
            if tot:
                lifts.append(hit / tot - base)
    else:
        # Too few episodes to resample — with one episode every draw is that episode and
        # the interval collapses to a point, which would read as certainty rather than as
        # the absence of replication. Fall back to blocks of days, and SAY SO.
        method = "block"
        block = block or CFG["windows"]["financial"]["length"]
        hit_days = set()
        for s, e in overlap(coalesce(episodes), widened):
            hit_days.update(range(s, e))
        per_day = np.array([[1.0 if d in hit_days else 0.0
                             for d in range(s, e)] for s, e in episodes][0]
                           if n == 1 else
                           [1.0 if d in hit_days else 0.0
                            for s, e in episodes for d in range(s, e)])
        lifts = [c - base for c in _block_draws(per_day, block, n_boot, rng)]

    if not lifts:
        return {"reason": f"cannot resample: {n} episode(s), "
                          f"{point['condition_days']} condition days — too few for either "
                          f"an episode or a block bootstrap"}

    lo, hi = np.percentile(lifts, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # A ZERO-WIDTH interval is the bootstrap failing, not succeeding. When the confidence
    # is exactly 0 or exactly 1, every resample reproduces it and the interval collapses
    # to the point estimate — which then prints as "excludes zero" and reads like an
    # extremely precise result. It is the opposite: the estimator is pinned at a boundary
    # and the bootstrap has no information about how uncertain that is.
    degenerate = (hi - lo) < 1e-9
    conf = point["confidence"]
    at_boundary = conf is not None and (conf <= 1e-9 or conf >= 1 - 1e-9)

    return {"lift": point["lift"], "ci_lo": float(lo), "ci_hi": float(hi),
            "se": float(np.std(lifts)), "n_episodes": n, "method": method,
            "block": block if method == "block" else None,
            "condition_days": point["condition_days"], "baseline": base,
            "spans_zero": bool(lo <= 0 <= hi), "n_boot": n_boot,
            "degenerate": bool(degenerate),
            "level": f"{int((1 - alpha) * 100)}%",
            "caveat": ("; ".join(c for c in (
                (f"only {n} condition episode(s), so the interval resamples {block}-day "
                 f"BLOCKS within them — blocks are not independent events"
                 if method == "block" else None),
                (f"confidence is exactly {conf:.0f}, so every resample reproduces it and "
                 f"the interval has ZERO WIDTH — this is the bootstrap failing at a "
                 f"boundary, not a precise estimate" if degenerate and at_boundary else
                 "the interval has zero width; treat it as uninformative"
                 if degenerate else None),
            ) if c) or None),
            "reason": None}


def persistence_head(ivs, d: int) -> list:
    """The days on which a persistence rule's head fires, i.e. where the regime holds
    now AND for d more days.

    `[+][0,d] P` at t requires P throughout [t, t+d] inclusive. An interval [s, e) covers
    t = s..e-1, so the rule needs t + d <= e - 1, giving the head interval [s, e-d).
    Verified against the engine: on sp500_gold with d=50 both produce
    [(1760,1773), (2063,2066)].
    """
    return [(s, e - d) for s, e in ivs if e - d > s]


def score_persistence(ivs, d: int) -> dict:
    """T5. Reports BOTH measures, with days as `confidence`.

    days     |head| / |regime days|, where the head is where the regime holds now and for
             d more days. This is what the rule literally asserts, and it is what the
             engine computes — so it is the number used for confidence and for the
             python-vs-vadalog comparison.
    episodes fraction of episodes lasting at least d. More interpretable ("how many hedges
             lasted long enough") but systematically higher, because an episode counts as a
             hit even if only its first day satisfies the rule. Kept as a diagnostic.

    On sp500_gold, d=50: days 0.113 vs episodes 0.667 — the gap is the point. Two of three
    episodes clear d, but only 16 of 141 regime-days are far enough from the end of their
    episode for the rule to hold.
    """
    lens = [e - s for s, e in ivs]
    if not lens:
        return {"confidence": None, "n_episodes": 0, "eligible": False,
                "reason": "no episodes"}

    head = persistence_head(ivs, d)
    head_days = days(head)
    regime_days = days(coalesce(list(ivs)))
    hits = sum(1 for L in lens if L >= d)

    return {"confidence": (head_days / regime_days) if regime_days else None,
            "metric": "days",
            "head_days": head_days, "regime_days": regime_days,
            "confidence_episodes": hits / len(lens),
            "n_episodes": len(lens), "hit_episodes": hits,
            "median_dwell": int(np.percentile(lens, 50)),
            "min_dwell": int(min(lens)), "max_dwell": int(max(lens)),
            "eligible": len(lens) >= 3,
            "reason": None if len(lens) >= 3 else f"only {len(lens)} episodes"}


def score_transition(from_ivs, to_ivs, d_from: int, a: int, b: int,
                     max_gap: float = float("inf")) -> dict:
    """T6: of the `from` episodes long enough to trigger the rule, what fraction are
    followed by a `to` episode covering all of [t+a, t+b]?"""
    to_sorted = sorted(to_ivs)
    triggered = hits = 0
    for fs, fe in sorted(from_ivs):
        if fe - fs < d_from:                     # body's [-] part not satisfied
            continue
        triggered += 1
        nxt = next(((ts, te) for ts, te in to_sorted if ts >= fe), None)
        if nxt is None or nxt[0] - fe > max_gap:
            continue
        if nxt[0] - fe <= a and nxt[1] - fe >= b:   # covers the whole [a,b] window
            hits += 1
    if triggered == 0:
        return {"confidence": None, "n_episodes": 0, "eligible": False,
                "reason": f"no episode reaches d_from={d_from}"}
    return {"confidence": hits / triggered, "n_episodes": triggered, "hit_episodes": hits,
            "eligible": triggered >= 3,
            "reason": None if triggered >= 3 else f"only {triggered} triggering episodes"}


def score_rule(rule, facts: PairFacts, max_gap: float = float("inf")) -> dict:
    """Score one already-fitted rule against one set of facts (any period)."""
    if not rule.accepted or not rule.parts:
        return {"confidence": None, "eligible": False, "reason": "rule not accepted"}
    c = rule.constants
    cond, concl = rebuild(rule.parts, facts, c)
    kind = rule.parts["kind"]
    if kind == "conditional":
        return score_conditional(cond, concl, c["w"], rule.parts.get("direction", "+"),
                                 facts.T0, facts.T1, int(c.get("w_lo", 0)),
                                 rule.parts.get("op", "diamond"))
    if kind == "persistence":
        return score_persistence(cond, c["d"])
    return score_transition(cond, concl, c["d_neg"], c["a"], c["b"], max_gap)


# ══════════════════════════════════════════════════════════════════════════════
#  Period splitting — the train/test machinery
# ══════════════════════════════════════════════════════════════════════════════

def day_range_for_dates(facts: PairFacts, lo, hi) -> "tuple[int, int]":
    """Convert a calendar range to the day-index range on this pair's timeline."""
    if facts.dates is None:
        raise ValueError(f"{facts.pair} has no dates map — cannot split by calendar")
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    days_in = [d for d, ts in facts.dates.items() if lo <= pd.Timestamp(ts) <= hi]
    if not days_in:
        return (0, 0)
    return (min(days_in), max(days_in) + 1)


def clip_facts(facts: PairFacts, day_lo: int, day_hi: int) -> PairFacts:
    """A new PairFacts holding only the part of every interval inside [day_lo, day_hi)."""
    out = {}
    for pred, ivs in facts.intervals.items():
        out[pred] = [(k, max(s, day_lo), min(e, day_hi)) for k, s, e in ivs
                     if min(e, day_hi) > max(s, day_lo)]
    clipped = PairFacts(facts.pair, facts.x, facts.y, out, dates=facts.dates)
    # Keep the requested window as the span, so baselines use the period's length rather
    # than the extent of whatever intervals happened to survive the clip.
    clipped.T0, clipped.T1 = day_lo, day_hi
    return clipped


def split_facts(facts: PairFacts, train=None, test=None) -> dict:
    """Split into {'train': PairFacts, 'test': PairFacts} by the config's calendar split."""
    train = train or CFG["split"]["train"]
    test = test or CFG["split"]["test"]
    tr = day_range_for_dates(facts, *train)
    te = day_range_for_dates(facts, *test)
    return {"train": clip_facts(facts, *tr), "test": clip_facts(facts, *te),
            "train_days": tr, "test_days": te}


def score_train_test(rule, facts: PairFacts, refit=None, train=None, test=None,
                     max_gap: float = float("inf")) -> dict:
    """Score a rule on train and test with FROZEN constants.

    `refit` (optional callable(facts) -> Rule) re-fits the constants on the training
    period only, which is the methodologically correct version: constants fitted on the
    full history have already seen the test years. When omitted, the rule's existing
    constants are used as-is and that limitation is recorded.
    """
    parts = split_facts(facts, train, test)
    scored = rule
    leakage = True
    if refit is not None:
        refitted = refit(parts["train"])
        if refitted is not None and refitted.accepted:
            scored = refitted
            leakage = False

    return {"rule": scored,
            "train": score_rule(scored, parts["train"], max_gap),
            "test": score_rule(scored, parts["test"], max_gap),
            "fitted_on_train_only": not leakage,
            "train_day_range": parts["train_days"],
            "test_day_range": parts["test_days"]}


# ══════════════════════════════════════════════════════════════════════════════
#  Per-slice scoring — the entry point for R2 (structural breaks)
# ══════════════════════════════════════════════════════════════════════════════

def score_by_slice(rule, facts: PairFacts, freq: str = "YS",
                   max_gap: float = float("inf")) -> pd.DataFrame:
    """Confidence per calendar slice, with constants frozen. A rule scoring
    0.85, 0.82, 0.79, 0.31, 0.28 across five years stopped holding in the fourth —
    that date is a candidate structural break (R2)."""
    if facts.dates is None:
        raise ValueError(f"{facts.pair} has no dates map")
    # Indexed BY DATE, holding day numbers — pd.Grouper needs a DatetimeIndex.
    ts = pd.Series(data=list(facts.dates.keys()),
                   index=pd.to_datetime(list(facts.dates.values()))).sort_index()
    rows = []
    for period, grp in ts.groupby(pd.Grouper(freq=freq)):
        if grp.empty:
            continue
        sc = score_rule(rule, clip_facts(facts, int(grp.min()), int(grp.max()) + 1),
                        max_gap)
        rows.append({"slice": str(period.date()), "n_days": len(grp),
                     **{k: v for k, v in sc.items() if not isinstance(v, dict)}})
    return pd.DataFrame(rows)
