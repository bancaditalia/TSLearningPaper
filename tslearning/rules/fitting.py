"""
Constant fitting — how each template's ⟨w⟩, ⟨d⟩, ⟨k⟩ are chosen from the data.



Two families:

  choose_w   for "condition now, conclusion within w" templates (T1-T4). Picks w by the
             ONE-STANDARD-ERROR RULE over a stress-vs-baseline gap.
  choose_d   for persistence/transition templates (T5, T6). Percentiles of observed
             dwell times.

Why the 1-SE rule for w. The gap is monotone in w by construction — widening w only
weakens an existential condition, so more of the condition-window qualifies and coverage
can only rise. "Pick the biggest gap" therefore always returns the largest w in the grid:
the grid limit selects w, not the data. The 1-SE rule instead takes the SMALLEST w whose
gap is statistically indistinguishable from the best, so extra days of slack have to beat
the noise to be adopted. Standard model-selection convention (Breiman et al. 1984;
Hastie et al., ESL §7.10).

Why the bootstrap resamples EPISODES, not days. One crash is ~30 consecutive stress days
all telling you the same thing. Resampling days would pretend you have hundreds of
independent observations instead of ~12 episodes, shrink the standard error, and hand you
back the boundary solution the rule exists to avoid.
"""
from __future__ import annotations

import numpy as np

from tslearning.config import CFG
from tslearning.rules.primitives import coalesce, days, overlap, shift

# The w values tried when fitting a window, min_w..max_w (config rules.min_w / max_w, in
# ticks). min_w defaults to 0 when the config has no key (the setting before 2026-10-26).
# Fixed per process: the run is chosen before tslearning is imported.
MIN_W = int((CFG.get("rules") or {}).get("min_w", 0))
W_GRID = range(MIN_W, int((CFG.get("rules") or {}).get("max_w", 25)) + 1)


# ══════════════════════════════════════════════════════════════════════════════
#  w — response window
# ══════════════════════════════════════════════════════════════════════════════

def gap_for(episodes, conclusion, w: int, direction: str, T0: int, T1: int) -> float:
    """Percentage-point gap between the conclusion's rate during `episodes` and its
    rate over the whole timeline.

    episodes   : condition intervals (exact market state)
    conclusion : conclusion intervals (regime / lead label)
    """
    span = T1 - T0
    n_cond = days(episodes)
    if n_cond == 0 or span == 0:
        return 0.0
    widened = shift(conclusion, w, direction)
    c1 = days(overlap(episodes, widened)) / n_cond * 100.0
    clipped = coalesce([(max(s, T0), min(e, T1)) for s, e in widened
                        if min(e, T1) > max(s, T0)])
    c2 = days(clipped) / span * 100.0
    return c1 - c2


def choose_w(episodes, conclusion, direction: str = "+", w_grid=W_GRID,
             T0: int = 0, T1: int = 0, n_boot: int = 500, seed: int = 42) -> dict:
    """Fit w by the 1-SE rule. Returns a dict with w, se, gap curve and diagnostics.

    `w` is None when there are no episodes to measure or the gap never turns positive —
    that is a legitimate outcome meaning "this pair shows no such effect", and callers
    should report it rather than forcing a constant.
    """
    episodes = list(episodes)
    if not episodes:
        return {"w": None, "se": None, "gap": {}, "best_gap": None,
                "n_episodes": 0, "reason": "no condition episodes"}

    gap = {w: gap_for(episodes, conclusion, w, direction, T0, T1) for w in w_grid}
    best_w = max(gap, key=gap.get)
    best_gap = gap[best_w]

    if best_gap <= 0:
        return {"w": None, "se": None, "gap": gap, "best_gap": best_gap,
                "n_episodes": len(episodes),
                "reason": f"no signal (best gap {best_gap:.2f}pp <= 0)"}

    rng = np.random.default_rng(seed)
    boots = [gap_for([episodes[i] for i in rng.integers(0, len(episodes), len(episodes))],
                     conclusion, best_w, direction, T0, T1)
             for _ in range(n_boot)]
    se = float(np.std(boots))

    w_star = min(w for w in sorted(gap) if gap[w] >= best_gap - se)
    return {"w": int(w_star), "se": se, "gap": gap, "best_gap": best_gap,
            "gap_at_w": gap[w_star], "n_episodes": len(episodes),
            "condition_days": days(episodes), "reason": None}


# ══════════════════════════════════════════════════════════════════════════════
#  d — persistence / transition
# ══════════════════════════════════════════════════════════════════════════════

def choose_lag(condition, conclusion_by_lag: dict, direction: str = "+",
               w_grid=W_GRID, T0: int = 0, T1: int = 0,
               n_boot: int = 500, seed: int = 42) -> dict:
    """Fit the lag k AND the window w together, by the 1-SE rule on both.

    `conclusion_by_lag` maps k -> the conclusion intervals at that lag.

    Why not argmax over k. Picking the lag with the largest gap is the same
    non-identifying criterion the 1-SE rule exists to avoid for w: with a handful of
    candidates and a noisy gap it tends to land on whichever lag happens to look best,
    and a boundary lag (the largest in the grid) is the tell. So: score every lag, take
    the best, bootstrap ITS standard error, and then keep the SMALLEST lag whose gap is
    within one SE of that. A shorter lead is the more parsimonious claim — and
    economically the more plausible one, since price discovery is fast.
    """
    per_lag = {}
    for k, conclusion in conclusion_by_lag.items():
        if not conclusion:
            continue
        fit = choose_w(condition, conclusion, direction, w_grid, T0, T1,
                       n_boot=n_boot, seed=seed)
        if fit["w"] is not None:
            per_lag[k] = fit

    if not per_lag:
        return {"k": None, "fit": None, "per_lag": {},
                "reason": "no lag yields a positive gap"}

    best_k = max(per_lag, key=lambda k: per_lag[k]["gap_at_w"])
    se = per_lag[best_k]["se"] or 0.0
    bar = per_lag[best_k]["gap_at_w"] - se
    k_star = min(k for k in sorted(per_lag) if per_lag[k]["gap_at_w"] >= bar)

    return {"k": int(k_star), "fit": per_lag[k_star], "per_lag": per_lag,
            "k_argmax": int(best_k), "se_at_best": se,
            "gaps": {k: per_lag[k]["gap_at_w"] for k in sorted(per_lag)},
            "reason": None}


def choose_d(ivs, pct: int = 25) -> dict:
    """d = pct-th percentile of dwell times (T5 hedge horizon).

    The 25th percentile is a conservative floor: "in 75% of episodes the regime lasted at
    least d", which also fixes in-sample confidence at ~0.75 by construction — the number
    you then re-measure out of sample.

    NOTE every interval inherits +seq_len from the window length, so d has a floor of one
    window. Report `d - seq_len` as the excess, or state the caveat.
    """
    dwell = [e - s for s, e in ivs]
    if not dwell:
        return {"d": None, "n": 0, "reason": "no intervals"}
    return {"d": int(np.percentile(dwell, pct)), "n": len(dwell),
            "dwell_min": int(min(dwell)), "dwell_max": int(max(dwell)),
            "dwell_median": int(np.percentile(dwell, 50)), "reason": None}


def choose_d_transition(from_ivs, to_ivs, pct: int = 25,
                        max_gap: float = float("inf")) -> dict:
    """Constants for a transition template (T6 regime reversal).

    Pairs each `from` interval with the next `to` interval starting after it, then fits,
    measured from t = end of the `from` regime:
        d_from : how long `from` held before t      -> [-](0, d_from]
        a, b   : window in which `to` holds after t -> [+][a, b]

    `a` and `b` take opposite percentile tails on purpose: `a` from the upper tail of the
    gaps, so the window opens after the `to` regime has begun in most transitions; `b`
    from the lower tail of the ends, so it closes before that regime has died in most.
    If a >= b no common window satisfies enough transitions — return nothing rather than a
    rule that cannot hold.

    max_gap constrains the rule, not just the sample: it caps how large `a` can get.
    Use float("inf") for no constraint (a "reversal" 18 months later is not a reversal).
    """
    to_sorted = sorted(to_ivs)
    froms, gaps, ends = [], [], []
    for fs, fe in sorted(from_ivs):
        nxt = next(((ts, te) for ts, te in to_sorted if ts >= fe), None)
        if nxt is None or nxt[0] - fe > max_gap:
            continue
        froms.append(fe - fs)
        gaps.append(nxt[0] - fe)
        ends.append(nxt[1] - fe)

    if not froms:
        return {"d_from": None, "a": None, "b": None, "n": 0,
                "reason": "no matched transitions"}

    d_from = int(np.percentile(froms, pct))
    a = int(np.percentile(gaps, 100 - pct))
    b = int(np.percentile(ends, pct))
    if a >= b:
        return {"d_from": d_from, "a": a, "b": b, "n": len(froms),
                "gap_median": int(np.percentile(gaps, 50)),
                "reason": f"no common window (a={a} >= b={b})"}
    return {"d_from": d_from, "a": a, "b": b, "n": len(froms),
            "gap_median": int(np.percentile(gaps, 50)), "reason": None}
