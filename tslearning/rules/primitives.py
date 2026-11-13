"""
Interval algebra — the primitives every rule template is built from.



Everything here operates on **interval facts**: a predicate maps to a list of
`(key, start, end)` with `[start, end)` half-open, in day indices on the real timeline.
The key is `(a, b)` for pair predicates, `a` for unary ones, and `(a, b, lag)` for
lead-lag ones.
"""
from __future__ import annotations

from collections import Counter, defaultdict


# ══════════════════════════════════════════════════════════════════════════════
#  Basic interval operations
# ══════════════════════════════════════════════════════════════════════════════

def coalesce(ivs):
    """Merge overlapping/touching [s,e) intervals."""
    ivs = sorted(ivs)
    if not ivs:
        return []
    out = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def overlap(*lists):
    """Intersection of several [(s,e)] lists — where ALL hold at once."""
    if not lists:
        return []
    cur = lists[0]
    for nxt in lists[1:]:
        cur = sorted((max(a, c), min(b, d))
                     for a, b in cur for c, d in nxt
                     if max(a, c) < min(b, d))
    return cur


def days(ivs) -> int:
    """Total covered days (assumes already coalesced, else double-counts)."""
    return sum(e - s for s, e in ivs)


def shift(ivs, w: int, direction: str = "+", w_lo: int = 0):
    """Widen intervals to express a temporal operator, then coalesce.

    The operator looks one way, the satisfying interval extends the other:
      "+"  <+>[a,w]  P holds between a and w days AHEAD   -> t in [s-w, e-a)
      "-"  <->[a,w]  P held between a and w days BEHIND   -> t in [s+a, e+w)
    The templates always use a = 0 (`w_lo`), which gives (s-w, e) / (s, e+w).
    """
    if direction == "+":
        return coalesce([(s - w, e - w_lo) for s, e in ivs if e - w_lo > s - w])
    if direction == "-":
        return coalesce([(s + w_lo, e + w) for s, e in ivs if e + w > s + w_lo])
    raise ValueError(f"direction must be '+' or '-', got {direction!r}")


def box_shift(ivs, w: int, direction: str = "+", w_lo: int = 0):
    """The days on which a BOX operator holds — the shrinking counterpart of `shift`.

      "+"  [+][a,w] P  P holds on EVERY day from a to w days ahead   -> t in [s-a, e-w)
      "-"  [-][a,w] P  P held on EVERY day from a to w days behind   -> t in [s+w, e+a)
    With a = 0, "+" gives [s, e-w): the same as `scoring.persistence_head` (T5), which is
    engine-verified. P's intervals are coalesced first, so touching runs count as one.
    """
    ivs = coalesce(ivs)
    if direction == "+":
        return [(s - w_lo, e - w) for s, e in ivs if e - w > s - w_lo]
    if direction == "-":
        return [(s + w, e + w_lo) for s, e in ivs if e + w_lo > s + w]
    raise ValueError(f"direction must be '+' or '-', got {direction!r}")


def satisfied(ivs, w: int, direction: str = "+", w_lo: int = 0, op: str = "diamond"):
    """Days on which `<±>[w_lo,w] P` (op="diamond") or `[±][w_lo,w] P` (op="box") holds."""
    if op == "box":
        return box_shift(ivs, w, direction, w_lo)
    return shift(ivs, w, direction, w_lo)


# ══════════════════════════════════════════════════════════════════════════════
#  Mutual exclusivity
# ══════════════════════════════════════════════════════════════════════════════

def make_exclusive(predicate_intervals: dict, labels, mode: str = "vote",
                   priority: "dict | None" = None) -> dict:
    """Force `labels` to be mutually exclusive: one winner per timepoint, then coalesce.

    Coalescing each predicate independently leaves the regime classes overlapping each
    other (overlapping windows disagree). Each covering window casts one vote — the
    real-pair path has one hard label per window, so there is no probability to sort on.

      mode="vote"      -> most votes wins, ties broken by `priority`
      mode="exclusive" -> keep a timepoint only if ONE label covers it (leaves gaps)

    Predicates not in `labels` pass through untouched.
    """
    priority = priority or {}
    votes, keys = defaultdict(Counter), {}
    for lab in labels:
        for key, s, e in predicate_intervals.get(lab, []):
            keys[lab] = key
            for t in range(int(s), int(e)):
                votes[t][lab] += 1

    winners = {}
    for t, c in votes.items():
        if mode == "exclusive":
            if len(c) == 1:
                winners[t] = next(iter(c))
        else:
            winners[t] = max(c, key=lambda l: (c[l], priority.get(l, 0)))

    out = {p: list(ivs) for p, ivs in predicate_intervals.items() if p not in labels}
    runs = defaultdict(list)
    for t in sorted(winners):
        lab = winners[t]
        if runs[lab] and runs[lab][-1][2] == t:
            runs[lab][-1] = (runs[lab][-1][0], runs[lab][-1][1], t + 1)
        else:
            runs[lab].append((keys[lab], t, t + 1))
    for lab in labels:
        out[lab] = runs.get(lab, [])
    return out


def count_overlaps(predicate_intervals: dict, labels) -> int:
    """How many adjacent interval pairs overlap across `labels` — 0 after make_exclusive."""
    flat = sorted((s, e, l) for l in labels for _k, s, e in predicate_intervals.get(l, []))
    return sum(1 for i in range(len(flat) - 1) if flat[i][1] > flat[i + 1][0])


# ══════════════════════════════════════════════════════════════════════════════
#  Query surface — what templates actually call
# ══════════════════════════════════════════════════════════════════════════════

class PairFacts:
    """The interval facts for ONE ground pair, with the query helpers templates use.

    `x` and `y` are the lowercase series names as they appear in the interval keys
    (e.g. "sp500", "gold"). `timeline` is (T0, T1) — the full span, used as the
    denominator for baseline rates.
    """

    def __init__(self, pair: str, x: str, y: str, predicate_intervals: dict,
                 dates=None):
        self.pair = pair
        self.x = x
        self.y = y
        self.intervals = predicate_intervals
        self.dates = dates                      # day -> date map, or None
        allivs = [(s, e) for p in predicate_intervals
                  for _k, s, e in predicate_intervals[p]]
        self.T0 = min((s for s, _ in allivs), default=0)
        self.T1 = max((e for _, e in allivs), default=0)

    @property
    def span(self) -> int:
        return self.T1 - self.T0

    def iv(self, pred, series=None, other=None):
        """Intervals for a predicate.
             iv("highVol", X)                  unary
             iv("isNegativeCorrelation", X)    any pair containing X
             iv("isNegativeCorrelation", X, Y) that exact pair (order-insensitive)
        """
        want = None
        if series is not None:
            want = set(series) if isinstance(series, (tuple, list, set)) else {series}
            if other is not None:
                want.add(other)
        return sorted((s, e) for key, s, e in self.intervals.get(pred, [])
                      if want is None
                      or want <= (set(key) if isinstance(key, tuple) else {key}))

    def market_of(self, series) -> "str | None":
        """The economy C with market(series, C), or None if the series has none."""
        for key, _s, _e in self.intervals.get("market", []):
            if isinstance(key, tuple) and key[0] == series:
                return key[1]
        return None

    def lead_iv(self, pred, lag, series=None, other=None):
        """Intervals for a lead-lag predicate at one lag. Keys are (a, b, lag);
        `pred` is directional — key (a,b,lag) means a leads b."""
        want = None
        if series is not None:
            want = {series, other} if other is not None else {series}
        return sorted((s, e) for key, s, e in self.intervals.get(pred, [])
                      if len(key) == 3 and key[2] == lag
                      and (want is None or want <= set(key[:2])))

    def lags_present(self, preds) -> list:
        """Which lags actually appear in the lead-lag predicates."""
        return sorted({k[2] for p in preds for k, _s, _e in self.intervals.get(p, [])
                       if len(k) == 3})

    def has(self, *preds) -> bool:
        """True if every named predicate has at least one interval."""
        return all(self.intervals.get(p) for p in preds)

    def calendar(self, ticks: int) -> "int | None":
        """Convert a tick duration to approximate calendar days via the dates map."""
        if self.dates is None or ticks is None:
            return None
        lo, hi = self.T0, min(self.T0 + ticks, max(self.dates))
        try:
            return int((self.dates[hi] - self.dates[lo]).days)
        except Exception:
            return None
