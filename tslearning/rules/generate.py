"""
Rule generation — load interval facts for pairs, run the templates, collect the rules.

Three label SOURCES, all producing the same predicate CSVs in the same per-pair layout,
so the templates are indifferent to which one was used:

    py         real_pairs_regime_py/<pair>/          exact Python labeler (141/141 vs reasoner)
    reasoner   real_pairs_regime_reasoner/<pair>/    Vadalog, recalibrated by dominance
    classifier real_pairs_regime_classifier/<pair>/  the distilled student's regime
                                                     predicates + the exact flags copied
                                                     from the py folder (infer.py)

Since 2026-09-02 the classifier source is one folder per pair like the others; the old
flat `temporal/` bundle is written only by `infer.py --legacy-bundle` and is not a rule
source any more.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from tslearning.config import CFG, PATHS
from tslearning.rules.primitives import PairFacts, count_overlaps, make_exclusive
from tslearning.rules.templates import TEMPLATES

REGIME_LABELS = list(CFG["classifier"]["softmax_group"])
FLAG_LABELS = list(CFG["classifier"]["multilabel"])
LEAD_LABELS = ["leadsPositively", "leadsNegatively", "noLead"]
# Macro STATE predicates (T7): monthly conditions mapped onto the pair's daily timeline.
# Discovered from the config rather than hardcoded, so adding another macro state is a
# config edit plus a macro_state run. See labeling/macro_state.py.
MACRO_LABELS = [p for p, k in CFG["predicates"]["kind"].items()
                if k == "unary_bin" and ("High" in p or "Low" in p
                                        or "Rising" in p or "Falling" in p
                                        or "Accelerating" in p or "Decelerating" in p)]
ALL_PREDICATES = REGIME_LABELS + FLAG_LABELS + LEAD_LABELS + MACRO_LABELS
PRED_KIND = CFG["predicates"]["kind"]

SOURCES = {
    "py":         "real_pairs_regime_py",
    "reasoner":   "real_pairs_regime_reasoner",
    "classifier": "real_pairs_regime_classifier",
}

# Tie-break for make_exclusive when two regime classes tie on votes at a timepoint.
#
# EMPTY ON PURPOSE. It used to demote isNoCorrelation ({pos: 2, neg: 2, none: 1}) on the
# theory that a residual class shouldn't win contested days. With window=50 and stride=1
# every day is covered by ~50 windows, so ties are the norm, and every tie went against
# "no correlation": on sp500_gold it labelled 615 of 1710 windows yet ended up with ZERO
# days after exclusivity, which silently made T3 unrunnable. A class-dependent thumb on
# the scale is not a modelling choice we can write down — the honest lever for class
# balance is the labelling dead band (--deadband), which decides the width of the
# "no correlation" band before segmentation ever sees it.
REGIME_PRIORITY: dict = {}


# Optional per-source folder override (generate_rules.py --labels-dir): lets a candidate
# classifier's labels be mined from their own folder without touching the paper's.
SOURCE_DIR_OVERRIDE: dict = {}


def source_dir(source: str) -> Path:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {sorted(SOURCES)}, got {source!r}")
    if source in SOURCE_DIR_OVERRIDE:
        return Path(SOURCE_DIR_OVERRIDE[source])
    return PATHS["data_processed"] / SOURCES[source]


def discover_pairs(source: str) -> list:
    """Every pair that has labels under this source."""
    root = source_dir(source)
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir() and any(d.glob("*.csv")))


def _series_of(name) -> str:
    name = str(name)
    for sep in ("_real_", "_synth_"):
        if sep in name:
            name = name.split(sep)[0]
    return "bitcoin" if name == "btc" else name


def _coalesce_presence(df: pd.DataFrame, kind: str) -> list:
    """Merge overlapping/adjacent [Start,End) per key.
       pair_bin -> (a,b) · lag_bin -> (a,b,lag) · unary -> series."""
    if df is None or df.empty:
        return []
    bykey = defaultdict(list)
    for _, r in df.iterrows():
        if kind == "pair_bin":
            key = (_series_of(r["X"]), _series_of(r["Y"]))
        elif kind == "lag_bin":
            key = (_series_of(r["X"]), _series_of(r["Y"]), int(r["Lag"]))
        else:
            key = _series_of(r["X"])
        bykey[key].append((int(r["Start"]), int(r["End"])))

    out = []
    for key, ivs in bykey.items():
        ivs.sort()
        cs, ce = ivs[0]
        for s, e in ivs[1:]:
            if s <= ce:
                ce = max(ce, e)
            else:
                out.append((key, cs, ce))
                cs, ce = s, e
        out.append((key, cs, ce))
    return out


def _window_presence(df: pd.DataFrame, kind: str) -> list:
    """One (key, start, end) per row, NOT merged: each window keeps its own vote in
    make_exclusive."""
    if df is None or df.empty:
        return []
    out = []
    for _, r in df.iterrows():
        if kind == "pair_bin":
            key = (_series_of(r["X"]), _series_of(r["Y"]))
        elif kind == "lag_bin":
            key = (_series_of(r["X"]), _series_of(r["Y"]), int(r["Lag"]))
        else:
            key = _series_of(r["X"])
        out.append((key, int(r["Start"]), int(r["End"])))
    return out


def normalize_pair(spec: str, source: str = "py") -> str:
    """Accept either the on-disk slug ('us_inflation_us_unemployment') or a pair spec
    ('US_Inflation:US_Unemployment', 'SP500:Gold') and return the slug.

    Labelling writes folders named by slug, so a colon spec on the command line would
    otherwise look for a directory that doesn't exist and silently report every predicate
    as missing.
    """
    root = source_dir(source)
    if (root / spec).exists():
        return spec
    try:
        from tslearning.datasets.pairs import resolve_pair
        return resolve_pair(spec)[2]
    except Exception:
        return spec


_CHECKED_STALE: set = set()      # pairs already checked this process (warn once)


def load_pair_facts(pair: str, source: str = "py", exclusive: bool = True,
                    mode: str = "vote", verbose: bool = False) -> PairFacts:
    """Read one pair's predicate CSVs and coalesce them into interval facts.

    `exclusive` restores mutual exclusivity of the regime classes — without it the three
    classes overlap each other, because each predicate is coalesced independently.
    """
    pair = normalize_pair(pair, source)
    root = source_dir(source)
    d = root / pair
    if source == "classifier" and pair not in _CHECKED_STALE:
        _CHECKED_STALE.add(pair)
        from tslearning.distill.inference import warn_if_stale
        warn_if_stale(pair)
    if not d.exists():
        hint = ("python scripts/infer.py -p " + pair if source == "classifier" else
                f"python scripts/prep_real_pairs.py -p {pair} --engine "
                + ("vadalog" if source == "reasoner" else "python"))
        raise FileNotFoundError(f"{d} does not exist — produce it first:\n    {hint}")

    intervals, raw_regime = {}, {}
    for pred in ALL_PREDICATES:
        for cand in (d / f"{pred}_{pair}.csv", d / f"{pred}.csv"):
            if cand.exists():
                df = pd.read_csv(cand)
                intervals[pred] = _coalesce_presence(df, PRED_KIND[pred])
                if pred in REGIME_LABELS:
                    raw_regime[pred] = _window_presence(df, PRED_KIND[pred])
                break
        else:
            intervals[pred] = []
            if pred in REGIME_LABELS:
                raw_regime[pred] = []

    if exclusive:
        # Vote with the WINDOWS, one vote each (fixed 2026-10-26). Voting on the
        # coalesced intervals gave each class at most one vote per day, so every day
        # covered by any positive window tied and went to isPositiveCorrelation (first
        # in the list): on simulated pairs 93% of days came out positive against 62% by
        # majority of windows.
        intervals = {**intervals, **raw_regime}
        before = count_overlaps(intervals, REGIME_LABELS)
        intervals = make_exclusive(intervals, REGIME_LABELS, mode=mode,
                                   priority=REGIME_PRIORITY)
        if verbose:
            print(f"  regime overlaps {before} -> "
                  f"{count_overlaps(intervals, REGIME_LABELS)}")

    dates = None
    for cand in (d / f"{pair}_dates.csv", root / f"{pair}_dates.csv"):
        if cand.exists():
            df = pd.read_csv(cand)
            dates = pd.to_datetime(df.set_index("day")["date"]).to_dict()
            break

    x, y = _pair_sides(pair, intervals)
    facts = PairFacts(pair, x, y, intervals, dates=dates)
    add_market_facts(facts)
    return facts


def add_market_facts(facts: PairFacts) -> None:
    """Add the static market(X, C) facts for the pair's series, from config `markets`.

    market holds over the whole timeline — it is a property of the series, not a state —
    so each fact spans [T0, T1). A series without an entry gets no fact, and a rule that
    joins through market(X, C) then has nothing to join on for it.
    """
    markets = {str(k).lower(): str(v).lower() for k, v in (CFG.get("markets") or {}).items()}
    facts.intervals["market"] = [((s, markets[s]), facts.T0, facts.T1)
                                 for s in (facts.x, facts.y) if s in markets]


def _pair_sides(pair: str, intervals: dict) -> "tuple[str, str]":
    """(x, y) as they appear in the interval keys.

    Taken from an actual pair-predicate key when one exists — the CSVs are the ground
    truth on naming (btc -> bitcoin, macro slugs with underscores). Falls back to the
    config-aware resolver, then to a naive split.
    """
    for pred in REGIME_LABELS:
        for key, _s, _e in intervals.get(pred, []):
            if isinstance(key, tuple) and len(key) >= 2:
                return key[0], key[1]
    try:
        from tslearning.datasets.pairs import resolve_pair
        a, b, _ = resolve_pair(pair)
        return a.lower(), b.lower()
    except Exception:
        a, _, b = pair.partition("_")
        return a, b


# ══════════════════════════════════════════════════════════════════════════════
#  Driver
# ══════════════════════════════════════════════════════════════════════════════

def generate_for_pair(pair: str, source: str = "py", templates=None,
                      ground: bool = False, verbose: bool = True, **template_kw) -> list:
    """Run every requested template against one pair. Returns a list of Rule."""
    names = list(templates or TEMPLATES)
    facts = load_pair_facts(pair, source, verbose=verbose)

    rules = []
    for name in names:
        spec = TEMPLATES[name]
        missing = [p for p in spec["needs"] if not facts.intervals.get(p)]
        if missing:
            from tslearning.rules.rule import Rule
            rules.append(Rule.rejected(name, facts, f"missing predicates: {missing}"))
            if verbose:
                print(f"  {name:22s} skipped — missing {missing}")
            continue
        rule = spec["fn"](facts, **template_kw)
        if rule is None:
            from tslearning.rules.rule import Rule
            rule = Rule.rejected(name, facts, "template returned nothing")
        rules.append(rule)
        if verbose:
            print(f"  {name:22s} " + (rule.render(ground) if rule.accepted
                                      else f"rejected — {rule.reason}"))
    return rules


def score_pair(pair: str, source: str = "py", templates=None, ground: bool = False,
               split: bool = True, ci: bool = False, verbose: bool = True,
               **template_kw) -> list:
    """Fit on the TRAINING period, freeze, then score on train and test (R1.1 + R1.2).

    `split=False` scores in-sample over the whole timeline — useful for a quick look, but
    an in-sample number is not a result: the constants were fitted on that same data.
    """
    from tslearning.rules import scoring
    from tslearning.rules.rule import Rule

    facts = load_pair_facts(pair, source, verbose=verbose)
    names = list(templates or TEMPLATES)
    out = []

    for name in names:
        spec = TEMPLATES[name]
        missing = [p for p in spec["needs"] if not facts.intervals.get(p)]
        if missing:
            out.append(Rule.rejected(name, facts, f"missing predicates: {missing}"))
            if verbose:
                print(f"  {name:22s} skipped — missing {missing}")
            continue

        if not split:
            rule = spec["fn"](facts, **template_kw) or Rule.rejected(
                name, facts, "template returned nothing")
            if rule.accepted:
                rule.scores["insample"] = scoring.score_rule(rule, facts)
            out.append(rule)
            if verbose:
                _print_scored(rule, ground, ("insample",))
            continue

        # Fit on train only — constants fitted on the full history have already seen
        # the test years, which is exactly the leakage train/test exists to expose.
        try:
            periods = scoring.split_facts(facts)
        except ValueError as e:
            rule = spec["fn"](facts, **template_kw) or Rule.rejected(name, facts, str(e))
            out.append(rule)
            if verbose:
                print(f"  {name:22s} no dates map — cannot split ({e})")
            continue

        rule = spec["fn"](periods["train"], **template_kw)
        if rule is None or not rule.accepted:
            out.append(rule or Rule.rejected(name, facts, "no fit on train"))
            if verbose:
                r = out[-1]
                print(f"  {name:22s} rejected on train — {r.reason}")
            continue

        rule.pair = facts.pair          # the slug, not the raw spec
        rule.scores["train"] = scoring.score_rule(rule, periods["train"])
        rule.scores["test"] = scoring.score_rule(rule, periods["test"])
        if ci and rule.accepted and (rule.parts or {}).get("kind") == "conditional":
            rule.scores["ci"] = scoring.bootstrap_lift(rule, periods["test"])
        out.append(rule)
        if verbose:
            _print_scored(rule, ground, ("train", "test"))
            if rule.scores.get("ci"):
                _print_ci(rule)
    return out


def _print_ci(rule) -> None:
    """One line for the bootstrap interval, always naming the EPISODE count.

    A lift printed without it is the number that misleads: "224 condition-days" sounds
    like a sample, and on the test period it is a single drawdown.
    """
    b = rule.scores.get("ci") or {}
    if b.get("reason"):
        print(f"      CI: {b['reason']}")
        return
    flag = ("DEGENERATE — no interval" if b.get("degenerate") else
            "SPANS ZERO" if b["spans_zero"] else "excludes zero")
    print(f"      lift {b['lift']:+.3f}   {b['level']} CI "
          f"[{b['ci_lo']:+.3f}, {b['ci_hi']:+.3f}]   {b['n_episodes']} episode(s)   "
          f"{flag}   [{b.get('method')} bootstrap]")


def _print_scored(rule, ground: bool, periods) -> None:
    if not rule.accepted:
        print(f"  {rule.template:22s} rejected — {rule.reason}")
        return
    bits = []
    for p in periods:
        sc = rule.scores.get(p, {})
        c = sc.get("confidence")
        flag = "" if sc.get("eligible", True) else "*"
        bits.append(f"{p} {'n/a' if c is None else f'{c:.2f}'}{flag}")
    print(f"  {rule.template:22s} {'  '.join(bits)}   {rule.render(ground)}")


def generate(pairs=None, source: str = "py", templates=None, ground: bool = False,
             ci: bool = False,
             verbose: bool = True, score: bool = False, split: bool = True,
             **template_kw) -> dict:
    """Run the template library over several pairs.

    `pairs=None` (or ["all"]) means every pair available under `source`.
    `score=True` adds temporal confidence; `split=True` fits on train and scores on test.
    Returns {"rules": [...], "table": DataFrame, "pairs": [...]}.
    """
    if pairs is None or list(pairs) == ["all"]:
        pairs = discover_pairs(source)
    pairs = [normalize_pair(q, source) for q in pairs]
    if not pairs:
        raise FileNotFoundError(
            f"no labelled pairs under {source_dir(source)} — run prep_real_pairs.py "
            "(py/reasoner) or infer.py (classifier) first")

    all_rules = []
    for pair in pairs:
        if verbose:
            print(f"\n{pair}  [{source}]")
        if score:
            all_rules += score_pair(pair, source, templates, ground=ground, split=split, ci=ci,
                                    verbose=verbose, **template_kw)
        else:
            all_rules += generate_for_pair(pair, source, templates, ground=ground,
                                           verbose=verbose, **template_kw)

    table = pd.DataFrame([r.to_row(ground) for r in all_rules])
    return {"rules": all_rules, "table": table, "pairs": pairs}
