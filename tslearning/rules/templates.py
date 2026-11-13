"""
The rule template library.  ← ADD NEW TEMPLATES HERE



Each template is one function decorated with @template. It receives a `PairFacts` and
returns a `Rule` (or None when the data shows no such effect — a legitimate, reportable
outcome, not a failure to route around).

────────────────────────────────────────────────────────────────────────────────
HOW TO ADD A TEMPLATE
────────────────────────────────────────────────────────────────────────────────

    @template("T8_myPhenomenon", needs=["highVol", "isPositiveCorrelation"])
    def t8_my_phenomenon(f: PairFacts) -> "Rule | None":
        condition  = f.iv("highVol", f.x)
        conclusion = f.iv("isPositiveCorrelation", f.x, f.y)
        fit = choose_w(condition, conclusion, "+", T0=f.T0, T1=f.T1)
        if fit["w"] is None:
            return None
        return Rule(
            template="T8_myPhenomenon", pair=f.pair, x=f.x, y=f.y,
            head="myPhenomenon({X},{Y})",
            body=f'highVol({{X}}), <+>[0,{fit["w"]}] isPositiveCorrelation({{X}},{{Y}})',
            constants={"w": fit["w"]},
            diagnostics={k: fit[k] for k in ("se", "best_gap", "gap_at_w", "n_episodes")},
        )

`needs` lists the predicates that must be present; the generator skips the template with
a clear reason if any are missing, instead of failing.

Write {X} and {Y} as literal placeholders in head/body — Rule.render() substitutes either
variables or ground series names (see rule.py). Never hardcode a series name here.

Keep head/body in the COMPACT readable form (that is what goes in the paper and the results
table). You do not need to make them engine-valid: verification pads each body atom to the
arity the .vada declares, so `highVol({X})` is executed as `highVol({X},_)` — several
predicates carry a value column (highVol/lowVol have Pct, trend* have Sum). If a template
needs something the automatic fix cannot produce, set `vada_head=` / `vada_body=` on the
Rule and verification uses those verbatim.

Templates are tied to a named financial phenomenon with a citation.
Do not add a template just because DatalogMTL can express it — that is what produced the
meaningless duration/recurrence rules the old miner emitted.
"""
from __future__ import annotations

from tslearning.rules.fitting import W_GRID
from tslearning.rules.fitting import (choose_d, choose_d_transition, choose_lag,
                                      choose_w)
from tslearning.rules.primitives import PairFacts, overlap
from tslearning.rules.rule import Rule

# name -> {"fn", "needs", "doc", "phenomenon", "cite", "schema", "fits"}
TEMPLATES: dict = {}


def template(name: str, needs=(), phenomenon: str = "", cite: str = "",
             schema: str = "", fits=()):
    """Register a rule template under `name`.

    The last four arguments are PAPER METADATA — they are what `scripts/rule_table.py`
    prints as Table 1 (the rule library). Keeping them on the decorator means a new
    template appears in the paper's method table automatically; a hardcoded table in the
    LaTeX would silently go stale the moment you add T8.

      phenomenon  the named effect, as a finance reader would call it
      cite        the source that names it — every template must have one (see the module
                  docstring: expressible-in-DatalogMTL is not a reason to include a rule)
      schema      the body with the fitted constants left SYMBOLIC (w, k, d …). The fitted
                  values differ per pair and belong in Table 2, not here.
      fits        which constants this template fits, in the order they should be shown
    """
    def deco(fn):
        TEMPLATES[name] = {"fn": fn, "needs": list(needs),
                           "doc": (fn.__doc__ or "").strip().split("\n")[0],
                           "phenomenon": phenomenon, "cite": cite,
                           "schema": schema, "fits": list(fits)}
        return fn
    return deco


def available() -> list:
    return sorted(TEMPLATES)


# ══════════════════════════════════════════════════════════════════════════════
#  T1 — Safe haven  (Baur & Lucey 2010; Baur & McDermott 2010)
# ══════════════════════════════════════════════════════════════════════════════

@template("T1_safeHaven", needs=["highVol", "trendDown", "isNegativeCorrelation"],
          phenomenon="Safe haven", cite="Baur & Lucey 2010; Baur & McDermott 2010",
          schema="highVol(X), trendDown(X), <+>[0,w] isNegativeCorrelation(X,Y)",
          fits=["w"])
def t1_safe_haven(f: PairFacts, *, w_grid=W_GRID, direction="+", **kw):
    """Y is negatively correlated with X during X's stressed, falling phases."""
    condition = overlap(f.iv("highVol", f.x), f.iv("trendDown", f.x))
    conclusion = f.iv("isNegativeCorrelation", f.x, f.y)
    fit = choose_w(condition, conclusion, direction, w_grid, f.T0, f.T1)
    if fit["w"] is None:
        return Rule.rejected("T1_safeHaven", f, fit["reason"], diagnostics=fit)
    w = fit["w"]
    return Rule(
        template="T1_safeHaven", pair=f.pair, x=f.x, y=f.y,
        head="safeHavenEpisode({Y},{X})",
        body=f"highVol({{X}}), trendDown({{X}}), <{direction}>[0,{w}] "
             f"isNegativeCorrelation({{X}},{{Y}})",
        vada_body=f"highVol({{X}}), trendDown({{X}}), <{direction}>[0,{w}] "
             f"isNegativeCorrelation({{X}},{{Y}})",
        constants={"w": w},
        diagnostics=_diag(fit),
        parts={"kind": "conditional",
               "condition": [("highVol", "x"), ("trendDown", "x")],
               "conclusion": ("isNegativeCorrelation", "xy"),
               "direction": direction},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  T2 — Contagion  (Longin & Solnik 2001; Ang & Chen 2002; Bae/Karolyi/Stulz 2003)
# ══════════════════════════════════════════════════════════════════════════════

@template("T2_contagion", needs=["tailDependent", "isPositiveCorrelation"],
          phenomenon="Contagion / co-exceedance",
          cite="Longin & Solnik 2001; Bae, Karolyi & Stulz 2003",
          schema="tailDependent(X,Y), <+>[0,w] isPositiveCorrelation(X,Y)",
          fits=["w"])
def t2_contagion(f: PairFacts, *, w_grid=W_GRID, direction="+", **kw):
    """Correlation turns positive around joint tail events (co-crashes)."""
    condition = f.iv("tailDependent", f.x, f.y)
    conclusion = f.iv("isPositiveCorrelation", f.x, f.y)
    fit = choose_w(condition, conclusion, direction, w_grid, f.T0, f.T1)
    if fit["w"] is None:
        return Rule.rejected("T2_contagion", f, fit["reason"], diagnostics=fit)
    w = fit["w"]
    return Rule(
        template="T2_contagion", pair=f.pair, x=f.x, y=f.y,
        head="contagion({X},{Y})",
        body=f"tailDependent({{X}},{{Y}}), <{direction}>[0,{w}] "
             f"isPositiveCorrelation({{X}},{{Y}})",
        vada_body=f"tailDependent({{X}},{{Y}}), <{direction}>[0,{w}] "
             f"isPositiveCorrelation({{X}},{{Y}})",
        constants={"w": w},
        diagnostics=_diag(fit),
        parts={"kind": "conditional",
               "condition": [("tailDependent", "xy")],
               "conclusion": ("isPositiveCorrelation", "xy"),
               "direction": direction},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  T3 — Decoupling in calm markets (converse of T2)
# ══════════════════════════════════════════════════════════════════════════════

@template("T3_decoupled", needs=["lowVol", "isNoCorrelation"],
          phenomenon="Decoupling in calm markets",
          cite="converse of T2 (Ang & Chen 2002)",
          schema="lowVol(X), lowVol(Y), <+>[0,w] isNoCorrelation(X,Y)",
          fits=["w"])
def t3_decoupled(f: PairFacts, *, w_grid=W_GRID, direction="+", **kw):
    """The pair decouples while both series are calm."""
    condition = overlap(f.iv("lowVol", f.x), f.iv("lowVol", f.y))
    conclusion = f.iv("isNoCorrelation", f.x, f.y)
    fit = choose_w(condition, conclusion, direction, w_grid, f.T0, f.T1)
    if fit["w"] is None:
        return Rule.rejected("T3_decoupled", f, fit["reason"], diagnostics=fit)
    w = fit["w"]
    return Rule(
        template="T3_decoupled", pair=f.pair, x=f.x, y=f.y,
        head="decoupled({X},{Y})",
        body=f"lowVol({{X}}), lowVol({{Y}}), <{direction}>[0,{w}] "
             f"isNoCorrelation({{X}},{{Y}})",
        constants={"w": w},
        diagnostics=_diag(fit),
        parts={"kind": "conditional",
               "condition": [("lowVol", "x"), ("lowVol", "y")],
               "conclusion": ("isNoCorrelation", "xy"),
               "direction": direction},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  T4 — Lead-lag under stress (price discovery / spillover)
# ══════════════════════════════════════════════════════════════════════════════

@template("T4_leadsUnderStress", needs=["highVol", "leadsPositively"],
          phenomenon="Lead-lag / price discovery under stress",
          # leadsPositively uses the SAME magnitude-gated concordance as lag 0
          # (py_labeler.compute_concordance: a day counts only if |X| and |Y| both rise or
          # both fall, then is split by sign). It is magnitude-and-sign, NOT sign-only, so
          # this is the price-discovery strand. The statistic differs from theirs
          # (nonparametric count vs. VAR information share) — state that as a deliberate
          # fat-tail-robustness choice. See references.md, lead-lag section.
          cite="Hasbrouck 1995; de Jong & Nijman 1997",
          schema="highVol(X), <+>[0,w] leadsPositively(X,Y,k)",
          fits=["w", "k"])
def t4_leads_under_stress(f: PairFacts, *, w_grid=W_GRID, direction="+",
                          lags=None, **kw):
    """X leads Y at lag k specifically while X is turbulent.

    Both k and w are fitted by the 1-SE rule (§3.1(d) keeps one lag, not five): score
    every lag, bootstrap the standard error of the best, then keep the SMALLEST lag whose
    gap is within one SE of it. Argmax over lags would be the same non-identifying
    criterion the 1-SE rule exists to avoid for w.
    CAVEAT: the underlying lead scores are raw lagged concordance. At small k the 50-day
    windows overlap ~98%, so this inherits any contemporaneous correlation until the
    net-lead fix lands.
    """
    condition = f.iv("highVol", f.x)
    lags = lags or f.lags_present(["leadsPositively"])
    # k is fitted by the SAME 1-SE rule as w, not by argmax over lags — see choose_lag.
    sel = choose_lag(condition,
                     {k: f.lead_iv("leadsPositively", k, f.x, f.y) for k in lags},
                     direction, w_grid, f.T0, f.T1)
    if sel["k"] is None:
        return Rule.rejected("T4_leadsUnderStress", f, sel["reason"])
    k, fit = sel["k"], sel["fit"]
    w = fit["w"]
    return Rule(
        template="T4_leadsUnderStress", pair=f.pair, x=f.x, y=f.y,
        head="leadsUnderStress({X},{Y})",
        body=f"highVol({{X}}), <{direction}>[0,{w}] leadsPositively({{X}},{{Y}},{k})",
        vada_body=f"highVol({{X}}), <{direction}>[0,{w}] leadsPositively({{X}},{{Y}},{k})",
        constants={"w": w, "k": k},
        diagnostics={**_diag(fit), "k_argmax": sel.get("k_argmax"),
                     "k_gaps": sel.get("gaps")},
        parts={"kind": "conditional",
               "condition": [("highVol", "x")],
               "conclusion": ("leadsPositively", "xy"),
               "direction": direction},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  T5 — Hedge horizon (how long a negative-correlation regime can be relied on)
# ══════════════════════════════════════════════════════════════════════════════

@template("T5_hedgeHorizon", needs=["isNegativeCorrelation"],
          phenomenon="Hedge horizon (regime persistence)",
          cite="Baur & Lucey 2010 (hedge vs. haven)",
          schema="isNegativeCorrelation(X,Y), [+][0,d] isNegativeCorrelation(X,Y)",
          fits=["d"])
def t5_hedge_horizon(f: PairFacts, *, pct=25, **kw):
    """Once established, a negative-correlation regime persists at least d."""
    fit = choose_d(f.iv("isNegativeCorrelation", f.x, f.y), pct=pct)
    if fit["d"] is None:
        return Rule.rejected("T5_hedgeHorizon", f, fit["reason"], diagnostics=fit)
    d = fit["d"]
    return Rule(
        template="T5_hedgeHorizon", pair=f.pair, x=f.x, y=f.y,
        head="hedgeHorizon({X},{Y})",
        body=f"isNegativeCorrelation({{X}},{{Y}}), [+][0,{d}] "
             f"isNegativeCorrelation({{X}},{{Y}})",
        vada_body=f"isNegativeCorrelation({{X}},{{Y}}), [+][0,{d}] "
             f"isNegativeCorrelation({{X}},{{Y}})",
        constants={"d": d},
        diagnostics={k: fit[k] for k in ("n", "dwell_min", "dwell_median", "dwell_max")},
        calendar_days={"d": f.calendar(d)},
        parts={"kind": "persistence", "pred": ("isNegativeCorrelation", "xy")},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  T6 — Regime reversal (correlation mean-reversion)
# ══════════════════════════════════════════════════════════════════════════════

@template("T6_regimeReversal", needs=["isNegativeCorrelation", "isPositiveCorrelation"],
          phenomenon="Correlation regime reversal",
          cite="Ang & Bekaert 2002 (regime switching)",
          schema="[-](0,d⁻] isNegativeCorrelation(X,Y), [+][a,b] isPositiveCorrelation(X,Y)",
          fits=["d_neg", "a", "b"])
def t6_regime_reversal(f: PairFacts, *, pct=25, max_gap=float("inf"), **kw):
    """A negative-correlation regime is followed by a positive one."""
    fit = choose_d_transition(f.iv("isNegativeCorrelation", f.x, f.y),
                              f.iv("isPositiveCorrelation", f.x, f.y),
                              pct=pct, max_gap=max_gap)
    if fit["reason"]:
        return Rule.rejected("T6_regimeReversal", f, fit["reason"], diagnostics=fit)
    d, a, b = fit["d_from"], fit["a"], fit["b"]
    return Rule(
        template="T6_regimeReversal", pair=f.pair, x=f.x, y=f.y,
        head="regimeReversal({X},{Y})",
        body=f"[-](0,{d}] isNegativeCorrelation({{X}},{{Y}}), [+][{a},{b}] "
             f"isPositiveCorrelation({{X}},{{Y}})",
        vada_body=f"[-](0,{d}] isNegativeCorrelation({{X}},{{Y}}), [+][{a},{b}] "
             f"isPositiveCorrelation({{X}},{{Y}})",
        constants={"d_neg": d, "a": a, "b": b},
        diagnostics={k: fit[k] for k in ("n", "gap_median")},
        calendar_days={"d_neg": f.calendar(d), "b": f.calendar(b)},
        parts={"kind": "transition",
               "from": ("isNegativeCorrelation", "xy"),
               "to": ("isPositiveCorrelation", "xy")},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  T7 — Macro conditioning (§3.2). THE MIXED-GRANULARITY TEMPLATE.
# ══════════════════════════════════════════════════════════════════════════════
#
# Why this one matters beyond breadth: every other template combines predicates derived at
# the SAME granularity (daily windows over daily returns). T7 intersects a MONTHLY macro
# state with a DAILY correlation regime, with no resampling of either — which is what
# interval semantics is for, and the one place the declarative layer does something a
# pandas pipeline would have to fake.
#
# The macro series is NOT a member of the pair (that would force both onto one clock and is
# refused by pairs.check_same_frequency). It is a background CONDITION, occupying the same
# slot `highVol(X)` occupies in T1. See labeling/macro_state.py.

@template("T7_macroConditioned",
          needs=["market", "inflationHigh", "isPositiveCorrelation"],
          phenomenon="Macro-conditioned co-movement",
          cite="Bekaert, Hoerova & Lo Duca 2013",
          schema="market(X,C), inflationHigh(C), <+>[0,w] isPositiveCorrelation(X,Y)",
          fits=["w"])
def t7_macro_conditioned(f: PairFacts, *, w_grid=W_GRID, direction="+",
                         macro_pred: str = "inflationHigh", **kw):
    """The pair co-moves more while inflation in X's economy is elevated.

    market(X, C) is what ties the macro state to the pair: the inflation is that of the
    economy X belongs to, stated in the rule rather than implied by the choice of pair.
    """
    c = f.market_of(f.x)
    if c is None:
        return Rule.rejected("T7_macroConditioned", f,
                             f"missing predicates: ['market'] (no economy for {f.x})")
    condition = f.iv(macro_pred, c)
    if not condition:
        return Rule.rejected("T7_macroConditioned", f,
                             f"missing predicates: ['{macro_pred}'] (none for {c})")
    conclusion = f.iv("isPositiveCorrelation", f.x, f.y)
    fit = choose_w(condition, conclusion, direction, w_grid, f.T0, f.T1)
    if fit["w"] is None:
        return Rule.rejected("T7_macroConditioned", f, fit["reason"], diagnostics=fit)
    w = fit["w"]
    return Rule(
        template="T7_macroConditioned", pair=f.pair, x=f.x, y=f.y,
        head="riskOffRegime({X},{Y})",
        body=f"market({{X}},C), {macro_pred}(C), <{direction}>[0,{w}] "
             f"isPositiveCorrelation({{X}},{{Y}})",
        constants={"w": w},
        diagnostics=_diag(fit),
        parts={"kind": "conditional",
               "condition": [("market", "xc"), (macro_pred, "cx")],
               "conclusion": ("isPositiveCorrelation", "xy"),
               "direction": direction},
    )


def _diag(fit: dict) -> dict:
    return {k: fit.get(k) for k in ("se", "best_gap", "gap_at_w", "n_episodes",
                                    "condition_days")}


# ══════════════════════════════════════════════════════════════════════════════
#  BACKWARD-PROPAGATING VARIANTS (condition ↔ conclusion swapped, ◇⁻)
# ══════════════════════════════════════════════════════════════════════════════
#
# Each forward conditional template asks "when the trigger holds NOW, does the
# conclusion appear within the next w days?"  The backward variant asks the
# symmetric question: "when the (former) conclusion holds NOW, did the (former)
# trigger occur within the PAST w days?"
#
# Logically equivalent claim, but different SCORING because the denominator
# changes: conditioning on the rarer event gives higher confidence (e.g. T2 / T2b).
#
# These are registered as separate templates so they mine, fit, and score
# independently, so the two framings can be compared and the more useful one
# produces more useful rules for each pair.


# ──────────────────────────────────────────────────────────────────────────────
#  T1b — Safe haven (backward)
# ──────────────────────────────────────────────────────────────────────────────

@template("T1b_safeHaven_bw",
          needs=["highVol", "trendDown", "isNegativeCorrelation"],
          phenomenon="Safe haven (backward)",
          cite="Baur & Lucey 2010; Baur & McDermott 2010",
          schema="stressedDown(X) :- highVol(X), trendDown(X); "
                 "isNegativeCorrelation(X,Y), <->[0,w] stressedDown(X)",
          fits=["w"])
def t1b_safe_haven_bw(f: PairFacts, *, w_grid=W_GRID, **kw):
    """When X,Y are negatively correlated, X was stressed and falling within the past w days."""
    condition = f.iv("isNegativeCorrelation", f.x, f.y)
    conclusion = overlap(f.iv("highVol", f.x), f.iv("trendDown", f.x))
    direction = "-"
    fit = choose_w(condition, conclusion, direction, w_grid, f.T0, f.T1)
    if fit["w"] is None:
        return Rule.rejected("T1b_safeHaven_bw", f, fit["reason"], diagnostics=fit)
    w = fit["w"]
    return Rule(
        template="T1b_safeHaven_bw", pair=f.pair, x=f.x, y=f.y,
        head="safeHavenEpisodeBw({Y},{X})",
        # highVol and trendDown must hold on the SAME day (Python scores their overlap),
        # and DatalogMTL has no conjunction under an operator — so name it first.
        body=f"isNegativeCorrelation({{X}},{{Y}}), <{direction}>[0,{w}] stressedDown({{X}})",
        vada_aux=["stressedDown({X}) :- highVol({X}), trendDown({X})"],
        constants={"w": w},
        diagnostics=_diag(fit),
        parts={"kind": "conditional",
               "condition": [("isNegativeCorrelation", "xy")],
               "conclusion_multi": [("highVol", "x"), ("trendDown", "x")],
               "conclusion": ("highVol", "x"),   # primary, for scoring
               "direction": direction},
    )


# ──────────────────────────────────────────────────────────────────────────────
#  T2b — Contagion (backward)
# ──────────────────────────────────────────────────────────────────────────────

@template("T2b_contagion_bw",
          needs=["tailDependent", "isPositiveCorrelation"],
          phenomenon="Contagion / co-exceedance (backward)",
          cite="Longin & Solnik 2001; Bae, Karolyi & Stulz 2003",
          schema="isPositiveCorrelation(X,Y), <->[0,w] tailDependent(X,Y)",
          fits=["w"])
def t2b_contagion_bw(f: PairFacts, *, w_grid=W_GRID, **kw):
    """When positively correlated, a co-crash occurred within the past w days."""
    condition = f.iv("isPositiveCorrelation", f.x, f.y)
    conclusion = f.iv("tailDependent", f.x, f.y)
    direction = "-"
    fit = choose_w(condition, conclusion, direction, w_grid, f.T0, f.T1)
    if fit["w"] is None:
        return Rule.rejected("T2b_contagion_bw", f, fit["reason"], diagnostics=fit)
    w = fit["w"]
    return Rule(
        template="T2b_contagion_bw", pair=f.pair, x=f.x, y=f.y,
        head="contagionBw({X},{Y})",
        body=f"isPositiveCorrelation({{X}},{{Y}}), <{direction}>[0,{w}] "
             f"tailDependent({{X}},{{Y}})",
        constants={"w": w},
        diagnostics=_diag(fit),
        parts={"kind": "conditional",
               "condition": [("isPositiveCorrelation", "xy")],
               "conclusion": ("tailDependent", "xy"),
               "direction": direction},
    )


# ──────────────────────────────────────────────────────────────────────────────
#  T3b — Decoupling (backward)
# ──────────────────────────────────────────────────────────────────────────────

@template("T3b_decoupled_bw",
          needs=["lowVol", "isNoCorrelation"],
          phenomenon="Decoupling in calm markets (backward)",
          cite="converse of T2 (Ang & Chen 2002)",
          schema="bothCalm(X,Y) :- lowVol(X), lowVol(Y); "
                 "isNoCorrelation(X,Y), <->[0,w] bothCalm(X,Y)",
          fits=["w"])
def t3b_decoupled_bw(f: PairFacts, *, w_grid=W_GRID, **kw):
    """When uncorrelated, both series were calm within the past w days."""
    condition = f.iv("isNoCorrelation", f.x, f.y)
    conclusion = overlap(f.iv("lowVol", f.x), f.iv("lowVol", f.y))
    direction = "-"
    fit = choose_w(condition, conclusion, direction, w_grid, f.T0, f.T1)
    if fit["w"] is None:
        return Rule.rejected("T3b_decoupled_bw", f, fit["reason"], diagnostics=fit)
    w = fit["w"]
    return Rule(
        template="T3b_decoupled_bw", pair=f.pair, x=f.x, y=f.y,
        head="decoupledBw({X},{Y})",
        # both calm on the SAME day (Python scores the overlap) — named first, as in T1b
        body=f"isNoCorrelation({{X}},{{Y}}), <{direction}>[0,{w}] bothCalm({{X}},{{Y}})",
        vada_aux=["bothCalm({X},{Y}) :- lowVol({X}), lowVol({Y})"],
        constants={"w": w},
        diagnostics=_diag(fit),
        parts={"kind": "conditional",
               "condition": [("isNoCorrelation", "xy")],
               "conclusion_multi": [("lowVol", "x"), ("lowVol", "y")],
               "conclusion": ("lowVol", "x"),   # primary, for scoring
               "direction": direction},
    )


# ──────────────────────────────────────────────────────────────────────────────
#  T4b — Lead-lag under stress (backward)
# ──────────────────────────────────────────────────────────────────────────────
#
# Forward T4: highVol(X) is fixed condition, leadsPositively(X,Y,k) varies by k.
# Backward:   leadsPositively(X,Y,k) is the condition (varies by k), highVol(X) is fixed.
# choose_lag expects a fixed condition and conclusion_by_lag, so we need a small adapter.

def _choose_lag_backward(conclusion_fixed, condition_by_lag: dict, direction: str = "-",
                         w_grid=W_GRID, T0: int = 0, T1: int = 0,
                         n_boot: int = 500, seed: int = 42) -> dict:
    """Like choose_lag, but the LAG varies the CONDITION side (backward T4).

    For each lag k, condition = leadsPositively at that lag, conclusion = highVol (fixed).
    The 1-SE rule on k still picks the smallest lag within one SE of the best.
    """
    per_lag = {}
    for k, condition in condition_by_lag.items():
        if not condition:
            continue
        fit = choose_w(list(condition), conclusion_fixed, direction, w_grid, T0, T1,
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


@template("T4b_leadsUnderStress_bw",
          needs=["highVol", "leadsPositively"],
          phenomenon="Lead-lag / price discovery under stress (backward)",
          cite="Hasbrouck 1995; de Jong & Nijman 1997",
          schema="leadsPositively(X,Y,k), <->[0,w] highVol(X)",
          fits=["w", "k"])
def t4b_leads_under_stress_bw(f: PairFacts, *, w_grid=W_GRID,
                               lags=None, **kw):
    """When X leads Y at lag k, X was turbulent within the past w days."""
    conclusion = f.iv("highVol", f.x)
    lags = lags or f.lags_present(["leadsPositively"])
    direction = "-"
    sel = _choose_lag_backward(
        conclusion,
        {k: f.lead_iv("leadsPositively", k, f.x, f.y) for k in lags},
        direction, w_grid, f.T0, f.T1)
    if sel["k"] is None:
        return Rule.rejected("T4b_leadsUnderStress_bw", f, sel["reason"])
    k, fit = sel["k"], sel["fit"]
    w = fit["w"]
    return Rule(
        template="T4b_leadsUnderStress_bw", pair=f.pair, x=f.x, y=f.y,
        head="leadsUnderStressBw({X},{Y})",
        body=f"leadsPositively({{X}},{{Y}},{k}), <{direction}>[0,{w}] highVol({{X}})",
        constants={"w": w, "k": k},
        diagnostics={**_diag(fit), "k_argmax": sel.get("k_argmax"),
                     "k_gaps": sel.get("gaps")},
        parts={"kind": "conditional",
               "condition": [("leadsPositively", "xy")],
               "conclusion": ("highVol", "x"),
               "lag_side": "condition",
               "direction": direction},
    )


# ──────────────────────────────────────────────────────────────────────────────
#  T7b — Macro-conditioned co-movement (backward)
# ──────────────────────────────────────────────────────────────────────────────

@template("T7b_macroConditioned_bw",
          needs=["market", "inflationHigh", "isPositiveCorrelation"],
          phenomenon="Macro-conditioned co-movement (backward)",
          cite="Bekaert, Hoerova & Lo Duca 2013",
          schema="isPositiveCorrelation(X,Y), market(X,C), <->[0,w] inflationHigh(C)",
          fits=["w"])
def t7b_macro_conditioned_bw(f: PairFacts, *, w_grid=W_GRID,
                              macro_pred: str = "inflationHigh", **kw):
    """When the pair co-moves, inflation in X's economy was elevated within the past w days."""
    c = f.market_of(f.x)
    if c is None:
        return Rule.rejected("T7b_macroConditioned_bw", f,
                             f"missing predicates: ['market'] (no economy for {f.x})")
    conclusion = f.iv(macro_pred, c)
    if not conclusion:
        return Rule.rejected("T7b_macroConditioned_bw", f,
                             f"missing predicates: ['{macro_pred}'] (none for {c})")
    condition = f.iv("isPositiveCorrelation", f.x, f.y)
    direction = "-"
    fit = choose_w(condition, conclusion, direction, w_grid, f.T0, f.T1)
    if fit["w"] is None:
        return Rule.rejected("T7b_macroConditioned_bw", f, fit["reason"], diagnostics=fit)
    w = fit["w"]
    return Rule(
        template="T7b_macroConditioned_bw", pair=f.pair, x=f.x, y=f.y,
        head="riskOffRegimeBw({X},{Y})",
        body=f"isPositiveCorrelation({{X}},{{Y}}), market({{X}},C), <{direction}>[0,{w}] "
             f"{macro_pred}(C)",
        constants={"w": w},
        diagnostics=_diag(fit),
        parts={"kind": "conditional",
               "condition": [("isPositiveCorrelation", "xy"), ("market", "xc")],
               "conclusion": (macro_pred, "cx"),
               "direction": direction},
    )
