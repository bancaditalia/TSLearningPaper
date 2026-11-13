"""
Building the prompt — what the model is shown, and what it is asked for.

────────────────────────────────────────────────────────────────────────────────
THE ONE RULE THAT MAKES THIS A MEASUREMENT
────────────────────────────────────────────────────────────────────────────────

**The model sees the TRAIN period only.** Its rules are then frozen and scored on the
test period, by the same code and the same protocol as the template fitter. If the model
saw the test period the comparison would be meaningless, and it would be meaningless in
the direction that flatters the model.

────────────────────────────────────────────────────────────────────────────────
TWO INPUT MODES, AND WHY BOTH
────────────────────────────────────────────────────────────────────────────────

  facts   the predicate intervals — the SAME input the template fitter receives.
          This is the like-for-like comparison: given identical evidence, does the model
          propose better rules than the template library?
  series  the raw daily returns. A harder and more realistic task (label AND mine), and
          the one usually meant by "give an LLM the time series". Expect it to do worse;
          the interesting question is HOW it fails — plausible rules with badly fitted
          constants is the failure mode to look for, and the one the 1-SE rule exists to
          prevent.

Run both. The gap between them isolates how much of the difficulty is in the labelling.
"""
from __future__ import annotations

import numpy as np

from tslearning.config import CFG

# ── operator crib, kept short on purpose ──────────────────────────────────────
OPERATORS = """\
<+>[0,w] P   P holds at some point within the next w days      ("diamond future")
<->[0,w] P   P holds at some point within the previous w days  ("diamond past")
[+][0,d] P   P holds throughout the next d days                ("box future")
[-](0,d] P   P held throughout the previous d days             ("box past")"""


def vocabulary() -> str:
    """The predicate vocabulary the model may use, read from the config registry."""
    kinds = CFG["predicates"]["kind"]
    lines, seen = [], set()
    describe = {
        "highVol": "X is in a turbulent phase (many large daily moves)",
        "lowVol": "X is in a calm phase",
        "trendUp": "X rose over the window", "trendDown": "X fell over the window",
        "tailDependent": "X and Y had their worst days on the SAME days",
        "isPositiveCorrelation": "X and Y co-moved in the same direction",
        "isNegativeCorrelation": "X and Y moved in opposite directions",
        "isNoCorrelation": "no clear directional relationship",
        "leadsPositively": "X's move at t agrees with Y's move at t+k",
        "market": "series X belongs to economy C (static, e.g. S&P 500 -> US)",
        "inflationHigh": "inflation in economy C was elevated (a MONTHLY fact)",
        "inflationLow": "inflation in economy C was low (a MONTHLY fact)",
    }
    for p, kind in kinds.items():
        if p not in describe or p in seen:
            continue
        seen.add(p)
        # macro predicates are about an ECONOMY, reached through market(X, C) — showing
        # them as f(X) would invite inflationHigh(X), which is about no economy at all
        if p == "market":
            sig = "market(X,C)"
        elif p.startswith("inflation"):
            sig = f"{p}(C)"
        elif kind.startswith("lag"):
            sig = f"{p}(X,Y,k)"
        elif kind.startswith("unary"):
            sig = f"{p}(X)"
        else:
            sig = f"{p}(X,Y)"
        lines.append(f"  {sig:<34} {describe[p]}")
    return "\n".join(lines)


def template_catalogue() -> str:
    """The existing templates, as worked examples of the shape wanted back."""
    from tslearning.rules.templates import TEMPLATES
    out = []
    for name in sorted(TEMPLATES):
        m = TEMPLATES[name]
        out.append(f"  {name}\n"
                   f"    phenomenon : {m.get('phenomenon') or '—'}\n"
                   f"    body       : {m.get('schema') or '—'}\n"
                   f"    fits       : {', '.join(m.get('fits') or []) or '—'}")
    return "\n".join(out)


def serialise_series(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str,
                     days=None, dates=None, max_days: int = 1800,
                     decimals: int = 4) -> str:
    """Daily log-returns as a compact table, indexed by the pair's own day numbers.

    Truncated from the END, keeping the EARLIEST training days, so the train/test boundary
    is never straddled — a model that glimpsed even a few test-period days would make the
    whole comparison worthless in the direction that flatters it.
    """
    a, b = np.asarray(a)[:max_days], np.asarray(b)[:max_days]
    rows = [f"day,date,{name_a},{name_b}"]
    for i in range(len(a)):
        day = days[i] if days is not None and i < len(days) else i
        d = str(dates[i])[:10] if dates is not None and i < len(dates) else ""
        rows.append(f"{day},{d},{a[i]:.{decimals}f},{b[i]:.{decimals}f}")
    out = "\n".join(rows)
    if len(np.asarray(a)) >= max_days:
        out += f"\n# truncated to the first {max_days} training days"
    return out


# ── operational definitions, needed ONLY for the raw-series task ──────────────
# Without these the model is guessing at our definitions rather than reasoning about the
# data, and any failure would be uninterpretable — we would have measured whether it can
# read our minds, not whether it can propose rules.
DEFINITIONS = """\
HOW THE PREDICATES ARE DEFINED — apply these to the returns above.

Everything is computed over a sliding window of {seq_len} trading days.

Two intermediate quantities, used below. THESE ARE NOT PREDICATES and must not appear
in a rule: a day is "big" if |return| exceeds tau (the 90th percentile of that series'
|daily return| over the training period), and "extreme" if the return is below qLow
(its 5th percentile).

  highVol(X)      {high_vol:g}% or more of the window's days are big for X
  lowVol(X)       {low_vol:g}% or fewer are
  trendUp(X)      the window's returns sum to +{trend:g} or more
  trendDown(X)    they sum to -{trend:g} or less
  tailDependent(X,Y)  the window contains at least {tail_min} days on which BOTH series were extreme

  For the correlation regime, a day counts only if |X| and |Y| BOTH rose or BOTH fell
  versus the previous day (a "magnitude-consistent" day). Among those days:
      same% = share of window days that are magnitude-consistent AND X,Y same sign
      diff% = share that are magnitude-consistent AND X,Y opposite sign
      score = same% - diff%
  same%, diff% and score are intermediate quantities too: NOT predicates, and must not
  appear in a rule (no comparisons such as same%(X,Y) > 50 either).
  isPositiveCorrelation(X,Y)   score >  +{deadband}
  isNegativeCorrelation(X,Y)   score <  -{deadband}
  isNoCorrelation(X,Y)         otherwise
  leadsPositively(X,Y,k)       the same score, computed with Y shifted k days later,
                               exceeding the lag-0 score (a NET lead)"""


def definitions_block() -> str:
    from tslearning.config import CFG
    from tslearning.labeling.exact_flags import label_constants
    c = label_constants()
    return DEFINITIONS.format(
        high_vol=c["high_vol_pct"], low_vol=c["low_vol_pct"], trend=c["trend"],
        tail_min=c["tail_min"],
        seq_len=CFG["windows"]["financial"]["length"],
        deadband=CFG.get("labels", {}).get("deadband", 10))


def serialise_facts(intervals: dict, max_per_pred: int = 60,
                    max_total: int = 900) -> str:
    """Predicate intervals as `predicate(args) @ [start, end)` lines.

    `intervals` is {predicate: [(key, start, end), ...]} where `key` is a series name, a
    (X, Y) tuple, or (X, Y, lag) for the lag predicates — so the arguments are rendered
    from the key rather than assumed.

    Capped per predicate and overall. `leadsPositively` alone spans five lags and would
    otherwise crowd out everything else, and a prompt where one predicate occupies most of
    the evidence is not a fair presentation of the data.
    """
    lines, total, truncated = [], 0, []
    for pred in sorted(intervals):
        ivs = intervals[pred]
        shown = 0
        for item in ivs:
            if total >= max_total:
                break
            if shown >= max_per_pred:
                break
            key, s, e = item[0], item[1], item[2]
            if isinstance(key, (tuple, list)):
                args = ",".join(str(k) for k in key)
            else:
                args = str(key)
            lines.append(f"{pred}({args}) @ [{s}, {e})")
            shown += 1
            total += 1
        if shown < len(ivs):
            truncated.append(f"{pred}: showed {shown} of {len(ivs)}")
    if truncated:
        lines.append("")
        lines.append("# truncated for length — " + "; ".join(truncated))
    return "\n".join(lines) if lines else "(no facts)"


PROPOSE_HEADER = """\
You are helping evaluate a rule-mining system for financial time series.

Your task: propose temporal rules, in DatalogMTL, describing how two assets behave
relative to one another. The rules will be FROZEN and then scored on a later, unseen
period — so a rule that merely restates what is usually true will score badly. Prefer
rules whose CONDITION selects a specific situation and whose CONCLUSION would not hold
anyway.

TEMPORAL OPERATORS
{operators}

PREDICATE VOCABULARY — except for the head of the rules you propose, use only these
{vocabulary}

Facts hold over half-open intervals [start, end): day 3 to day 6 covers days 3, 4, 5.

The output contract is stated at the END of this message, after the data.
"""

# The contract goes LAST, after the data and the definitions.
#
# Why: a model answers the last thing it read. With the contract in the header and 14,000
# tokens of returns after it, llama3.1:8b responded to the trailing definitions instead —
# it wrote Python functions implementing highVol and trendDown, which is a perfectly
# sensible reply to "here are some definitions" and no reply at all to the actual task.
PROPOSE_CONTRACT = """\
────────────────────────────────────────────────────────────────────────────────
YOUR TASK — this is what to do with everything above
────────────────────────────────────────────────────────────────────────────────

Do NOT write code. Do NOT implement the definitions. Do NOT explain the data.

Propose temporal RULES in DatalogMTL, using only the predicate vocabulary given above,
and return them as a single JSON object with no prose outside it:

{
  "rules": [
    {
      "name": "short_identifier",
      "phenomenon": "the named effect, as a finance reader would call it",
      "head": "myEffect(X,Y)",
      "body": "P(X), Q(Y), <+>[0,5] R(X,Y)",
      "constants": {"w": 5},
      "rationale": "one or two sentences"
    }
  ]
}

P, Q and R above are PLACEHOLDERS showing the shape only — replace them with predicates
from the vocabulary list. Choose the predicates, operators and constants yourself, from
what you see in the data.

Requirements:
- The HEAD is a NEW predicate you invent to name the effect. Do NOT reuse a vocabulary
  predicate as the head — vocabulary predicates are observations; the head is the claim
  those observations support.
- The BODY is a comma-separated list of atoms. Exactly ONE atom carries a temporal
  operator; it is the conclusion. The others, unprefixed, are the condition.
- In the body, use ONLY the predicate names from the vocabulary list, spelled exactly.
  Anything else is unscoreable and will be discarded.
- Put every constant as a NUMBER inside the body, and repeat it in "constants".
- Use the variables X and Y, not the asset names.
- Between 3 and 7 rules. Fewer good rules beat more weak ones.
- A rule whose conclusion holds nearly always anyway scores badly — prefer conditions
  that select a specific situation.
- The condition must actually OCCUR in the data. A condition that never holds cannot
  be scored, however sensible it reads.

"FOLLOWED BY" IS THE OPERATOR — there is no implication arrow. Do not write `->` or
`=>` anywhere in a body.

    WRONG   P(X), <+>[0,20] Q(X) -> R(X)
    RIGHT   P(X), Q(X), <+>[0,20] R(X)

Everything before the operator is the situation; the operator says the conclusion
follows within that many days. That is already "A, then B".

    WRONG   <+>[0,20] P(X,Y) -> Q(X,Y)
    RIGHT   P(X,Y), <+>[0,20] Q(X,Y)

Return the JSON object now."""


def propose_prompt(name_a: str, name_b: str, data_block: str, mode: str,
                   lo: str, hi: str) -> str:
    # The template catalogue is deliberately NOT shown: with it in the prompt the model
    # copied the templates (run 2026-09-24: safe haven, contagion, decoupling, lead-lag,
    # near-verbatim), which turns the comparison into templates vs. templates.
    head = PROPOSE_HEADER.format(operators=OPERATORS, vocabulary=vocabulary(),
                                 name_a=name_a, name_b=name_b)
    what = ("DAILY LOG-RETURNS for the two assets — the raw input to the pipeline"
            if mode == "series" else
            "PREDICATE FACTS already derived from the two assets")
    defs = ("\n\n" + definitions_block()) if mode == "series" else ""
    return (f"{head}\n"
            f"DATA — {what}, covering {lo} to {hi} ONLY.\n"
            f"This is the training period. Your rules will be scored on a later period "
            f"you cannot see.\n\n"
            f"{data_block}{defs}\n\n"
            f"{PROPOSE_CONTRACT}\n")


EXPLAIN_HEADER = """\
You are explaining the output of a rule-mining system to a finance audience.

Below are temporal rules in DatalogMTL, each with the scores it achieved on a held-out
period. Explain what each rule ASSERTS in plain language, and what its scores mean.

How to read the scores:
- confidence: of the days when the rule's CONDITION held, the fraction on which the
  CONCLUSION also held.
- baseline: how often the conclusion held anyway, ignoring the condition.
- lift = confidence - baseline. This is the number that matters. A confidence of 1.00
  against a baseline of 0.95 means the rule is close to restating the obvious.
- n: condition-days, the sample size.

Be accurate rather than encouraging. If a rule is near-vacuous or its effect is negative,
say so plainly. Do not describe a negative or negligible result as a finding.

RULES
{rules}
"""


def explain_prompt(rules_block: str) -> str:
    return EXPLAIN_HEADER.format(rules=rules_block)
