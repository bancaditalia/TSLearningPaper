"""
The two paper tables, generated from the registry and the result CSVs.

────────────────────────────────────────────────────────────────────────────────
WHY TWO TABLES AND NOT ONE
────────────────────────────────────────────────────────────────────────────────

They answer different questions and belong in different sections.

  TABLE 1 (§ method)   the rule LIBRARY. One row per template, no pair, no numbers.
                       "here is what we encode, and here is the finance literature that
                       names it" — the defence against the charge that the templates were
                       chosen because DatalogMTL happens to express them.
  TABLE 2 (§ results)  one row per template x pair, REJECTIONS INCLUDED.

Table 1 is built from the `@template(...)` metadata, so adding T8 updates the paper.
A hardcoded LaTeX table goes stale the first time the library changes.

────────────────────────────────────────────────────────────────────────────────
THREE CHOICES IN TABLE 2 THAT ARE ARGUMENTS, NOT FORMATTING
────────────────────────────────────────────────────────────────────────────────

1. REJECTED RULES ARE ROWS.  "sp500-nasdaq: isNegativeCorrelation never fires in 2,515
   days" is the control that licenses reading bitcoin's 0.00 as a FINDING rather than a
   broken pipeline. Drop the rejected rows and the table quietly becomes a success list —
   the exact selection effect the whole apparatus exists to avoid.

2. BASELINE AND n SIT NEXT TO EVERY CONFIDENCE.  Without them 1.00 reads as a discovery
   when it can be near-vacuous (T2/T7 on bitcoin: baseline 0.91). `n` also carries the
   eligibility rule — under MIN_CONDITION_DAYS the cell is blank, NOT zero, because a
   confidence measured on 12 days is absent evidence, not evidence of absence.

3. TRAIN AND TEST ARE TWO COLUMNS.  The degradation is itself a result (a pair's
   relationship drifts). Collapsing it to one number invites the reviewer to assume
   in-sample fitting, which is precisely what the frozen-constant protocol rules out.

The verification column (|delta| between the Python scorer and the reasoner) is what makes
the engine-agreement claim visible instead of asserted in prose. A column of 0.000 costs
one column and settles the question.
"""
from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

from tslearning.config import CFG, PATHS
from tslearning.rules.scoring import MIN_CONDITION_DAYS

# Display names. Anything not listed falls back to a readable guess, so a new pair still
# renders — it just may want an entry here to look right in print.
# PLAIN TEXT — no LaTeX escapes here. Escaping belongs in to_latex(), so the CSV stays
# machine-readable. (An earlier version stored "S\\&P 500", which leaked the backslash
# into rule_results.csv and broke a plain `Pair.str.contains("S&P")` filter.)
SERIES_LABEL = {
    "sp500": "S&P 500", "nasdaq": "NASDAQ", "gold": "Gold",
    "bitcoin": "Bitcoin", "vix": "VIX", "ust10y": "US 10y Treasury",
    "tlt": "US Treasury 20y+ (TLT)",
    "ust10z": "US 10y zero-coupon (GSW)",
}
# ── The symbol registry — ONE place a fitted constant is named ────────────────
# Keyed by the key used in `Rule.constants` (and therefore by the `const_*` column in
# rules.csv, and by `fits=` on the template). Everything that prints a constant reads
# from here: the schema in Table 1, the Fitted column in Table 1, the Constants column
# in Table 2. Writing the symbol out by hand in any of those three is how `d_neg` came
# to print as `d1` in one column and `d_neg` in the next.
# `check_symbols()` below enforces it, so the drift cannot come back silently.
SYMBOL = {"w": "w", "k": "k", "d": "d", "d_neg": "d⁻", "a": "a", "b": "b"}

# Print order for Table 2's Constants cell.
CONST_ORDER = ["w", "k", "d", "d_neg", "a", "b"]
CONST_LABEL = [(f"const_{key}", SYMBOL[key]) for key in CONST_ORDER]

# A lift this small is noise at our sample sizes; a baseline this high makes any
# confidence near-vacuous regardless of lift.
LIFT_NEGLIGIBLE = 0.05
BASELINE_VACUOUS = 0.85


def _series_label(name: str) -> str:
    """'us_inflation' -> 'US Inflation'; 'sp500' -> 'S&P 500'."""
    if name in SERIES_LABEL:
        return SERIES_LABEL[name]
    bits = name.split("_")
    # country codes stay upper-case, the rest is title-cased
    return " ".join(b.upper() if len(b) == 2 or b == "us" else b.title() for b in bits)


def _split_pair(pair: str) -> "list[str]":
    """Split a pair slug into its two series, tolerating multi-token names.

    'sp500_gold' is unambiguous, but 'us_inflation_us_unemployment' is not: splitting on
    '_' gives four tokens and renders as 'Us–Inflation–Us–Unemployment'. Match against the
    configured series instead, longest first, so multi-token names stay whole.
    """
    from tslearning.config import CFG

    known = sorted((s.lower() for s in CFG.get("series_freq", {})), key=len, reverse=True)
    p = str(pair).lower()
    for k in known:
        if p.startswith(k + "_"):
            rest = p[len(k) + 1:]
            if rest in known or "_" not in rest:
                return [k, rest]
    return p.split("_", 1) if "_" in p else [p]


def pair_label(pair: str) -> str:
    if " – " in str(pair):                 # already a display label
        return str(pair)
    return " – ".join(_series_label(s) for s in _split_pair(pair))


def _cell(v, p: int = 2, sign: bool = False) -> str:
    """Half-up rounding, so a figure never disagrees with the same one rounded by hand."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    q = Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-p), ROUND_HALF_UP)
    return f"+{q}" if sign and q > 0 else str(q)


def template_label(name: str, short: bool = False) -> str:
    """'T1_safeHaven' -> 'T1 safe haven', or just 'T1' when short=True."""
    num, _, rest = str(name).partition("_")
    if short:
        return num
    words = re.sub(r"(?<!^)(?=[A-Z])", " ", rest).lower()
    return f"{num} {words}".strip()


# ══════════════════════════════════════════════════════════════════════════════
#  Table 1 — the rule library
# ══════════════════════════════════════════════════════════════════════════════

def check_symbols(templates=None) -> "list[str]":
    """Every fitted constant must be in SYMBOL and appear under that name in the schema.

    This is the guard for the class of bug where a constant is called `d_neg` in the code,
    `d1` in the rule body and `d⁻` in the results table — three names for one number, and
    nothing to notice it but a reader.
    """
    from tslearning.rules.templates import TEMPLATES

    problems = []
    for n in sorted(templates or TEMPLATES):
        meta = TEMPLATES.get(n) or {}
        schema = meta.get("schema") or ""
        for key in meta.get("fits") or []:
            sym = SYMBOL.get(key)
            if sym is None:
                problems.append(f"{n}: fits '{key}', which has no entry in SYMBOL")
            elif schema and not re.search(rf"(?<![A-Za-z]){re.escape(sym)}", schema):
                problems.append(
                    f"{n}: fits '{key}' (prints as '{sym}') but the schema does not "
                    f"use that symbol: {schema}")
    return problems


# Map the prose cite strings from @template metadata to BibTeX keys.
# This is the single place that connects templates.py cite= to references.md @article{}.
CITE_TO_BIBTEX = {
    "Baur & Lucey 2010; Baur & McDermott 2010":
        "baur2010hedge,baur2010safehaven",
    "Longin & Solnik 2001; Bae, Karolyi & Stulz 2003":
        "longin2001extreme",
    "converse of T2 (Ang & Chen 2002)":
        "ang2002asymmetric",
    "Hasbrouck 1995; de Jong & Nijman 1997":
        "hasbrouck1995onesecurity,dejong1997highfrequency",
    "Baur & Lucey 2010 (hedge vs. haven)":
        "baur2010hedge",
    "Ang & Bekaert 2002 (regime switching)":
        "ang2002regimeshifts",
    "Bekaert, Hoerova & Lo Duca 2013":
        "bekaert2013risk",
}


# Hand-written LaTeX schemas — the plain-text versions in templates.py are for terminals;
# these are for the paper. Fill in the strings you want to appear in the table.
SCHEMA_LATEX = {
    "T1_safeHaven":        r"$\mathit{highVol}(X), \mathit{trendDown}(X), \diamondplus_{[0,w]} \mathit{isNegativeCorrelation(X,Y)} \to \mathit{safeHaven}(X,Y)$",
    "T1b_safeHaven_bw":    r"$\mathit{highVol}(X), \mathit{trendDown}(X) \to \mathit{stressedDown}(X)$; $\mathit{isNegativeCorrelation}(X,Y), \diamondminus_{[0,w]} \mathit{stressedDown}(X) \to \mathit{safeHavenBw}(X,Y)$",
    "T3b_decoupled_bw":    r"$\mathit{lowVol}(X), \mathit{lowVol}(Y) \to \mathit{bothCalm}(X,Y)$; $\mathit{isNoCorrelation}(X,Y), \diamondminus_{[0,w]} \mathit{bothCalm}(X,Y) \to \mathit{decoupledBw}(X,Y)$",
    "T2_contagion":    r"$\mathit{tailDependent}(X,Y), \diamondplus_{[0,w]} \mathit{isPositiveCorrelation}(X,Y) \to \mathit{contagion}(X,Y)$",
    "T2b_contagion_bw":    r"$\mathit{isPositiveCorrelation}(X,Y), \diamondminus_{[0,w]} \mathit{tailDependent}(X,Y) \to \mathit{contagionBw}(X,Y)$",
    "T4b_leadsUnderStress_bw": r"$\mathit{leadsPositively}(X,Y,k), \diamondminus_{[0,w]} \mathit{highVol}(X) \to \mathit{leadsUnderStressBw}(X,Y,k)$",
    "T7b_macroConditioned_bw": r"$\mathit{isPositiveCorrelation}(X,Y), \mathit{market}(X,C), \diamondminus_{[0,w]} \mathit{inflationHigh}(C) \to \mathit{macroConditionedBw}(X,Y)$",
    "T3_decoupled":        r"$\mathit{lowVol}(X), \mathit{lowVol}(Y), \diamondplus_{[0,w]} \mathit{isNoCorrelation}(X,Y) \to \mathit{decoupled}(X,Y)$",
    "T4_leadsUnderStress": r"$\mathit{highVol}(X), \diamondplus_{[0,w]} \mathit{leadsPositively}(X,Y,k) \to \mathit{leadsUnderStress}(X,Y,k)$",
    "T5_hedgeHorizon":     r"$\mathit{isNegativeCorrelation}(X,Y), \boxplus_{[0,d]} \mathit{isNegativeCorrelation}(X,Y) \to \mathit{hedgeHorizon}(X,Y)$",
    "T6_regimeReversal":   r"$\boxminus_{[0, d^{-}]}\mathit{isNegativeCorrelation}(X,Y), \boxplus_{[a,b]}\mathit{isPositiveCorrelation}(X,Y) \to \mathit{regimeReversal}(X,Y)$",
    "T7_macroConditioned": r"$\mathit{market}(X,C), \mathit{inflationHigh}(C), \diamondplus_{[0,w]} \mathit{isPositiveCorrelation}(X,Y) \to \mathit{macroConditioned}(X,Y)$",
}


def _schema_latex(template_name: str, plain_schema: str) -> str:
    """Return the hand-written LaTeX schema if available, else the plain-text one."""
    latex = SCHEMA_LATEX.get(template_name)
    if latex:
        return latex
    return plain_schema


def _cite_latex(cite_str: str) -> str:
    """Convert a prose cite string to \\cite{key1,key2} for LaTeX."""
    keys = CITE_TO_BIBTEX.get(cite_str)
    if keys:
        return f"\\cite{{{keys}}}"
    # fallback: return the prose string as-is (will show in the table)
    return cite_str


# Fitted-constant symbols in math mode, for the boxed library
_SYMBOL_MATH = {"w": "w", "k": "k", "d": "d", "d_neg": "d^{-}", "a": "a", "b": "b"}


def _template_sort_key(name: str):
    """T1, T1b, T2, T2b, ...: numeric order, backward variant after its forward one."""
    m = re.match(r"T(\d+)(b?)", name)
    return (int(m.group(1)), m.group(2)) if m else (999, name)


# Longest rule (in visible characters) that fits on one line of the box at \small in an
# LNCS text block; longer rules are broken before the arrow.
_BOX_MAX_CHARS = 72


def _visible_len(latex: str) -> int:
    """Rough rendered length of a schema formula, for the line-break decision."""
    s = re.sub(r"\\mathit\{([^}]*)\}", r"\1", latex)
    s = re.sub(r"\\(diamondplus|diamondminus|boxplus|boxminus)", "O", s)
    s = re.sub(r"_\{[^}]*\}", "_[0,w]", s)
    return len(s.replace(r"\to", ">").replace("$", ""))


def library_box(templates=None, caption: str = "", label: str = "") -> str:
    r"""The rule library as a framed box instead of a (rotated) table.

    One entry per template:
        T1  Safe haven [cite]                     (fitted constants: w)
            rule                                   <- one line per rule,
            rule                                      auxiliary rules first
    Rules are the hand-written LaTeX schemas (SCHEMA_LATEX), split on "; ".
    """
    from tslearning.rules.templates import TEMPLATES

    names = sorted([n for n in (templates or TEMPLATES) if n in TEMPLATES],
                   key=_template_sort_key)
    lines = []
    for i, n in enumerate(names):
        meta = TEMPLATES[n]
        code = n.split("_")[0]
        phen = (meta.get("phenomenon") or template_label(n)).replace("&", r"\&")
        cite = _cite_latex(meta.get("cite") or "")
        cite = "~" + cite if cite.startswith("\\cite") else ""
        fits = ", ".join(f"${_SYMBOL_MATH.get(c, c)}$" for c in (meta.get("fits") or []))
        head = (rf"\textbf{{{code}}}\quad {phen}{cite}"
                + (rf"\hspace{{1em plus 1fill}}{{\footnotesize (fitted constants: {fits})}}" if fits else ""))
        schema = SCHEMA_LATEX.get(n)
        rules = [r.strip() for r in schema.split("$; $")] if schema else \
                [meta.get("schema") or ""]
        rules = [("" if r.startswith("$") else "$") + r + ("" if r.endswith("$") else "$")
                 for r in rules]
        # A rule never wraps mid-formula: a long one is broken before "→ head" and the
        # head goes on its own, further-indented line (LNCS text width ~12 cm).
        split_rules = []
        for r in rules:
            if _visible_len(r) > _BOX_MAX_CHARS and r"\to" in r:
                body, headr = r.rsplit(r"\to", 1)
                split_rules.append([rf"\mbox{{{body.rstrip()}$}}",
                                    rf"\hspace*{{3em}}\mbox{{$\to{headr}}}"])
            else:
                split_rules.append([rf"\mbox{{{r}}}"])
        lines.append(head + r"\\")
        for j, parts in enumerate(split_rules):
            end_entry = j == len(split_rules) - 1
            for q, part in enumerate(parts):
                last_line = end_entry and q == len(parts) - 1
                gap = "" if (last_line and i == len(names) - 1) else \
                      (r"\\[5pt]" if last_line else r"\\")
                lines.append(rf"\hspace*{{1.5em}}{part}{gap}")
    return _run_tag() + "\n".join([
        r"\begin{figure}[t]", r"\centering",
        r"\fbox{\begin{minipage}{0.95\textwidth}\small",
        *lines,
        r"\end{minipage}}",
        rf"\caption{{\small{{{periods(caption)}}}}}", rf"\label{{{label}}}", r"\end{figure}"])


def library_table(templates=None, strict: bool = True,
                  latex_cites: bool = False) -> pd.DataFrame:
    """One row per registered template, from the @template metadata. No data involved.

    Parameters
    ----------
    latex_cites : bool
        If True, the Source column uses ``\\cite{key}`` instead of prose references.
    """
    from tslearning.rules.templates import TEMPLATES

    problems = check_symbols(templates)
    if problems and strict:
        raise ValueError("constant symbols disagree between schema and fits:\n  "
                         + "\n  ".join(problems))

    names = sorted(templates or TEMPLATES)
    rows = []
    for n in names:
        meta = TEMPLATES.get(n)
        if meta is None:
            continue
        cite_raw = meta.get("cite") or ""
        schema_plain = meta.get("schema") or ""
        rows.append({
            "#": n.split("_")[0],
            "Phenomenon": meta.get("phenomenon") or template_label(n),
            "DatalogMTL body (schema)": (
                _schema_latex(n, schema_plain) if latex_cites else schema_plain),
            # rendered through SYMBOL, never printed as the raw dict key
            "Fitted": ", ".join(SYMBOL.get(c, c) for c in (meta.get("fits") or [])) or "—",
            "Source": _cite_latex(cite_raw) if latex_cites else cite_raw,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  Table 2 — results
# ══════════════════════════════════════════════════════════════════════════════

def _constants(row) -> str:
    bits = [f"{lab}={int(row[col])}" for col, lab in CONST_LABEL
            if col in row.index and pd.notna(row[col])]
    return ", ".join(bits) or "—"


def _conclusion_absent(row) -> bool:
    """Baseline exactly 0: the conclusion never holds on the test period, so confidence
    and baseline are both 0 by construction and the rule measures nothing."""
    b = row.get("test_baseline")
    return b is not None and not pd.isna(b) and float(b) == 0.0


def _verdict(row) -> str:
    """One-word reading. Uses CI when available, falls back to lift direction."""
    if not row.get("accepted", False):
        return f"not fitted: {row.get('reason') or 'unknown'}"
    if _conclusion_absent(row):
        return "ineligible (conclusion never holds on test)"

    n = row.get("test_condition_days")
    if pd.isna(n) or not row.get("test_eligible", True):
        why = row.get("test_reason")
        if pd.isna(why) or not why:
            # persistence/transition templates gate on episodes, not days, and do not
            # always record a reason — say which count fell short rather than "nan"
            why = (f"n = {int(n)} < {MIN_CONDITION_DAYS} condition days"
                   if pd.notna(n) else "too few episodes on the test slice")
        return f"ineligible ({why})"

    lift, base = row.get("test_lift"), row.get("test_baseline")
    if pd.isna(lift):
        return "not scored"
    # Order matters: a high baseline disqualifies a rule even when the lift looks fine,
    # because the conclusion predicate was almost always true anyway.
    if pd.notna(base) and base > BASELINE_VACUOUS and lift < 2 * LIFT_NEGLIGIBLE:
        return "near-vacuous (baseline high)"
    if lift <= -LIFT_NEGLIGIBLE:
        return "negative"

    # If CI is available, use it to distinguish significant from merely directional
    spans_zero = row.get("ci_spans_zero")
    # A ZERO-WIDTH interval is the bootstrap failing at a boundary (every resample gives
    # the same value), not a precise estimate — it can never make a rule significant.
    # Same rule as multiple_testing.py. (Found 2026-10-16: classifier T2 gold/bitcoin.)
    degenerate = row.get("ci_degenerate")
    degenerate = bool(degenerate) if pd.notna(degenerate) else False
    if lift >= LIFT_NEGLIGIBLE:
        if pd.notna(spans_zero) and not spans_zero and not degenerate:
            return "significant"
        return "positive lift"
    return "negligible"


def results_table(rules_csv=None, verification_csv=None, pairs=None,
                  templates=None, period: str = "test",
                  scored_only: bool = False) -> pd.DataFrame:
    """Table 2 from `results/rules.csv` (+ `results/verification.csv` if present).

    Rejected rules are KEPT — see the module docstring. Ineligible scores are blanked
    rather than shown as 0.00.
    """
    res = Path(PATHS["results"])
    rules_csv = Path(rules_csv or res / "rules.csv")
    verification_csv = Path(verification_csv or res / "verification.csv")
    if not rules_csv.exists():
        raise FileNotFoundError(
            f"{rules_csv} not found — run scripts/generate_rules.py --score first")

    df = pd.read_csv(rules_csv)
    if pairs:
        df = df[df.pair.isin(pairs)]
    if templates:
        df = df[df.template.isin(templates)]
    if df.empty:
        return pd.DataFrame()

    # |delta| python vs. vadalog, keyed on (template, pair) for the requested period
    delta = {}
    if verification_csv.exists():
        v = pd.read_csv(verification_csv)
        v = v[v.period == period] if "period" in v.columns else v
        for _, r in v.iterrows():
            # ONLY the engine's own comparison (agreement_abs_diff = |Python - Vadalog|,
            # set when the reasoner ran). There used to be a fallback to
            # |scored - python_confidence|, two Python numbers: when every engine call
            # failed (2026-10-27, "Is a directory"), that fallback gave Δ = 0 and the
            # caption then claimed the reasoner had reproduced every rule. No engine
            # number -> no Δ, and no engine sentence in the caption.
            d = r.get("agreement_abs_diff")
            if pd.notna(r.get("vadalog_confidence")) and d is not None \
                    and not (isinstance(d, float) and pd.isna(d)):
                delta[(r["template"], r["pair"])] = float(d)

    def cell(v, p=2):
        # Half-UP, not Python's default half-even: a baseline of 0.625 printed as "0.62"
        # contradicts the same figure rounded by hand anywhere else in the paper, and
        # "banker's rounding" is not an explanation a reviewer should have to be given.
        return "—" if pd.isna(v) else str(
            Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-p), ROUND_HALF_UP))

    rows = []
    for _, r in df.sort_values(["template", "pair"]).iterrows():
        ok = bool(r.get("accepted", False))
        elig = ok and bool(r.get("test_eligible", True)) \
            and pd.notna(r.get("test_condition_days")) and not _conclusion_absent(r)
        n = r.get("test_condition_days")
        d = delta.get((r["template"], r["pair"]))
        # CI column: [lo, hi] when available
        ci_lo, ci_hi = r.get("ci_ci_lo"), r.get("ci_ci_hi")
        if elig and pd.notna(ci_lo) and pd.notna(ci_hi):
            ci_str = f"[{cell(ci_lo, 3)}, {cell(ci_hi, 3)}]"
        else:
            ci_str = "—"

        rows.append({
            "Rule": template_label(r["template"], short=True),
            "Pair": pair_label(r["pair"]),
            "Constants": _constants(r) if ok else "—",
            # train confidence is shown even when the test slice is ineligible: it is what
            # the constants were fitted on, and its absence would hide the fit entirely
            "Train": cell(r.get("train_confidence")) if ok else "—",
            "Test": cell(r.get("test_confidence")) if elig else "—",
            "Base": cell(r.get("test_baseline")) if elig else "—",
            "Lift": cell(r.get("test_lift"), 3) if elig else "—",
            "CI": ci_str,
            "n": "—" if not elig or pd.isna(n) else str(int(n)),
            "Δ": "—" if d is None else f"{d:.3f}",
            "Verdict": _verdict(r),
        })
    out = pd.DataFrame(rows)
    # Drop the verification column entirely when nothing was verified — an all-dash
    # column reads as a failed check rather than a check that was never run.
    if (out["Δ"] == "—").all():
        out = out.drop(columns=["Δ"])

    # SCORED ROWS FIRST. Rejections stay (dropping them turns the table into a success
    # list — the NASDAQ row *is* the control), but burying 11 scored rules under 31 rows
    # of dashes makes the exhibit unreadable. Order, don't hide.
    out = out.assign(_s=(out["Test"] != "—").astype(int)) \
             .sort_values(["_s", "Rule", "Pair"], ascending=[False, True, True]) \
             .drop(columns=["_s"]).reset_index(drop=True)
    if scored_only:
        out = out[out["Test"] != "—"].reset_index(drop=True)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Rendering
# ══════════════════════════════════════════════════════════════════════════════

def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no rows)_"
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _run_tag() -> str:
    """A LaTeX comment naming the run, at the top of every table a named run writes
    (invisible in the PDF; the base run's tables are unchanged)."""
    from tslearning.config import BASE_RUN, run_summary
    return "" if CFG["run"]["name"] == BASE_RUN else f"% {run_summary()}\n"


def periods(text: str) -> str:
    """Fill the <TRAIN> / <TEST> placeholders of a caption with the run's periods
    (config split), as years: "2015--2021". Captions are written with placeholders so a
    run on another period never prints the paper's years."""
    def span(p):
        a, b = str(p[0])[:4], str(p[1])[:4]
        return a if a == b else f"{a}--{b}"
    return (text.replace("<TRAIN>", span(CFG["split"]["train"]))
                .replace("<TEST>", span(CFG["split"]["test"])))


def to_latex(df: pd.DataFrame, caption: str = "", label: str = "",
             *, rotate: bool = False, fontsize: str = "") -> str:
    """Render a DataFrame as a LaTeX figure (booktabs).

    Parameters
    ----------
    rotate : bool
        Wrap the tabular in ``\\rotatebox{90}{…}`` for landscape orientation.
    fontsize : str
        LaTeX font-size command WITHOUT the backslash, e.g. ``"footnotesize"``.
        Applied inside the figure before the tabular.
    """
    caption = periods(caption)
    if df.empty:
        return "% (no rows)"
    cols = list(df.columns)
    # left-align text, right-align the numeric columns
    numeric = {"Train", "Test", "Base", "Lift", "CI", "n", "Δ"}
    spec = "".join("r" if c in numeric else "l" for c in cols)

    def esc(v):
        s = str(v)
        # Pass through raw LaTeX (e.g. \cite{...} or $\diamondplus_{...}$)
        if s.startswith("\\") or s.startswith("$"):
            return s
        # temporal operators as written by an LLM ("<+>[5,10]") -> the paper's notation
        s = re.sub(r"<([+-])>\[(-?\d+),(-?\d+)\]",
                   lambda m: (r"$\diamondplus_{[%s,%s]}$" if m.group(1) == "+"
                              else r"$\diamondminus_{[%s,%s]}$") % (m.group(2), m.group(3)), s)
        if "$" in s:
            # math segments (odd positions after splitting on $) pass through untouched;
            # only the text between them is escaped
            parts = s.split("$")
            return "$".join(p if i % 2 else _esc_text(p) for i, p in enumerate(parts))
        return _esc_text(s)

    def _esc_text(s):
        s = s.replace("–", "--").replace("—", "--")
        for ch in ("_", "%", "#"):
            s = s.replace(ch, "\\" + ch)
        s = s.replace("&", "\\&") if "\\&" not in s else s
        s = s.replace("<", "$<$").replace(">", "$>$")
        s = s.replace("Δ", "$|\\Delta|$")
        s = s.replace("×", "$\\times$")         # speed-up "282,006×" (not in T1 fonts)
        # superscript-minus constants (d⁻) -> proper math mode
        s = re.sub(r"([A-Za-z])⁻", r"$\1^-$", s)
        return s

    tab_lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
                 " & ".join(esc(c) for c in cols) + " \\\\", "\\midrule"]
    tab_lines += [" & ".join(esc(v) for v in r) + " \\\\"
                  for r in df.itertuples(index=False)]
    tab_lines += ["\\bottomrule", "\\end{tabular}"]
    tab_block = "\n".join(tab_lines)

    if rotate:
        # Use sidewaysfigure (from the rotating package) so the caption
        # rotates with the table and nothing is cut off the page.
        lines = ["\\begin{sidewaysfigure}", "\\centering"]
        if fontsize:
            lines.append(f"\\{fontsize}")
        lines.append(tab_block)
        if caption:
            lines.append(f"\\caption{{\\small{{{caption}}}}}")
        if label:
            lines.append(f"\\label{{{label}}}")
        lines.append("\\end{sidewaysfigure}")
        return _run_tag() + "\n".join(lines)

    lines = ["\\begin{figure}[t]", "\\centering"]
    if fontsize:
        lines.append(f"\\{fontsize}")
    lines.append(f"\\resizebox{{\\textwidth}}{{!}}{{%")
    lines.append(tab_block)
    lines.append("}")  # close resizebox
    if caption:
        lines.append(f"\\caption{{\\small{{{caption}}}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append("\\end{figure}")
    return _run_tag() + "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Table 3 — proposer comparison (template fitter vs. LLM)
# ══════════════════════════════════════════════════════════════════════════════
#
# Both proposers are scored by the SAME code on the SAME held-out period, so the two sets
# of rows are directly comparable. What the table is for is not "who wins on the best
# rule" — it is the SPREAD. A proposer whose best rule is excellent and whose median rule
# has negative lift has not produced knowledge; it has produced candidates that something
# else had to sort.

def _llm_source(f: Path) -> str:
    """The label source a run was scored on (meta.source; older runs default to py)."""
    import json
    try:
        return (json.loads(f.read_text()).get("meta") or {}).get("source") or "py"
    except Exception:
        return "?"


def _pick_llm_files(results_dir: Path, llm_run: "str | None" = None,
                    source: str = "py") -> "list[Path]":
    """Select which llm_*.json files to use.

    Only runs scored on `source` are kept — the comparison puts them next to template
    rules from rules.csv, i.e. the Python labeller. A run on another label source would be
    a different comparison; the skipped files are listed in `_pick_llm_files.skipped`.

    * llm_run=None   → for each (pair, provider, model) keep only the newest file.
                        Timestamped names always beat legacy ``_run_NN`` names (they
                        are newer by construction); within each kind, the highest
                        timestamp / run number wins. Plain alphabetical order would get
                        this wrong: "run_01" sorts AFTER "20260924_110120".
    * llm_run=<str>  → keep only files whose name contains the string (a timestamp
                        like ``20260922_143012``, or ``run_01`` for legacy files).
    """
    all_files = sorted(results_dir.glob("llm_*.json"))
    _pick_llm_files.skipped = [f for f in all_files if _llm_source(f) != source]
    all_files = [f for f in all_files if _llm_source(f) == source]
    if not all_files:
        return []

    if llm_run:
        return [f for f in all_files if llm_run in f.stem]

    # group by everything before the last _YYYYMMDD_HHMMSS or _run_NN segment (all files
    # here share one source, so "latest per pair and model" can't pick another source's run)
    import re
    tag_re = re.compile(r"^(.+?)_(?:(\d{8}_\d{6})|run_(\d+))$")
    groups: dict[str, list[tuple]] = {}
    for f in all_files:
        m = tag_re.match(f.stem)
        if m:
            tag, ts, run_n = m.groups()
            # (1, timestamp) outranks any (0, run number)
            key = (1, ts) if ts else (0, f"{int(run_n):06d}")
        else:
            tag, key = f.stem, (0, "")
        groups.setdefault(tag, []).append((key, f))
    out = []
    for entries in groups.values():
        latest = max(entries, key=lambda e: e[0])[1]
        if not _llm_pool_settings()["pool"]:
            out.append(latest)
            continue
        # POOL: every run of this pair/model made with the same prompt as the latest
        # one — repeated sampling of one experiment
        h = _llm_meta(latest).get("prompt_hash")
        out += [f for _k, f in sorted(entries, key=lambda e: e[0])
                if h is not None and _llm_meta(f).get("prompt_hash") == h] or [latest]
    return out


def _llm_meta(f: Path) -> dict:
    import json
    try:
        return json.loads(f.read_text()).get("meta") or {}
    except Exception:
        return {}


def _llm_pool_settings() -> dict:
    """paper.llm in the config: pool the runs of one prompt, keep the top_k by TRAIN lift."""
    from tslearning.config import CFG
    c = (CFG.get("paper") or {}).get("llm") or {}
    return {"pool": bool(c.get("pool", True)), "top_k": int(c.get("top_k", 5)),
            "box_top_k": int(c["box_top_k"]) if c.get("box_top_k") else None}


_LLM_NOTED: set = set()          # run files already warned about in this process


def _warn_if_llm_stale(f: Path) -> None:
    """LLM rules are scored ONCE, when the model answers; template rules are rescored at
    every step 8. A run older than the pair's current Python labels was scored — and, in
    series mode, prompted — on labels that no longer exist."""
    import json
    if f.name in _LLM_NOTED:
        return
    _LLM_NOTED.add(f.name)
    try:
        pair = (json.loads(f.read_text()).get("meta") or {}).get("pair")
    except Exception:
        return
    d = PATHS["data_processed"] / "real_pairs_regime_py" / str(pair)
    csvs = list(d.glob("*.csv")) if d.exists() else []
    if csvs and f.stat().st_mtime < max(c.stat().st_mtime for c in csvs):
        print(f"WARNING: {f.name} is older than the current {pair} labels — its rules were "
              f"scored on labels that have changed since. Rerun llm_rules.py.")


def llm_rows(results_dir=None, pairs=None,
             llm_run: "str | None" = None) -> "list[dict]":
    """Every scored rule from the selected LLM run(s) in the results folder.

    Rules are deduplicated by (pair, body, constants) — repeated runs proposing the same
    rule produce one row, not one per file. With pooling on (config paper.llm), only the
    top_k rules by TRAIN lift are kept for the table: they are chosen on the training
    period, like the templates' constants, never on the test period. Each unique rule gets
    a stable ID (L1, L2, …), usable as a cross-reference to :func:`llm_rule_listing`.

    Parameters
    ----------
    llm_run : str or None
        Substring filter on filename.  ``None`` picks the latest run per
        (pair, provider, model) group.
    """
    import json

    results_dir = Path(results_dir or PATHS["results"])
    seen: dict[tuple, dict] = {}          # (pair, template, constants_str) -> row
    seen_body: set[tuple] = set()         # dedup key for ALL rules (incl. ineligible)
    rule_bodies: list[dict] = []          # for the listing; includes ineligible rules
    unscored: list[dict] = []             # accepted but not scorable on test
    idx = 0
    candidates: list[dict] = []           # every unique parsed rule, with its train lift
    counts = {"runs": 0, "proposed": 0, "parsed": 0}
    per_model: dict[str, dict] = {}       # who -> {"runs": n, "temps": [...]}, for the box
    picked = _pick_llm_files(results_dir, llm_run)
    for f in getattr(_pick_llm_files, "skipped", []):
        if (llm_run is None or llm_run in f.stem) and f.name not in _LLM_NOTED:
            _LLM_NOTED.add(f.name)
            print(f"note: {f.name} skipped — scored on '{_llm_source(f)}' labels, the "
                  f"comparison uses the Python labeller's (rules.csv)")
    for f in picked:
        _warn_if_llm_stale(f)
    for f in picked:
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        meta = d.get("meta") or {}
        pair = meta.get("pair")
        if not pair or (pairs and pair not in pairs):
            continue
        # a hand-written fixture is not an experiment; keep it out of the results
        if meta.get("provider") == "file":
            continue
        who = f"LLM {meta.get('model') or meta.get('provider') or '?'}"
        counts["runs"] += 1
        pm = per_model.setdefault(who, {"runs": 0, "temps": []})
        pm["runs"] += 1
        pm["temps"].append(meta.get("temperature"))
        counts["proposed"] += int(meta.get("n_proposed") or 0)
        counts["parsed"] += len(d.get("accepted") or [])
        for a in d.get("accepted") or []:
            s = (a.get("scores") or {}).get("test") or {}
            cstr = ", ".join(f"{k}={v}" for k, v in
                             (a.get("constants") or {}).items()) or "—"
            # the BODY identifies a rule; names repeat across runs ("myEffect")
            key = (who, pair, " ".join(str(a.get("body", "")).split()), cstr)
            # Constants column: the window exactly as the model wrote it in the body
            # (e.g. "<+>[5,10]"), plus the lag if there is one — not our w / w_lo encoding
            import re as _re
            ops = _re.findall(r"(?:<[+-]>|\[[+-]\])\s*[\[(]\s*-?\d+\s*,\s*-?\d+\s*[\])]",
                              str(a.get("body", "")))
            k = (a.get("constants") or {}).get("k")
            cstr = ", ".join([o.replace(" ", "") for o in ops]
                             + ([f"k={k}"] if k is not None else [])) or cstr
            if key in seen_body:
                continue
            idx += 1
            lid = f"L{idx}"
            seen_body.add(key)
            # Use the JSON head+body (variables) rather than the grounded scored rule
            head = a.get("head", "?").replace("{", "").replace("}", "")
            body = a.get("body", "?").replace("{", "").replace("}", "")
            # the same head checks as propose.to_rules, applied here too so that runs
            # saved before they existed are listed the same way
            from tslearning.config import CFG
            from tslearning.llm.propose import head_reuses_vocabulary, normalise_head
            head, unbound = normalise_head(head, body)
            reused = head_reuses_vocabulary(
                head, set(CFG["predicates"]["kind"]) | {"market"})
            flags = ([f"unbound head variable dropped: {', '.join(unbound)}"] if unbound
                     else []) + ([f"head reuses vocabulary predicate {reused}"] if reused
                                 else [])
            train_lift = ((a.get("scores") or {}).get("train") or {}).get("lift")
            rule_bodies.append({
                "id": lid, "who": who,
                "rule": f"{head} :- {body}",
                "flags": flags,
                "phenomenon": (a.get("diagnostics") or {}).get("phenomenon", ""),
                "eligible": s.get("lift") is not None and bool(s.get("baseline")),
                "train_lift": train_lift, "key": key,
            })
            # Unscorable on test: its condition never holds (no lift), or its conclusion
            # never holds (baseline 0 — confidence and baseline are both 0 by
            # construction, so "lift 0.000" would measure nothing). Keep it in the
            # table, listed last, instead of dropping it silently or scoring it.
            if s.get("lift") is None or not s.get("baseline"):
                unscored.append({
                    "Proposer": who, "Pair": pair, "Rule": lid, "Constants": cstr,
                    "Train": ((a.get("scores") or {}).get("train") or {}).get("confidence"),
                    "Test": None, "Base": None, "Lift": None,
                    "n": s.get("condition_days"), "input": meta.get("input_mode"),
                    "run": f.name, "reason": s.get("reason"), "_body": body})
                continue
            seen[key] = {
                "Proposer": who, "Pair": pair,
                "Rule": lid,
                "Constants": cstr,
                "Train": ((a.get("scores") or {}).get("train") or {}).get("confidence"),
                "Test": s.get("confidence"), "Base": s.get("baseline"),
                "Lift": s.get("lift"), "n": s.get("condition_days"),
                "input": meta.get("input_mode"), "run": f.name,
                # for the comparison's CI / Verdict / About-pair columns
                "_a": a, "_src": meta.get("source", "py"), "_body": body,
            }
    # POOLED SELECTION — top_k by train lift, chosen without looking at the test period
    st = _llm_pool_settings()
    ranked = sorted((b for b in rule_bodies if b["train_lift"] is not None),
                    key=lambda b: b["train_lift"], reverse=True)
    if st["pool"]:
        # top_k PER PROPOSER (model): one model's rules must not crowd out another's
        top = []
        for w in dict.fromkeys(b["who"] for b in ranked):
            top += [b for b in ranked if b["who"] == w][:st["top_k"]]
        keep = {(b["who"], b["key"]) for b in top}
        keep_ids = {b["id"] for b in top}
        for b in rule_bodies:
            b["selected"] = (b["who"], b["key"]) in keep
        rows = [r for k, r in seen.items() if (r["Proposer"], k) in keep]
        unscored = [u for u in unscored if u["Rule"] in keep_ids]
    else:
        for b in rule_bodies:
            b["selected"] = True
        rows = list(seen.values())
    counts.update(unique=len(rule_bodies), scorable_on_train=len(ranked),
                  selected=sum(b["selected"] for b in rule_bodies),
                  pooled=st["pool"], top_k=st["top_k"])

    # stash the listing, the unscorable rules and the counts for callers that need them
    llm_rows._bodies = rule_bodies
    llm_rows._per_model = per_model
    llm_rows._unscored = unscored
    llm_rows._counts = counts
    return rows


# how an LLM rule that could not be scored on the test period is shown in Table 3
LLM_UNSCORED_LABEL = "ineligible"


def llm_rule_listing(results_dir=None, pairs=None,
                     llm_run: "str | None" = None) -> str:
    """Plain-text listing of LLM rules with their L-ids.

    Always rescans with the given arguments, so the listing is built from exactly the
    same LLM run(s) as the comparison table.
    """
    llm_rows(results_dir, pairs, llm_run=llm_run)
    c = llm_rows._counts
    lines = [f"# {c['runs']} run(s), {c['proposed']} proposed, {c['parsed']} parsed, "
             f"{c['unique']} unique, {c['scorable_on_train']} scorable on train"
             + (f"; * = the top {c['top_k']} by TRAIN lift per model, shown in the comparison"
                if c["pooled"] else "")]
    for b in llm_rows._bodies:
        tl = "   n/a" if b["train_lift"] is None else f"{b['train_lift']:+.3f}"
        lines.append(f"{'*' if b.get('selected') else ' '} {b['id']:<4} "
                     f"{b['who'].removeprefix('LLM '):<18} train lift {tl}  {b['rule']}"
                     + (f"   [{'; '.join(b['flags'])}]" if b.get("flags") else ""))
    return "\n".join(lines)


_ATOM_ARGS = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*\(([^)]*)\)")


def _about_pair(rule: str) -> str:
    """"yes" if the rule BODY relates the two series of the pair, else "no".

    A body is about the pair when one atom mentions both X and Y (isPositiveCorrelation(X,Y),
    tailDependent(X,Y), leadsPositively(X,Y,k)), or when some atoms are about X and others
    about Y. highVol(X) -> trendDown(X) is about X alone; tailDependent(X,X) pairs a series
    with itself and is not about the pair either. Applied identically to templates and LLM
    rules; it is a reported property, not a filter.
    """
    body = str(rule).split(":-", 1)[-1]
    vars_per_atom = []
    for args in _ATOM_ARGS.findall(body):
        toks = {t.strip().strip('"{}') for t in args.split(",")}
        vars_per_atom.append(toks & {"X", "Y"})
    joint = any(v == {"X", "Y"} for v in vars_per_atom)
    seen = set().union(*vars_per_atom) if vars_per_atom else set()
    return "yes" if joint or seen == {"X", "Y"} else "no"


def _llm_verdict_and_ci(row: dict, facts_cache: dict) -> "tuple[str, str]":
    """Verdict and 95% CI for one LLM rule, by EXACTLY the template path: the same
    bootstrap (scoring.bootstrap_lift, same seed and resamples) on the same test slice,
    and the same _verdict() thresholds."""
    from tslearning.config import CFG
    from tslearning.llm import propose as prop
    from tslearning.rules import scoring
    from tslearning.rules.generate import load_pair_facts

    a, pair, src = row["_a"], row["Pair"], row.get("_src", "py")
    if (pair, src) not in facts_cache:
        facts = load_pair_facts(pair, src, verbose=False)
        facts_cache[(pair, src)] = (facts, scoring.split_facts(facts)["test"])
    facts, test = facts_cache[(pair, src)]
    dg = a.get("diagnostics") or {}
    payload = {"rules": [{"name": str(a.get("template", "L")).removeprefix("LLM_"),
                          "head": re.sub(r"[{}]", "", a.get("head") or ""),
                          "body": re.sub(r"[{}]", "", a.get("body") or ""),
                          "constants": dg.get("declared_constants") or a.get("constants")}]}
    rules, _bad = prop.to_rules(payload, pair, facts.x, facts.y,
                                set(CFG["predicates"]["kind"]))
    ci = scoring.bootstrap_lift(rules[0], test) if rules else {}
    s = (a.get("scores") or {}).get("test") or {}
    vrow = {"accepted": True, "test_baseline": s.get("baseline"),
            "test_condition_days": s.get("condition_days"),
            "test_eligible": s.get("eligible", True), "test_reason": s.get("reason"),
            "test_lift": s.get("lift"),
            "ci_spans_zero": ci.get("spans_zero"), "ci_degenerate": ci.get("degenerate")}
    vrow = {k: (float("nan") if v is None else v) for k, v in vrow.items()}
    lo, hi = ci.get("ci_lo"), ci.get("ci_hi")
    ci_str = "—" if lo is None else f"[{_cell(lo, 3)}, {_cell(hi, 3)}]"
    return _verdict(vrow), ci_str


def comparison_table(rules_csv=None, results_dir=None, pairs=None,
                     llm_run: "str | None" = None) -> pd.DataFrame:
    """Template-fitter rules and LLM rules side by side, sorted by lift.

    RESTRICTED TO PAIRS BOTH PROPOSERS ATTEMPTED. Comparing the template fitter's rules
    across three pairs against an LLM run on one is not a comparison — the pair mix would
    drive the aggregate. Only pairs with at least one scored LLM rule are included, so the
    two columns face the same data.
    """
    res = Path(results_dir or PATHS["results"])
    rules_csv = Path(rules_csv or res / "rules.csv")
    rows = llm_rows(res, pairs, llm_run=llm_run)
    contested = {r["Pair"] for r in rows}
    if not contested:
        return pd.DataFrame()

    status_rows, omitted = [], {}          # omitted: {template: n unscored rows not shown}
    if rules_csv.exists():
        df = pd.read_csv(rules_csv)
        df = df[df.pair.isin(contested)]
        scored = df.accepted & df["test_lift"].notna() & \
            ~df.apply(_conclusion_absent, axis=1)
        for _, r in df[scored].iterrows():
            rows.append({"Proposer": "template", "Pair": r.pair,
                         "Rule": template_label(r.template),
                         "Constants": _constants(r),
                         "Train": r.get("train_confidence"),
                         "Test": r.get("test_confidence"), "Base": r.get("test_baseline"),
                         "Lift": r.get("test_lift"), "n": r.get("test_condition_days"),
                         "input": "facts", "run": "rules.csv",
                         "_verdict": _verdict(r),
                         "_ci": (f"[{_cell(r.get('ci_ci_lo'), 3)}, "
                                 f"{_cell(r.get('ci_ci_hi'), 3)}]"
                                 if pd.notna(r.get("ci_ci_lo")) else "—"),
                         "_body": r.get("rule", "")})
        # Templates with NO scored row on the contested pairs: one representative row
        # each, marked with its status; any further unscored rows are only counted.
        done = set(df[scored].template)
        for t, g in df[~scored].groupby("template", sort=True):
            if t in done:
                omitted[t] = len(g)
                continue
            r = g.iloc[0]
            status_rows.append({
                "Proposer": "template", "Pair": pair_label(r.pair),
                "Rule": template_label(r.template),
                "Constants": _constants(r) if r.accepted else "—",
                "Train": _cell(r.get("train_confidence")) if r.accepted else "—",
                "Test": "—", "Base": "—",
                "Lift": "ineligible" if r.accepted else "not fitted",
                "CI": "—", "n": "—",
                "Verdict": "ineligible" if r.accepted else "not fitted",
                "About pair": _about_pair(r.get("rule", "")) if r.accepted else "—"})
            omitted[t] = len(g) - 1

    # LLM rows: CI and Verdict computed exactly as for the templates
    facts_cache = {}
    for r in rows:
        if "_a" in r:
            r["_verdict"], r["_ci"] = _llm_verdict_and_ci(r, facts_cache)
    out = pd.DataFrame(rows).sort_values("Lift", ascending=False)
    table = [{
        "Proposer": r["Proposer"], "Pair": pair_label(r["Pair"]), "Rule": r["Rule"],
        "Constants": r["Constants"],
        "Train": _cell(r.get("Train")),
        "Test": _cell(r["Test"]), "Base": _cell(r["Base"]),
        "Lift": _cell(r["Lift"], 3, sign=True), "CI": r.get("_ci", "—"),
        "n": "—" if pd.isna(r["n"]) else str(int(r["n"])),
        "Verdict": _short_verdict(r.get("_verdict", "")),
        "About pair": _about_pair(r.get("_body", "")),
    } for r in out.to_dict("records")]
    # per LLM rule: what the boxed listing (llm_box.py) prints next to it
    info = {(r["Proposer"], r["Rule"]): {"verdict": _short_verdict(r.get("_verdict", "")),
                                         "lift": r.get("Lift"),
                                         "about": _about_pair(r.get("_body", ""))}
            for r in rows if "_a" in r}
    info.update({(r["Proposer"], r["Rule"]): {"verdict": LLM_UNSCORED_LABEL, "lift": None,
                                              "about": _about_pair(r.get("_body", ""))}
                 for r in getattr(llm_rows, "_unscored", [])})
    comparison_table.llm_info = info
    # rules that could not be scored go LAST, marked, with no numbers
    table += status_rows
    for r in getattr(llm_rows, "_unscored", []):
        if r["Pair"] not in contested:
            continue
        table.append({
            "Proposer": r["Proposer"], "Pair": pair_label(r["Pair"]), "Rule": r["Rule"],
            "Constants": r["Constants"], "Train": _cell(r.get("Train")),
            "Test": "—", "Base": "—",
            "Lift": LLM_UNSCORED_LABEL, "CI": "—",
            "n": "—", "Verdict": LLM_UNSCORED_LABEL,
            "About pair": _about_pair(r.get("_body", ""))})
    comparison_table.omitted = omitted
    return pd.DataFrame(table)


def _short_verdict(v: str) -> str:
    """'ineligible (n = 3 < 5 …)' -> 'ineligible'; the reasons are in the full tables."""
    return re.sub(r"\s*\(.*\)$", "", str(v)).replace("not fitted:", "not fitted").split(":")[0] \
        if v else "—"


def comparison_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-proposer spread — the number the comparison actually turns on."""
    if df.empty:
        return df
    # only scored rules have a lift; unscorable LLM rules are listed in the table but
    # can't enter a spread statistic
    df = df[~df["Lift"].isin(["ineligible", "not fitted"])]
    d = df.assign(_l=df["Lift"].astype(float))
    g = d.groupby("Proposer")["_l"]
    return pd.DataFrame({
        "Proposer": g.size().index,
        "rules": g.size().values,
        "mean lift": [f"{v:+.3f}" for v in g.mean().values],
        "positive": [f"{int((d[d.Proposer == p]._l > 0).sum())}/{int(n)}"
                     for p, n in zip(g.size().index, g.size().values)],
        "best": [f"{v:+.3f}" for v in g.max().values],
        "worst": [f"{v:+.3f}" for v in g.min().values],
    })


CAPTION_COMPARISON = (
    "Rule proposers compared on identical terms: constants frozen on <TRAIN>, scored "
    "on <TEST> by the same code. The comparison is not about the best rule but about "
    "the \\emph{spread} --- a proposer whose median rule has negative lift has produced "
    "candidates, not knowledge. \\emph{CI} and \\emph{Verdict} are computed for every "
    "rule by the same bootstrap and thresholds as in Figure~\\ref{fig:rule-results}; "
    "\\emph{About pair} states whether the rule's body relates the two series (an atom "
    "on both, or atoms on each), as opposed to one series alone."
)

UNSCORED_NOTE = (
    " Listed last: \\emph{not fitted} (no constants on <TRAIN>), \\emph{ineligible} "
    "(condition or conclusion never holds on <TEST>)."
)


def _llm_note() -> str:
    """Caption sentence on how the LLM rows were obtained (pooling, selection)."""
    c = getattr(llm_rows, "_counts", None)
    if not c or not c.get("runs"):
        return ""
    sel = (f" LLM: per model, the top {c['top_k']} by \\emph{{train}} lift"
           if c["pooled"] else f" {c['selected']} LLM rules")
    return (f"{sel} of {c['unique']} distinct rules ({c['proposed']} proposals, "
            f"{c['runs']} run{'s' if c['runs'] != 1 else ''}).")


def _omitted_note(n: int) -> str:
    return (f" {n} unscored template row{'s' if n != 1 else ''} omitted."
            if n else "")

CAPTION_LIBRARY = (
    "The rule library. Each template encodes a named phenomenon from the finance "
    "literature; constants are fitted per pair (Figure~\\ref{fig:rule-results}) by the "
    "1-SE rule on the training period only."
)
CAPTION_RESULTS = (
    "Rule results, constants frozen on <TRAIN> and scored on <TEST>. "
    "\\emph{Base} is the unconditional rate of the conclusion predicate and "
    "\\emph{Lift} = Test $-$ Base; a high baseline makes a high confidence vacuous. "
    "\\emph{CI} is the 95\\% bootstrap confidence interval on lift. "
    "\\emph{n} is the number of condition days; below "
    f"{MIN_CONDITION_DAYS} the rule is not scored (blank, not zero). "
    "$|\\Delta|$ is the disagreement between the Python scorer and the DatalogMTL "
    "reasoner re-evaluating the same frozen rule. Rules that could not be fitted are "
    "listed with the reason: these are results, not omissions."
)


def _delta_mode(df: pd.DataFrame) -> "str | None":
    """How the engine check appears in a pub table.

      "zero"   every engine-verified rule has |Δ| = 0.000 -> column dropped, one caption
               sentence instead (a column of zeros is space, not information)
      "shown"  some rule disagrees -> the column stays, so a mismatch can never be hidden
      None     no engine numbers at all -> neither column nor sentence
    """
    if "Δ" not in df.columns:
        return None
    vals = df["Δ"].astype(str)
    checked = vals[vals != "—"]
    if checked.empty:
        return None
    return "zero" if (checked == "0.000").all() else "shown"


def _caption_results_pub(tag: "str | None", not_fitted: "list[str]",
                         delta: "str | None" = "shown") -> str:
    """Caption for the pub table — mentions label source and lists omitted templates."""
    source = "classifier-derived" if tag == "classifier" else "exact"
    nf = ""
    if not_fitted:
        nf = (f" Templates not shown (could not be fitted on any pair): "
              f"{', '.join(not_fitted)}.")
    engine = {
        "zero": "; the reasoner reproduces every confidence ($|\\Delta| = 0$).",
        "shown": "; $|\\Delta|$: Python vs reasoner.",
        None: "",
    }[delta]
    return (
        f"Rule results ({source} labels), fitted on <TRAIN>, scored on <TEST>. "
        f"\\emph{{Lift}} = Test $-$ Base (unconditional rate); \\emph{{n}}: condition "
        f"days"
        f"{engine or '.'}{nf}"
    )


def run(out_dir=None, pairs=None, templates=None, fmt: str = "both",
        period: str = "test", scored_only: bool = False, tag: "str | None" = None,
        llm_run: "str | None" = None, in_dir=None, verbose: bool = True) -> dict:
    """Build both tables and write them next to the other result files.

    `tag` selects a labelled set of inputs and suffixes the outputs, so the tables from two
    label sources can sit side by side without overwriting each other:

        tag=None          reads rules.csv / verification.csv, writes rule_results.*
        tag="classifier"  reads rules_classifier.csv / verification_classifier.csv,
                          writes rule_results_classifier.*
    """
    # in_dir = where the inputs are read (rules*, verification*, llm_*.json),
    # out_dir = where the tables go. Both default to results/; they used to be one folder.
    in_dir = Path(in_dir or PATHS["results"])
    out_dir = Path(out_dir or PATHS["results"])
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = f"_{tag}" if tag else ""
    rules_csv = in_dir / f"rules{sfx}.csv"
    verification_csv = in_dir / f"verification{sfx}.csv"
    if not rules_csv.exists():
        raise FileNotFoundError(
            f"{rules_csv} not found — run generate_rules.py --score"
            + (f" --source {tag} --out {rules_csv}" if tag else "") + " first")

    t1 = library_table(templates)
    t1_tex = library_table(templates, latex_cites=True)  # \cite{} for LaTeX
    t2 = results_table(pairs=pairs, templates=templates, period=period,
                       scored_only=scored_only, rules_csv=rules_csv,
                       verification_csv=verification_csv)
    if verbose and len(t2):
        n_scored = int((t2["Test"] != "—").sum())
        n_rej = len(t2) - n_scored
        print(f"\n{len(t2)} rows: {n_scored} scored, {n_rej} not fitted or ineligible."
              + ("" if scored_only else
                 "\nRejections are kept deliberately — a template that cannot fire on a "
                 "pair is a result,\nnot an omission. Use --scored-only for just the "
                 "numbers, or -p to restrict the pairs."))

    # The LLM comparison is anchored to the exact-labeller rules; only build it untagged.
    t3 = comparison_table(rules_csv=rules_csv, results_dir=in_dir, pairs=pairs,
                          llm_run=llm_run) if not tag else pd.DataFrame()
    # (name, df_csv_md, df_tex, caption, label, rotate, fontsize)
    # df_tex may differ from df_csv_md (e.g. library uses \cite{} in LaTeX)
    tables = [
        ("rule_library", t1, t1_tex, CAPTION_LIBRARY, "fig:rule-library",
         True, ""),
        (f"rule_results{sfx}", t2, t2, CAPTION_RESULTS, f"fig:rule-results{sfx}",
         True, ""),
    ]
    if not t3.empty:
        n_omit = sum(getattr(comparison_table, "omitted", {}).values())
        tables.append(("rule_comparison", t3, t3,
                        CAPTION_COMPARISON + _llm_note() + UNSCORED_NOTE
                        + _omitted_note(n_omit),
                        "fig:rule-comparison", False, ""))

    written = []
    for name, df, df_l, cap, lab, rot, fsize in tables:
        df.to_csv(out_dir / f"{name}.csv", index=False)
        written.append(out_dir / f"{name}.csv")
        if fmt in ("md", "both"):
            (out_dir / f"{name}.md").write_text(to_markdown(df) + "\n")
            written.append(out_dir / f"{name}.md")
        if fmt in ("tex", "both"):
            (out_dir / f"{name}.tex").write_text(
                to_latex(df_l, cap, lab, rotate=rot, fontsize=fsize) + "\n")
            written.append(out_dir / f"{name}.tex")
        if verbose:
            print(f"\n### {name}\n")
            print(to_markdown(df))

    if not t3.empty:
        summary = comparison_summary(t3)
        summary.to_csv(out_dir / "rule_comparison_summary.csv", index=False)
        written.append(out_dir / "rule_comparison_summary.csv")

        listing = llm_rule_listing(results_dir=in_dir, pairs=pairs, llm_run=llm_run)
        # name the listing after the run(s) it lists: llm_rules_20260924_110120.txt
        run_re = re.compile(r"_(\d{8}_\d{6}|run_\d+)$")
        run_ids = sorted({m.group(1) for f in _pick_llm_files(in_dir, llm_run)
                          if (m := run_re.search(f.stem))})
        listing_path = out_dir / (f"llm_rules_{'_'.join(run_ids)}.txt" if run_ids
                                  else "llm_rules.txt")
        listing_path.write_text(listing + "\n")
        written.append(listing_path)

        # the same rules as framed LaTeX boxes (full + _pub), named after the runs so a
        # new set of runs never overwrites them. Never allowed to stop the other tables.
        try:
            from tslearning.rules import llm_box
            written += llm_box.write(out_dir, run_ids, llm_rows._bodies,
                                     getattr(comparison_table, "llm_info", {}),
                                     getattr(llm_rows, "_per_model", {}),
                                     _llm_pool_settings()["top_k"],
                                     _llm_pool_settings()["box_top_k"])
        except Exception as e:                                  # pragma: no cover
            print(f"WARNING: LLM rule box not written ({type(e).__name__}: {e})")

        if verbose:
            print("\n### proposer spread\n")
            print(to_markdown(summary))
            print("\nRead the SPREAD, not the winner: a proposer whose best rule is "
                  "excellent and whose\nmedian rule has negative lift produced candidates "
                  "that something else had to sort.")
            print(f"\n### LLM rule listing -> {listing_path}\n")
            print(listing)

    # ── Publication tables: chosen templates only ─────────────────────
    from tslearning.config import CFG
    chosen = CFG.get("paper", {}).get("chosen_templates")
    if chosen:
        # Map short labels (T1b, T2, …) back to the full template names
        short_to_full = {}
        for c in chosen:
            short_to_full[c.split("_")[0]] = c

        def _write_pub(name, df, caption, label, rotate=False):
            df.to_csv(out_dir / f"{name}.csv", index=False)
            written.append(out_dir / f"{name}.csv")
            if fmt in ("md", "both"):
                (out_dir / f"{name}.md").write_text(to_markdown(df) + "\n")
                written.append(out_dir / f"{name}.md")
            if fmt in ("tex", "both"):
                (out_dir / f"{name}.tex").write_text(
                    to_latex(df, caption, label, rotate=rotate) + "\n")
                written.append(out_dir / f"{name}.tex")
            if verbose:
                print(f"\n### {name} ({len(df)} rows, chosen templates only)\n")
                print(to_markdown(df))

        # ── rule_library_pub ──
        lib_pub = t1[t1["#"].isin(short_to_full.keys())].reset_index(drop=True)
        lib_pub_tex = t1_tex[t1_tex["#"].isin(short_to_full.keys())].reset_index(drop=True)
        if not lib_pub.empty:
            cap_lib = (
                "The rule library (selected templates). Each template encodes a named "
                "phenomenon from the finance literature; constants are fitted per pair "
                "(Figure~\\ref{fig:rule-results" + sfx + "-pub}) by the 1-SE rule on the "
                "training period only."
            )
            _write_pub("rule_library_pub", lib_pub, cap_lib, "fig:rule-library-pub",
                        rotate=True)
            # also write the LaTeX version with \cite{}
            if fmt in ("tex", "both"):
                (out_dir / "rule_library_pub.tex").write_text(
                    to_latex(lib_pub_tex, cap_lib, "fig:rule-library-pub",
                             rotate=True) + "\n")

        # ── rule_library_box(_pub): the library as a framed box, not a rotated table ──
        if fmt in ("tex", "both"):
            box_note = (" Auxiliary rules, listed first, name conjunctions under temporal "
                        "operators.")
            (out_dir / "rule_library_box.tex").write_text(
                library_box(None, CAPTION_LIBRARY + box_note, "fig:rule-library") + "\n")
            written.append(out_dir / "rule_library_box.tex")
            chosen_full = list(short_to_full.values())
            if chosen_full:
                cap_box = (
                    "Rule library (selected templates), body $\\to$ head; constants fitted "
                    "per pair on the training period (1-SE rule).")
                (out_dir / "rule_library_box_pub.tex").write_text(library_box(
                    chosen_full, cap_box + box_note, "fig:rule-library-pub") + "\n")
                written.append(out_dir / "rule_library_box_pub.tex")

        # ── rule_results_pub ──
        if not t2.empty:
            chosen_rows = t2[t2["Rule"].isin(short_to_full.keys())].copy()
            fitted = chosen_rows[chosen_rows["Test"] != "—"]
            # For templates with NO fitted rows, keep one representative row
            fitted_shorts = set(fitted["Rule"].unique())
            representatives = []
            for s in short_to_full:
                if s not in fitted_shorts:
                    unfitted = chosen_rows[chosen_rows["Rule"] == s]
                    if not unfitted.empty:
                        representatives.append(unfitted.iloc[[0]])
            pub = pd.concat([fitted] + representatives, ignore_index=True) \
                  if representatives else fitted.reset_index(drop=True)
            # Shorten verbose verdicts in pub tables
            if "Verdict" in pub.columns:
                pub["Verdict"] = pub["Verdict"].str.replace(
                    r"^not fitted:.*", "not fitted", regex=True)
                pub["Verdict"] = pub["Verdict"].str.replace(
                    r"^ineligible \(.*\)", "ineligible", regex=True)
            # Caption notes which extra unfitted rows were dropped (not all pairs shown)
            n_dropped = len(chosen_rows) - len(pub)
            drop_note = (f" {n_dropped} unfitted rows omitted."
                         if n_dropped > 0 else "")
            # decided on ALL scored rules (t2), so the caption's "every scored rule" holds
            delta = _delta_mode(t2)
            if delta == "zero" and "Δ" in pub.columns:
                pub = pub.drop(columns=["Δ"])
            cap_pub = _caption_results_pub(tag, [], delta) + drop_note
            _write_pub(f"rule_results{sfx}_pub", pub, cap_pub,
                        f"fig:rule-results{sfx}-pub")

        # ── rule_comparison_pub ──
        if not t3.empty:
            # Keep all LLM rows + only chosen-template rows from the fitter.
            # Template rows use full labels ("T2 contagion"), so match on prefix.
            is_llm = t3["Proposer"].str.startswith("LLM")
            is_chosen = t3["Rule"].str.split(" ").str[0].isin(short_to_full.keys())
            comp_pub = t3[is_llm | is_chosen].drop(columns=["Train"], errors="ignore") \
                .reset_index(drop=True)
            if not comp_pub.empty:
                omit = getattr(comparison_table, "omitted", {})
                n_omit = sum(v for t, v in omit.items()
                             if t.split("_")[0] in short_to_full)
                cap_comp = (
                    "Template and LLM rules, fitted on <TRAIN>, scored on <TEST> by the "
                    "same code."
                    + _llm_note() + UNSCORED_NOTE + _omitted_note(n_omit)
                )
                _write_pub("rule_comparison_pub", comp_pub, cap_comp,
                            "fig:rule-comparison-pub")

    return {"library": t1, "results": t2, "comparison": t3, "written": written}
