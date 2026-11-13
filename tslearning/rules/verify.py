"""
R1.3 — verification. Re-evaluate a frozen rule with the reasoner on held-out data.

This is the step that literally closes the neurosymbolic loop: the mined rule is handed
back to the same DatalogMTL engine that defined the vocabulary, run over a period it was
not fitted on, and its confidence is measured from the engine's own derivations.

Two engines, deliberately interchangeable:

    engine="python"    scoring.py over the interval facts we already have. No engine
                       needed. Fast, and the number to compare against.
    engine="vadalog"   compose preLabeling.vada + the frozen rule, run it, read the
                       derived head, compute confidence from the engine's own output.

Agreement between the two is itself a result: it says the Python labeler faithfully
reproduces the declarative semantics. Disagreement localises a bug in one of them.

────────────────────────────────────────────────────────────────────────────────
WHY THE EDB IS A CONTINUOUS TIMELINE, NOT DISJOINT BLOCKS
────────────────────────────────────────────────────────────────────────────────

Training batches use a disjoint-block layout so the reasoner CANNOT join across window
pairs — each pair must be labelled independently. Verification needs the opposite: the
mined rule's temporal operators (<+>[0,w], [+][0,d]) reason ACROSS windows on the real
timeline, so the EDB is one continuous stretch of daily flow facts per series, at real day
indices. `/\\50` then slides over it, exactly as the real-pair path intends, and derived
predicates come out as real-timeline intervals — comparable to the Python facts day for day.

Consequence: series identifiers are plain ("sp500", "gold"), not "sp500_real_123",
because each series is one continuous flow relation rather than one relation per window.

────────────────────────────────────────────────────────────────────────────────
WHERE THE GENERATED PROGRAM LIVES
────────────────────────────────────────────────────────────────────────────────

The composed program is written to data/processed/verification/<tag>.vada and executed
FROM THERE, so the file you inspect is the file that ran. That requires the engine's
repository to be the repo root (`reasoner_repository: "."` in configs/config.yaml, and the
engine configured to match), because the program sits outside tslearning/symbolic/.
Program names are then repository-relative: "tslearning/symbolic/preLabeling.vada",
"data/processed/verification/T1_safeHaven_sp500_gold_test.vada".
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from tslearning.config import CFG, PATHS

def _ruleset_default() -> Path:
    """preLabeling.vada for the current run (generated if the run's window or
    cut-offs differ from the base program; see reasoner_client.ruleset_path)."""
    from tslearning.labeling.reasoner_client import ruleset_path
    return ruleset_path()
ANNOTATION_ANCHOR = "%% ======== ANNOTATIONS ======== %%"


# ══════════════════════════════════════════════════════════════════════════════
#  Program composition
# ══════════════════════════════════════════════════════════════════════════════

def head_signature(rule) -> "tuple[str, int]":
    """('safeHavenEpisode', 2) from the rule's head template."""
    m = re.match(r"\s*(\w+)\s*\((.*)\)\s*$", rule.head)
    if not m:
        raise ValueError(f"cannot parse head: {rule.head!r}")
    return m.group(1), len([a for a in m.group(2).split(",") if a.strip()])


# The CSV options string preLabeling.vada uses on every @bind. Kept verbatim, including
# the literal backslash-n of recordSeparator.
CSV_OPTS = ("csv useHeaders=true, delimiter=',',nullString='NULL', "
            "quoteMode='MINIMAL', recordSeparator='\\n'")


def output_block(name: str, arity: int) -> str:
    """The @output/@bind/@mapping/@temporalMappings block for a derived predicate,
    in the same shape preLabeling.vada uses for its own outputs."""
    lines = ['@output("%s").' % name,
             '@bind("%s","%s","${R}","%s_${I}.csv").' % (name, CSV_OPTS, name)]
    for i, t in enumerate(["X", "Y", "Z"][:arity]):
        lines.append('@mapping("%s",%d,"%s","string").' % (name, i, t))
    lines.append('@mapping("%s",%d,"Start","int").' % (name, arity))
    lines.append('@mapping("%s",%d,"End","int").' % (name, arity + 1))
    lines.append('@temporalMappings("%s",%d,%d,-1,-1,"[_,_]").'
                 % (name, arity, arity + 1))
    return "\n".join(lines)


def predicate_arities(text: str) -> dict:
    """{predicate: arity} read from the program's own @mapping declarations.

    Arity = the mapped columns that are NOT the temporal Start/End pair. Several
    predicates that read as unary carry a value column and are really binary:

        highVol(X,Pct)   lowVol(X,Pct)
        trendUp(X,Sum)   trendDown(X,Sum)   deadZone(X,Sum)

    Deriving this from the .vada rather than hardcoding it means the executable rule
    follows the program if its signatures ever change.
    """
    temporal = set()
    for m in re.finditer(r'^\s*@temporalMappings\("(\w+)",(\d+),(\d+)', text, re.M):
        temporal.add((m.group(1), int(m.group(2))))
        temporal.add((m.group(1), int(m.group(3))))

    cols = {}
    for m in re.finditer(r'^\s*@mapping\("(\w+)",(\d+),', text, re.M):
        pred, idx = m.group(1), int(m.group(2))
        if (pred, idx) in temporal:
            continue
        cols.setdefault(pred, set()).add(idx)
    return {p: len(ix) for p, ix in cols.items()}


def to_vadalog(body: str, arities: dict) -> str:
    """Pad each atom in a rule body to its declared arity with anonymous variables.

        highVol(X), trendDown(X)   ->   highVol(X,_), trendDown(X,_)

    Atoms already at (or above) their declared arity are left alone, and predicates the
    program doesn't declare are left alone — this only fixes the known-arity mismatch, it
    does not invent structure.
    """
    def pad(m):
        pred, args = m.group(1), m.group(2).strip()
        want = arities.get(pred)
        if want is None:
            return m.group(0)
        have = [a for a in args.split(",") if a.strip()]
        if len(have) >= want:
            return m.group(0)
        return f"{pred}({','.join(have + ['_'] * (want - len(have)))})"

    return re.sub(r"(\w+)\s*\(([^()]*)\)", pad, body)


def input_block(name: str, arity: int, filename: str, lag: bool = False) -> str:
    """An @input/@bind/@mapping block reading interval facts from a CSV.

    Columns written by `write_fact_csvs`:
        unary  X,Start,End              -> arity 1
        pair   X,Y,Start,End            -> arity 2
        lag    X,Y,Lag,Start,End        -> arity 3 (Lag is an int term)
    Same shape as afterPreLabeling.vadanb, which is the precedent for feeding derived
    predicates back in as an EDB.
    """
    lines = ['@input("%s").' % name,
             '@bind("%s","%s","${R}","%s").' % (name, CSV_OPTS, filename)]
    terms = ["X", "Y"][:min(arity, 2)]
    for i, t in enumerate(terms):
        lines.append('@mapping("%s",%d,"%s","string").' % (name, i, t))
    if lag:
        lines.append('@mapping("%s",2,"Lag","int").' % name)
    lines.append('@mapping("%s",%d,"Start","int").' % (name, arity))
    lines.append('@mapping("%s",%d,"End","int").' % (name, arity + 1))
    lines.append('@temporalMappings("%s",%d,%d,-1,-1,"[_,_)").'
                 % (name, arity, arity + 1))
    return "\n".join(lines)


def rule_predicates(rule) -> list:
    """[(predicate, arity, is_lag)] for every atom the rule's body references."""
    parts = rule.parts or {}
    specs = list(parts.get("condition", [])) + list(parts.get("conclusion_multi", []))
    for key in ("conclusion", "pred", "from", "to"):
        if parts.get(key):
            specs.append(parts[key])

    out, seen = [], set()
    for pred, role in specs:
        if pred in seen:
            continue
        seen.add(pred)
        is_lag = "lag" in CFG["predicates"]["kind"].get(pred, "")
        # 'xc'/'yc' = market(X, C): a binary static fact (series, economy)
        arity = 3 if is_lag else (2 if role in ("xy", "xc", "yc", "xx", "yy") else 1)
        out.append((pred, arity, is_lag))
    return out


def write_fact_csvs(rule, facts, out_dir, period: str = "test",
                    verbose: bool = True) -> dict:
    """Write the rule's referenced predicates as interval-fact CSVs for the period.

    This is the EDB for `mode="facts"`: the predicates are NOT recomputed by the engine,
    they are handed to it as facts on the real timeline — which is what makes the mined
    rule's temporal operators meaningful and sidesteps preLabeling's per-series max()
    aggregates entirely.
    """
    from tslearning.rules import scoring

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    periods = scoring.split_facts(facts)
    scoped = periods.get(period, facts)

    written = {}
    for pred, arity, is_lag in rule_predicates(rule):
        rows = []
        for key, s, e in scoped.intervals.get(pred, []):
            k = key if isinstance(key, tuple) else (key,)
            if is_lag:
                rows.append({"X": k[0], "Y": k[1], "Lag": k[2], "Start": s, "End": e})
            elif arity == 2:
                rows.append({"X": k[0], "Y": k[1] if len(k) > 1 else k[0],
                             "Start": s, "End": e})
            else:
                rows.append({"X": k[0], "Start": s, "End": e})
        cols = (["X", "Y", "Lag", "Start", "End"] if is_lag else
                ["X", "Y", "Start", "End"] if arity == 2 else ["X", "Start", "End"])
        fname = f"edb_{pred}.csv"
        pd.DataFrame(rows, columns=cols).to_csv(out_dir / fname, index=False)
        written[pred] = {"file": fname, "arity": arity, "lag": is_lag, "rows": len(rows)}
        if verbose:
            print(f"    EDB {pred:24s} {len(rows):4d} intervals -> {fname}")
    return written


def render_facts_program(rule, facts_spec: dict, ground: bool = True) -> str:
    """A self-contained program: @input the predicate facts + the mined rule + @output.

    No preLabeling rules, so nothing is recomputed and the per-series max() aggregates
    that make preLabeling window-layout-dependent never come into play. The rule is used
    exactly as written — arities here match the CSVs, so no padding is needed.
    """
    name, arity = head_signature(rule)
    head, body = rule.render_parts(ground=ground, vadalog=True)

    blocks = ["%% ======== EDB: derived predicate facts, real timeline ======== %%", ""]
    for pred, meta in facts_spec.items():
        blocks.append(input_block(pred, meta["arity"], meta["file"], meta["lag"]))
        blocks.append("")

    blocks += ["%% ======== MINED RULE UNDER VERIFICATION ======== %%",
               f"% template : {rule.template}",
               f"% pair     : {rule.pair}",
               f"% constants: {rule.constants}",
               f"% fitted on: {CFG['split']['train'][0]}..{CFG['split']['train'][1]}",
               *[a.rstrip(".") + "." for a in rule.render_aux(ground=ground)],
               f"{head} :- {body}.",
               "",
               output_block(name, arity),
               "",
               "%% ======== ANNOTATIONS ======== %%",
               "",
               '@timeGranularity("no_date").',
               '@temporalType("int").',
               "",
               '@library("math:","math").',
               ""]
    return "\n".join(blocks)


def output_predicates(text: str) -> set:
    """Names declared with @output (excluding anything also declared @input)."""
    outs = set(re.findall(r'^\s*@output\("(\w+)"\)', text, re.M))
    ins = set(re.findall(r'^\s*@input\("(\w+)"\)', text, re.M))
    return outs - ins


def strip_outputs(text: str, keep=()) -> str:
    """Remove the @output-connected annotations (@output/@bind/@mapping/
    @temporalMappings) for every derived predicate except those in `keep`.

    The @input bindings are untouched — the engine still needs to read flow/tau/qLow.
    Only the OUTPUT side is pruned, so a verification run derives everything internally
    but writes just the predicates being verified instead of 25 CSVs.
    """
    drop = output_predicates(text) - set(keep)
    if not drop:
        return text
    pattern = re.compile(
        r'^\s*@(?:output|bind|mapping|temporalMappings)\(\s*"(' + "|".join(map(re.escape, drop)) + r')"',
        re.M)
    kept = [ln for ln in text.splitlines() if not pattern.match(ln)]
    return "\n".join(kept)


def render_program(rule, ruleset: "Path | None" = None, ground: bool = True,
                   keep_condition_outputs: bool = False) -> str:
    """preLabeling.vada, stripped of its own outputs, + the frozen rule + its @output.

    `ground=True` verifies the specific claim about these two series; ground=False leaves
    the variables free, so the rule fires for any pair present in the EDB.

    `keep_condition_outputs=False` (the default) emits ONLY the rule's head. The body
    atoms are still derived internally — the engine needs them to fire the rule — they are
    simply not written out, because the confidence denominator doesn't need them: the
    condition is exact market state, already computed in Python. Set
    True to have the engine emit them too, e.g. to cross-check the two derivations.
    """
    ruleset = Path(ruleset or _ruleset_default())
    base = ruleset.read_text()
    name, arity = head_signature(rule)

    keep = []
    if keep_condition_outputs:
        keep = [p for p, _role in rule.parts.get("condition", [])]
        concl = rule.parts.get("conclusion") or rule.parts.get("pred") or rule.parts.get("from")
        if concl:
            keep.append(concl[0])
        if rule.parts.get("to"):
            keep.append(rule.parts["to"][0])

    # strip_outputs leaves the kept predicates' ORIGINAL blocks in place — don't
    # regenerate them, their mappings carry extra columns (highVol has a Pct term) that a
    # generic block would get wrong.
    body_text = strip_outputs(base, keep=keep)

    # The executable rule. A template can supply vada_head/vada_body explicitly; otherwise
    # the body's atoms are padded to the arities the program declares. Only the BODY is
    # padded — the head is a predicate we are defining, and its arity must stay in step
    # with output_block below.
    rule_head, rule_body = rule.render_parts(ground=ground, vadalog=True)
    if rule.vada_body is None:
        rule_body = to_vadalog(rule_body, predicate_arities(base))
    rule_line = f"{rule_head} :- {rule_body}."
    readable = rule.render(ground=ground)
    # helper rules: their bodies are padded like the rule's own
    aux_lines = []
    for a in rule.render_aux(ground=ground):
        a_head, a_body = a.rstrip(".").split(":-", 1)
        aux_lines.append(f"{a_head.strip()} :- "
                         f"{to_vadalog(a_body.strip(), predicate_arities(base))}.")

    lines = ["",
             "%% ======== MINED RULE UNDER VERIFICATION ======== %%",
             f"% template : {rule.template}",
             f"% pair     : {rule.pair}",
             f"% constants: {rule.constants}",
             f"% fitted on: {CFG['split']['train'][0]}..{CFG['split']['train'][1]}",
             f"% kept outputs: {', '.join(dict.fromkeys(keep)) or '(head only)'}"]
    if rule_line != readable:
        lines.append(f"% as reported : {readable}")
        lines.append("% executed below with atoms padded to the arities this program declares"
                     if rule.vada_body is None else
                     "% executed below using the template's explicit vada_body")
    lines += [*aux_lines, rule_line, "", output_block(name, arity), ""]
    text = "\n".join(lines)
    body = body_text
    if ANNOTATION_ANCHOR in body:
        return body.replace(ANNOTATION_ANCHOR, text + "\n" + ANNOTATION_ANCHOR, 1)
    return body + text


# ══════════════════════════════════════════════════════════════════════════════
#  EDB — a continuous timeline for the verification period
# ══════════════════════════════════════════════════════════════════════════════

def build_edb_csv(pair: str, out_path, period: str = "test", lo=None, hi=None,
                  verbose: bool = True) -> dict:
    """Write the flow facts for one pair over the verification period.

    One continuous relation per series at real day indices — see the module docstring.
    Returns {"path", "n_days", "series", "date_from", "date_to", "day_offset"}.
    """
    from tslearning.datasets.pairs import describe, load_series

    d = describe(pair)
    if lo is None or hi is None:
        lo, hi = CFG["split"][period]
    series = load_series([d["a"], d["b"]])

    idx = None
    for s in (series[d["a"]], series[d["b"]]):
        i = s.dropna().sort_index().loc[lo:hi].index
        idx = i if idx is None else idx.intersection(i)
    idx = idx.sort_values()
    if len(idx) == 0:
        raise ValueError(f"{pair}: no aligned observations in {lo}..{hi}")

    a = series[d["a"]].loc[idx].to_numpy(float)
    b = series[d["b"]].loc[idx].to_numpy(float)

    rows, obs = [], 0
    for name, arr in ((d["a_key"], a), (d["b_key"], b)):
        for t, v in enumerate(arr):
            rows.append({"window_name": name, "observation_id": obs, "value": float(v),
                         "start_time": t, "end_time": t + 1})
            obs += 1

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["window_name", "observation_id", "value",
                                "start_time", "end_time"]).to_csv(out_path, index=False)
    if verbose:
        print(f"  EDB: {len(idx)} days x 2 series -> {out_path.name} "
              f"({idx[0].date()}..{idx[-1].date()})")
    return {"path": out_path, "n_days": len(idx), "series": (d["a_key"], d["b_key"]),
            "date_from": str(idx[0].date()), "date_to": str(idx[-1].date()),
            "dates": {i: str(ts.date()) for i, ts in enumerate(idx)}}


# ══════════════════════════════════════════════════════════════════════════════
#  Verification
# ══════════════════════════════════════════════════════════════════════════════

def verify_python(rule, facts, period: str = "test", max_gap=float("inf")) -> dict:
    """Score the frozen rule on `period` using the interval facts we already have."""
    from tslearning.rules import scoring
    parts = scoring.split_facts(facts)
    sc = scoring.score_rule(rule, parts[period] if period in parts else facts, max_gap)
    return {"engine": "python", "period": period, **sc}


def _translate_days(ivs, edb_dates: dict, facts_dates: dict) -> list:
    """Re-index reasoner intervals from EDB day numbers onto the pair's full timeline.

    build_edb_csv numbers the verification period from 0, while the Python interval facts
    are numbered over the whole labelled history. Both are indexes into aligned trading
    days, so the shared calendar date is the join key.
    """
    if not edb_dates or not facts_dates:
        return list(ivs)
    date_to_full = {str(v): k for k, v in
                    ((d, pd.Timestamp(v).date()) for d, v in facts_dates.items())}
    out = []
    for s, e in ivs:
        ds, de = edb_dates.get(int(s)), edb_dates.get(int(e) - 1)
        fs, fe = date_to_full.get(str(ds)), date_to_full.get(str(de))
        if fs is None or fe is None:
            continue
        out.append((fs, fe + 1))
    return out


def verify_vadalog(rule, pair: str, period: str = "test", ground: bool = True,
                   result_dir=None, tag: "str | None" = None, run: bool = True,
                   facts=None, keep_body_outputs: bool = False, mode: str = "facts",
                   verbose: bool = True) -> dict:
    """Verify with the engine.

    mode="facts"    (default) EDB = the predicate INTERVAL facts for the period; the
        program is @input blocks + the mined rule. This is the afterPreLabeling.vadanb
        shape. Nothing is recomputed.
    mode="prelabel" EDB = raw daily flow; the program is preLabeling.vada + the rule, so
        the engine derives the predicates itself. Appealing, but preLabeling's
        `Max = max(C)` aggregates are per-SERIES, and in the training layout each series
        name IS one window — on a continuous timeline they collapse to one value for the
        whole series and highVol/correlation never fire per window. Kept for the
        disjoint-block case and for experimentation; not the default.
    """
    """Compose the program, (optionally) run it, and score from the engine's own output.

    `run=False` writes the program and EDB and stops — the dry-run path, so the whole
    thing can be built and inspected without an engine, then executed by hand.
    """
    from tslearning.labeling import reasoner_client
    from tslearning.rules.generate import _coalesce_presence
    from tslearning.rules.primitives import PairFacts, coalesce, days, overlap

    data_dir = PATHS["data_processed"]
    result_dir = Path(result_dir or (data_dir / "verification"))
    result_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or f"{rule.template}_{pair}_{period}"

    if facts is None:
        from tslearning.rules.generate import load_pair_facts
        facts = load_pair_facts(pair, verbose=False)

    prog_path = result_dir / f"{tag}.vada"
    if mode == "facts":
        facts_spec = write_fact_csvs(rule, facts, result_dir, period, verbose=verbose)
        program = render_facts_program(rule, facts_spec, ground=ground)
        edb = {"path": result_dir, "dates": None, "spec": facts_spec}
    else:
        program = render_program(rule, ground=ground,
                                 keep_condition_outputs=keep_body_outputs)
        edb = build_edb_csv(pair, result_dir / f"{tag}_edb.csv", period, verbose=verbose)
    prog_path.write_text(program)
    if verbose:
        print(f"  program: {prog_path.name} "
              f"({len(program.splitlines())} lines, mode={mode})")

    if not run:
        return {"engine": "vadalog", "period": period, "ran": False,
                "program": prog_path, "edb": edb["path"], "tag": tag,
                "confidence": None,
                "reason": "dry run — program and EDB written, not executed"}

    # Run the program from where it was written — no staging, no temp copy. This requires
    # the engine's repository to cover data/processed/verification/, i.e.
    # reasoner_repository = "." in the config. program_name() raises with instructions if
    # the folder is out of reach.
    try:
        ruleset = reasoner_client.program_name(prog_path)
    except ValueError as e:
        return {"engine": "vadalog", "period": period, "ran": False,
                "program": prog_path, "edb": edb["path"], "tag": tag,
                "confidence": None, "reason": str(e)}
    if verbose:
        print(f"  programName: {ruleset}")
    try:
        reasoner_client.evaluate(edb["path"], tag, result_dir,
                                 ruleset=ruleset, timeout=None)
        # (facts mode binds its inputs through ${R}; X is unused but harmless)
    except Exception as e:
        return {"engine": "vadalog", "period": period, "ran": False,
                "program": prog_path, "edb": edb["path"], "tag": tag,
                "confidence": None, "reason": f"reasoner error: {e}"}

    # The HEAD comes from the engine — that is what is being verified.
    head_name, _ = head_signature(rule)
    kinds = CFG["predicates"]["kind"]

    def _read(pred, kind):
        f = result_dir / f"{pred}_{tag}.csv"
        if not f.exists():
            return []
        return [(s, e) for _k, s, e in _coalesce_presence(pd.read_csv(f), kind)]

    head_ivs = _read(head_name, "pair_bin")

    # The CONDITION (denominator) comes from whichever source has it. By default the body
    # atoms are not emitted, so it is taken from the exact Python facts — legitimate,
    # since the condition is exact market state, not a model output.
    # `parts` names its atoms differently per kind — "condition" for conditional (T1-T4),
    # "pred" for persistence (T5), "from"/"to" for transition (T6) — so go through
    # scoring.rebuild, which knows all three, instead of only reading "condition".
    from tslearning.rules import scoring
    periods = scoring.split_facts(facts)
    scoped = periods.get(period, facts)

    cond_specs = rule.parts.get("condition", [])
    cond_from = "vadalog"
    cond_lists = [_read(p, kinds.get(p, "unary_bin")) for p, _role in cond_specs]
    if any(cond_lists):
        condition = (overlap(*cond_lists) if len(cond_lists) > 1 else cond_lists[0])
    else:
        cond_from = "python"
        condition, _conclusion = scoring.rebuild(rule.parts, scoped, rule.constants)
        # In "facts" mode the EDB already carries full-timeline day indices, so the head
        # needs no translation. In "prelabel" mode the EDB restarts at 0 for the period.
        if edb.get("dates"):
            head_ivs = _translate_days(head_ivs, edb["dates"], facts.dates or {})

    kind = rule.parts.get("kind", "conditional")
    cond_days = days(coalesce(condition))

    if kind == "transition":
        # A transition rule fires at the switch INSTANT: the only t with the old regime
        # ending at t and the new one starting at t is the boundary, so the head is a
        # zero-length interval ([1070,1070)). Counting days would score every correct
        # firing as 0 — count EVENTS instead, against the episodes that could trigger.
        d_from = rule.constants.get("d_neg") or rule.constants.get("d_from") or 0
        triggered = sum(1 for s, e in condition if e - s >= d_from)
        hits = len(head_ivs)
        conf = (hits / triggered) if triggered else None
        return {"engine": "vadalog", "period": period, "ran": True,
                "program": prog_path, "edb": edb["path"], "tag": tag,
                "confidence": conf, "n_episodes": triggered, "hit_episodes": hits,
                "condition_days": cond_days, "head_intervals": len(head_ivs),
                "condition_from": cond_from, "metric": "episodes",
                "comparable_to_python": True,
                "eligible": triggered >= 3,
                "reason": None if triggered else f"no episode reaches d={d_from}"}

    hit = days(overlap(coalesce(condition), coalesce(head_ivs)))
    conf = (hit / cond_days) if cond_days else None

    # Both sides now measure DAYS for the kinds that reach here (conditional T1-T4 and
    # persistence T5 — score_persistence reports days as its confidence, keeping episodes
    # as a diagnostic). The transition kind returned earlier, measured in events on both
    # sides. So every comparison is like-for-like.
    return {"engine": "vadalog", "period": period, "ran": True,
            "program": prog_path, "edb": edb["path"], "tag": tag,
            "confidence": conf, "condition_days": cond_days, "hit_days": hit,
            "head_intervals": len(head_ivs), "condition_from": cond_from,
            "metric": "days", "comparable_to_python": True,
            "eligible": cond_days >= 30,
            "reason": None if cond_days else "no condition days"}


def verify(rule, facts=None, pair=None, engine: str = "python", period: str = "test",
           ground: bool = True, run: bool = True, verbose: bool = True,
           keep_body_outputs: bool = False, mode: str = "facts",
           max_gap=float("inf"), run_dir=None) -> dict:
    """Verify one rule. engine = "python" | "vadalog" | "both".

    run_dir: the engine artifacts (program, EDB, derived CSVs) go to run_dir/<tag>/, one
    folder per rule — the EDB file names repeat across rules (edb_<predicate>.csv), so a
    shared folder would leave each program next to another rule's facts.
    """
    pair = pair or rule.pair
    out = {"template": rule.template, "pair": pair, "period": period,
           "rule": rule.render(ground=True), "constants": dict(rule.constants)}

    if engine in ("python", "both"):
        if facts is None:
            from tslearning.rules.generate import load_pair_facts
            facts = load_pair_facts(pair, verbose=False)
        out["python"] = verify_python(rule, facts, period, max_gap)
        if verbose:
            c = out["python"]["confidence"]
            print(f"  python   confidence: {'n/a' if c is None else f'{c:.3f}'}")

    if engine in ("vadalog", "both"):
        tag = f"{rule.template}_{pair}_{period}"
        out["vadalog"] = verify_vadalog(rule, pair, period, ground=ground, run=run,
                                        facts=facts, keep_body_outputs=keep_body_outputs,
                                        mode=mode, verbose=verbose, tag=tag,
                                        result_dir=(Path(run_dir) / tag) if run_dir else None)
        if verbose:
            v = out["vadalog"]
            c = v.get("confidence")
            print(f"  vadalog  confidence: {'n/a' if c is None else f'{c:.3f}'}"
                  + (f"   ({v['reason']})" if v.get("reason") else ""))

    if "python" in out and "vadalog" in out:
        p, v = out["python"].get("confidence"), out["vadalog"].get("confidence")
        same_metric = out["vadalog"].get("comparable_to_python", True)
        out["agreement"] = (None if (p is None or v is None or not same_metric)
                            else abs(p - v))
        if verbose:
            if out["agreement"] is not None:
                verdict = "agree" if out["agreement"] < 0.05 else "DISAGREE"
                print(f"  {verdict}: |python - vadalog| = {out['agreement']:.3f}")
            elif p is not None and v is not None and not same_metric:
                print(f"  not comparable: python counts episodes, vadalog counts days "
                      f"({p:.3f} vs {v:.3f})")
    return out
