"""
LLM rule proposal — parse a completion into Rule objects and score them like any other.

────────────────────────────────────────────────────────────────────────────────
WHY THIS IS CHEAP TO BUILD, AND WHY THAT MATTERS
────────────────────────────────────────────────────────────────────────────────

The scoring layer is source-agnostic: `scoring.score_train_test` takes a `Rule` and has no
idea whether a template fitter or a language model produced it. So the LLM arm is not a
parallel evaluation pipeline — it is one more rule *proposer* feeding the existing one.

That is also the honest framing for the paper. The comparison is not "LLM versus us"; it
is "an LLM proposing rules in our language, scored by our machinery". The contribution
being measured is the LANGUAGE and the SCORING, with the template fitter and the model as
two proposers judged on identical terms:

    fit on train  ->  freeze the constants  ->  score on test  ->  report lift

────────────────────────────────────────────────────────────────────────────────
WHAT COUNTS AS A FAILURE, AND WHY FAILURES ARE RESULTS
────────────────────────────────────────────────────────────────────────────────

Every way a proposal can fail is recorded rather than skipped, because each says something
different about what a model can and cannot do:

  malformed JSON        it could not follow an explicit output contract
  unknown predicate     it invented vocabulary — the rule is unscoreable
  unparseable body      it wrote something outside the language it was given
  parsed but rejected   the rule is well-formed and the data does not support it

The last one is the interesting case, and the expected failure mode is a plausible
template with a badly fitted constant: strong in-sample, weak out. That is precisely what
the 1-SE rule exists to prevent, so a gap there is evidence for the fitting procedure, not
merely against the model.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from tslearning.rules.rule import Rule

# atom, optionally prefixed by a temporal operator:  <+>[0,5] isNegativeCorrelation(X,Y)
_ATOM = re.compile(r"""
    ^\s*
    (?P<op>(?:<[+-]>|\[[+-]\])\s*[\[(]\s*(?P<a>-?\d+)\s*,\s*(?P<b>-?\d+)\s*[\])])?
    \s*
    (?P<pred>[A-Za-z_][A-Za-z0-9_]*)
    \s*\(\s*(?P<args>[^)]*)\s*\)
    \s*$""", re.X)


def _split_atoms(body: str) -> "list[str]":
    """Split a rule body on top-level commas (those outside parentheses/brackets)."""
    out, depth, cur = [], 0, ""
    for ch in body:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return [a.strip() for a in out if a.strip()]


_HEAD = re.compile(r"^\s*(?P<pred>[A-Za-z_][A-Za-z0-9_]*)\s*\(\s*(?P<args>[^)]*)\)\s*$")


def normalise_head(head: str, body: str) -> "tuple[str, list]":
    """Drop head arguments the body never binds -> (head, dropped).

    `myLead(X,Y,k) :- …, leadsPositively(X,Y,1)` leaves k unbound: an unsafe rule, which
    the engine refuses. That is rule-FORMING, not the proposal we are testing, so it is
    repaired rather than rejected, and the repair is recorded. Constants (quoted strings,
    numbers) are kept."""
    m = _HEAD.match(head or "")
    if not m:
        return head, []
    bound = set()
    for a in _split_atoms(body or ""):
        am = _ATOM.match(a)
        if am:
            bound |= {t.strip() for t in am.group("args").split(",")}
    keep, dropped = [], []
    for t in [t.strip() for t in m.group("args").split(",") if t.strip()]:
        if t.strip("{}") in bound or t.startswith('"') or t.lstrip("-").isdigit():
            keep.append(t)
        else:
            dropped.append(t)
    if not dropped:
        return head, []
    return f"{m.group('pred')}({','.join(keep)})", dropped


def head_reuses_vocabulary(head: str, known_predicates) -> "str | None":
    """The vocabulary predicate the head redefines (`trendDown(X) :- …`), or None.

    NOT repaired and NOT rejected: such a rule may score well, but it restates the
    vocabulary instead of proposing a relation — an LLM failure, reported in the
    discussion. Flagged so the listing shows it."""
    m = _HEAD.match(head or "")
    return m.group("pred") if m and m.group("pred") in set(known_predicates) else None


def _side(tok: str, name_a: str, name_b: str) -> "str | None":
    if tok in ("x", name_a.lower()):
        return "x"
    if tok in ("y", name_b.lower()):
        return "y"
    return None


def _role(args: str, name_a: str, name_b: str, pred: str = "",
          economy_of: "dict | None" = None) -> str:
    """Which side of the pair an atom refers to: 'x', 'y', 'xy', 'xc'/'yc' (market),
    'cx'/'cy' (a predicate on that side's economy, via market) or 'macro'.

    `economy_of` maps an economy variable (e.g. 'c') to the side market() bound it to.
    """
    toks = [t.strip().strip('"').lower() for t in args.split(",") if t.strip()]
    toks = [t for t in toks if not t.isdigit()]
    if pred == "market" and len(toks) == 2:
        side = _side(toks[0], name_a, name_b)
        return f"{side}c" if side else "macro"
    if len(toks) == 1 and economy_of and toks[0] in economy_of:
        return f"c{economy_of[toks[0]]}"
    if len(toks) >= 2:
        # a series paired with ITSELF — tailDependent(X,X) — is not the pair: it has no
        # facts, so it must not be read as tailDependent(X,Y) (which turned an LLM rule
        # into T2 contagion, 2026-10-06)
        s0, s1 = _side(toks[0], name_a, name_b), _side(toks[1], name_a, name_b)
        if s0 and s0 == s1:
            return s0 * 2
        return "xy"
    if not toks:
        return "macro"
    t = toks[0]
    if t in ("x", name_a.lower()):
        return "x"
    if t in ("y", name_b.lower()):
        return "y"
    return "macro"


def _window_length(series: str) -> int:
    """The window length (days/months) of a series' frequency, from the config."""
    from tslearning.config import CFG
    freq = {k.lower(): v for k, v in CFG.get("series_freq", {}).items()}.get(
        str(series).lower(), "financial")
    return int(CFG["windows"].get(freq, CFG["windows"]["financial"])["length"])


def parse_body(body: str, name_a: str, name_b: str, known_predicates) -> dict:
    """Body string -> the `parts` dict the scorer needs, plus any constants found.

    Only the CONDITIONAL shape is parsed: unprefixed condition atoms and one
    operator-prefixed conclusion, with a diamond (<+>, <->) or a box ([+], [-]) operator.
    "P, [+][0,d] P" is therefore T5's persistence shape, scored the same way (days).
    """
    atoms = _split_atoms(body)
    if not atoms:
        raise ValueError("empty body")

    from tslearning.config import CFG
    kinds = CFG["predicates"]["kind"]

    condition, conclusion, direction, window, unknown = [], None, "+", None, []
    window_lo = 0
    lag = None
    # first pass: market(X, C) binds the economy variable C to a side of the pair, so a
    # later inflationHigh(C) is read as "the economy of X"
    economy_of = {}
    for a in atoms:
        m = _ATOM.match(a)
        if m and m.group("pred") == "market":
            toks = [t.strip().strip('"').lower() for t in m.group("args").split(",")]
            if len(toks) == 2 and _side(toks[0], name_a, name_b):
                economy_of[toks[1]] = _side(toks[0], name_a, name_b)
    for a in atoms:
        m = _ATOM.match(a)
        if not m:
            raise ValueError(f"cannot parse atom {a!r}")
        pred, args, op = m.group("pred"), m.group("args"), m.group("op")
        if pred not in known_predicates:
            unknown.append(pred)
            continue
        # A lag belongs to the BODY, not to a separate constants dict. Only a genuine lag
        # predicate carries one — `leadsPositively(X,Y,3)`. A stray integer on a
        # two-argument predicate (`isNoCorrelation(Y,X,1)`, which a model did write) is
        # noise and must not turn into a lag lookup: it would search for a fact that
        # cannot exist and score a confidence of 0 against a baseline of 0.
        nums = [t.strip() for t in args.split(",") if t.strip().lstrip("-").isdigit()]
        if nums and kinds.get(pred, "").startswith("lag"):
            lag = int(nums[-1])
        role = _role(args, name_a, name_b, pred, economy_of)
        if op is None:
            condition.append((pred, role))
        else:
            if conclusion is not None:
                raise ValueError("more than one operator-prefixed atom; "
                                 "the conditional shape allows exactly one conclusion")
            # <±> = at some point in the window; [±] = on every day of it (as in T5/T6)
            box = op.strip().startswith("[")
            direction = "-" if op.strip()[1] == "-" else "+"
            window = int(m.group("b"))
            window_lo = int(m.group("a"))
            conclusion = (pred, role)

    if unknown:
        raise ValueError(f"unknown predicate(s): {sorted(set(unknown))}")
    if conclusion is None:
        raise ValueError("no operator-prefixed conclusion atom found")
    if not condition:
        raise ValueError("no condition atoms — a rule with no condition is vacuous "
                         "by construction (its confidence is its baseline)")
    # A conclusion that restates a condition atom (same predicate, same side — pair
    # predicates are symmetric, so (X,Y) and (Y,X) count as the same) is satisfied by the
    # condition's own episode unless the offset reaches past the window length: every
    # predicate is a per-WINDOW fact, so once it holds it holds for >= seq_len days
    # (the smear). "tailDependent now -> tailDependent 10-20 days later" scored +0.80 lift
    # this way (2026-10-01) without saying anything about the data.
    seq_len = _window_length(name_a)
    # Diamond only: "P now, [+][0,d] P" is persistence (T5's shape) — it asks whether P
    # LASTS, which the condition's own episode does not guarantee, so it is not vacuous.
    if not box and any(c[0] == conclusion[0] and c[1] == conclusion[1] for c in condition) \
            and window_lo < seq_len:
        raise ValueError(f"vacuous: the conclusion {conclusion[0]} restates a condition "
                         f"atom within the window length ({seq_len} days), so the same "
                         f"episode satisfies it")
    if window_lo > window:
        raise ValueError(f"empty window [{window_lo},{window}]")

    return {"parts": {"kind": "conditional", "condition": condition,
                      "conclusion": conclusion, "direction": direction,
                      "op": "box" if box else "diamond"},
            "w": window, "w_lo": window_lo, "k": lag}


def to_rules(payload: dict, pair: str, x: str, y: str,
             known_predicates) -> "tuple[list, list]":
    """(rules, rejections). Rejections carry the reason, and are reported, not dropped."""
    rules, bad = [], []
    for i, r in enumerate(payload.get("rules") or []):
        name = str(r.get("name") or f"LLM{i + 1}")
        tag = f"LLM_{name}"
        body = (r.get("body") or "").strip()
        head = (r.get("head") or f"{name}({{X}},{{Y}})").strip()
        try:
            if not body:
                raise ValueError("no body")
            parsed = parse_body(body, x, y, known_predicates)
        except Exception as e:
            bad.append({"name": tag, "body": body, "reason": str(e),
                        "phenomenon": r.get("phenomenon"),
                        "rationale": r.get("rationale")})
            continue

        # THE BODY IS AUTHORITATIVE. The model also returns a "constants" dict, and it
        # does not always agree with what it wrote — one reply carried {"k": 1, "w": 10}
        # for a body with no lag atom, which sent the scorer looking for a lagged fact
        # that cannot exist and produced a spurious 0.00 confidence against a 0.00
        # baseline. Take only the constants the parse actually found; keep the model's
        # own dict as a diagnostic, since the disagreement is itself informative.
        consts = {"w": parsed["w"]}
        if parsed.get("w_lo"):
            consts["w_lo"] = parsed["w_lo"]
        if parsed.get("k") is not None:
            consts["k"] = parsed["k"]
        declared = dict(r.get("constants") or {})
        mismatch = {k: v for k, v in declared.items() if consts.get(k) != v}
        head, unbound = normalise_head(head, body)
        reused = head_reuses_vocabulary(head, known_predicates)
        # placeholders so the rule renders and grounds like any other
        for a, ph in ((x, "{X}"), (y, "{Y}")):
            body = re.sub(rf'(?<![A-Za-z_"]){re.escape(a)}(?![A-Za-z0-9_"])', ph, body)
            head = re.sub(rf'(?<![A-Za-z_"]){re.escape(a)}(?![A-Za-z0-9_"])', ph, head)
        body = re.sub(r"(?<![A-Za-z_{])X(?![A-Za-z0-9_}])", "{X}", body)
        body = re.sub(r"(?<![A-Za-z_{])Y(?![A-Za-z0-9_}])", "{Y}", body)
        head = re.sub(r"(?<![A-Za-z_{])X(?![A-Za-z0-9_}])", "{X}", head)
        head = re.sub(r"(?<![A-Za-z_{])Y(?![A-Za-z0-9_}])", "{Y}", head)

        rules.append(Rule(template=tag, pair=pair, x=x, y=y, head=head, body=body,
                          constants=consts, parts=parsed["parts"],
                          diagnostics={"source": "llm",
                                       "phenomenon": r.get("phenomenon"),
                                       "rationale": r.get("rationale"),
                                       "declared_constants": declared,
                                       "constants_mismatch": mismatch or None,
                                       "unbound_head_dropped": unbound or None,
                                       "head_reuses_vocabulary": reused}))
    return rules, bad


def score_llm_rules(rules, facts, train=None, test=None, verbose: bool = True) -> list:
    """Score proposed rules on train and test with the constants the model chose.

    NOTE the asymmetry, and state it in the paper: the template fitter SELECTS its
    constants on the training data by the 1-SE rule, whereas the model asserts them. That
    is the comparison being made — not two fitters, but a fitted proposer against an
    asserted one. Refitting the model's constants would measure something else (and would
    make it a template library with extra steps).
    """
    from tslearning.rules import scoring

    out = []
    for r in rules:
        try:
            r.scores = scoring.score_train_test(r, facts, refit=None,
                                                train=train, test=test)
        except Exception as e:
            r.accepted, r.reason = False, f"scoring failed: {e}"
        out.append(r)
        if verbose:
            s = r.scores.get("test", {}) if r.scores else {}
            lift = s.get("lift")
            print(f"  {r.template:<28} "
                  + (f"test {s.get('confidence', float('nan')):.2f} "
                     f"lift {lift:+.3f} n={s.get('condition_days', 0)}"
                     if lift is not None else f"not scored ({r.reason or 'ineligible'})"))
    return out


def save(rules, bad, meta: dict, out_path) -> Path:
    """Write proposals, scores and rejections as one JSON record of the run."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"meta": meta,
           "accepted": [{"template": r.template, "head": r.head, "body": r.body,
                         "constants": r.constants, "scores": r.scores,
                         "diagnostics": r.diagnostics} for r in rules],
           "rejected": bad}
    out_path.write_text(json.dumps(rec, indent=2, default=str))
    return out_path
