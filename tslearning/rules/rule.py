"""
The Rule object and its rendering — including GROUNDING.

A template produces a Rule whose head/body carry `{X}` and `{Y}` placeholders. Rendering
substitutes them, and that is the only place the variable-vs-ground decision is made.

────────────────────────────────────────────────────────────────────────────────
GROUNDING — the whole mechanism is `Rule.render(ground=...)` below
────────────────────────────────────────────────────────────────────────────────

    render(ground=False)   safeHavenEpisode(Y,X) :- highVol(X), ...
    render(ground=True)    safeHavenEpisode("gold","sp500") :- highVol("sp500"), ...

Why it matters: a rule written with variables reads
as a claim about ALL pairs, but the constants were fitted on ONE substitution — the
variables are decoration unless you evaluate the frozen rule on other ground pairs and
report support. Two honest options:

  * claim about one pair  -> render ground, validate with train/test confidence on it
  * claim about a class   -> render with variables, and earn them with support + stability

Ground is the safer default for a single-pair result, which is what the paper's punchline
("was Bitcoin a safe haven for US equities") actually is.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Rule:
    """A mined rule: template identity, the ground pair it came from, the fitted
    constants, and the diagnostics behind them.

    `head` and `body` are format strings containing {X} and {Y}. Constants are already
    interpolated by the template — only the series terms stay symbolic, so that grounding
    remains a single decision made at render time.
    """
    template: str
    pair: str
    x: str
    y: str
    head: str = ""
    body: str = ""
    constants: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    calendar_days: dict = field(default_factory=dict)
    accepted: bool = True
    reason: "str | None" = None
    # Declarative description of what to measure, so the rule can be RE-SCORED on a
    # different period (or pair) without re-running the template. See rules/scoring.py.
    #   {"kind": "conditional", "condition": [(pred, "x"|"y"|"xy"), ...],
    #    "conclusion": (pred, "xy"), "direction": "+", "lag": None}
    #   {"kind": "persistence", "pred": (pred, "xy")}
    #   {"kind": "transition",  "from": (pred,"xy"), "to": (pred,"xy")}
    parts: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    # ── engine-executable variant (optional) ─────────────────────────────────
    # `head`/`body` are the compact, readable form used in the paper and the results
    # table. If the engine needs something different, set these; verification uses them
    # instead. Leave them None and verify.py derives the executable form automatically by
    # padding each atom to the arity declared in the .vada (highVol(X) -> highVol(X,_)).
    # Set them only when that automatic fix is not what you want.
    vada_head: "str | None" = None
    vada_body: "str | None" = None
    # Helper rules the body needs (e.g. a conjunction under a temporal operator must be
    # named first: stressedDown(X) :- highVol(X), trendDown(X)). Same {X}/{Y} format as
    # head/body; verify.py writes them into the program before the rule itself.
    vada_aux: list = field(default_factory=list)

    def render_aux(self, ground: bool = False) -> list:
        """The helper rules with terms substituted, one string per rule."""
        t = self.terms(ground)
        return [a.format(**t) for a in self.vada_aux]

    # ── grounding ────────────────────────────────────────────────────────────
    def terms(self, ground: bool) -> dict:
        """The substitution applied to {X}/{Y}. Ground terms are quoted, as DatalogMTL
        constants must be; variables are bare uppercase."""
        if ground:
            return {"X": f'"{self.x}"', "Y": f'"{self.y}"'}
        return {"X": "X", "Y": "Y"}

    def render_parts(self, ground: bool = False, vadalog: bool = False) -> "tuple[str, str]":
        """(head, body) with terms substituted.

        `vadalog=True` prefers `vada_head`/`vada_body` when a template supplied them —
        the engine-executable variant, for when the compact readable form isn't valid
        DatalogMTL (e.g. arity, or a helper predicate the body needs).
        """
        t = self.terms(ground)
        head = (self.vada_head or self.head) if vadalog else self.head
        body = (self.vada_body or self.body) if vadalog else self.body
        return head.format(**t), body.format(**t)

    def render(self, ground: bool = False, vadalog: bool = False) -> str:
        """The DatalogMTL rule as text."""
        if not self.accepted:
            return ""
        head, body = self.render_parts(ground, vadalog)
        return f"{head} :- {body}."

    # ── constructors / serialisation ─────────────────────────────────────────
    @classmethod
    def rejected(cls, template: str, facts, reason: str, diagnostics=None) -> "Rule":
        """A template that ran but found no effect. Kept rather than dropped: 'the rule
        does not hold for this pair' is a reportable result, and the contrast between
        pairs that pass and fail is itself a finding (§3.3)."""
        return cls(template=template, pair=facts.pair, x=facts.x, y=facts.y,
                   accepted=False, reason=reason, diagnostics=diagnostics or {})

    def to_row(self, ground: bool = False) -> dict:
        """Flat dict for a results table."""
        row = {"template": self.template, "pair": self.pair, "x": self.x, "y": self.y,
               "accepted": self.accepted, "reason": self.reason,
               "rule": self.render(ground)}
        row.update({f"const_{k}": v for k, v in self.constants.items()})
        row.update({f"cal_{k}": v for k, v in self.calendar_days.items() if v is not None})
        row.update({f"diag_{k}": v for k, v in self.diagnostics.items()
                    if not isinstance(v, dict)})
        for period, sc in self.scores.items():
            row.update({f"{period}_{k}": v for k, v in sc.items()
                        if not isinstance(v, dict)})
        return row

    def __str__(self) -> str:
        if not self.accepted:
            return f"[{self.template}] {self.pair}: rejected — {self.reason}"
        consts = ", ".join(f"{k}={v}" for k, v in self.constants.items())
        return f"[{self.template}] {self.pair}: {self.render(ground=True)}  ({consts})"
