"""
The LLM-proposed rules as framed LaTeX boxes, like the rule-library box.

    llm_rules_box_<run ids>.tex       every distinct rule each model proposed (parsed)
    llm_rules_box_pub_<run ids>.tex   per model, the rules shown in the comparison table:
                                      the top k by TRAIN lift (config paper.llm.top_k)

The run ids (the timestamps of the pooled runs) are in the file name, as for the
llm_rules_<run ids>.txt listing, so a new set of runs writes new files and never
overwrites the boxes of an earlier set.

Selection is the same as the comparison table's — by TRAIN lift, never by test results —
so the _pub box shows exactly the rules the table scores. Each rule carries its train and
test lift, its verdict and whether it is about the pair, computed by the same code as the
table.
"""
from __future__ import annotations

import re
from pathlib import Path

_OPS = {"<+>": r"\diamondplus", "<->": r"\diamondminus",
        "[+]": r"\boxplus", "[-]": r"\boxminus"}
_ATOM = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)")
_OP = re.compile(r"(<[+-]>|\[[+-]\])\s*([\[(])\s*(-?\d+)\s*,\s*(-?\d+)\s*([\])])")


def rule_latex(rule: str) -> str:
    """'head :- body' (LLM syntax) -> '$body \\to head$' in the paper's notation."""
    head, _, body = str(rule).partition(":-")
    if not body:
        head, body = "", head

    def conv(s: str) -> str:
        s = _OP.sub(lambda m: f"{_OPS[m.group(1)]}_{{{m.group(2)}{m.group(3)},"
                              f"{m.group(4)}{m.group(5)}}} ", s)
        s = _ATOM.sub(lambda m: rf"\mathit{{{m.group(1)}}}({m.group(2).replace(' ', '')})",
                      s)
        return " ".join(s.split()).replace(" ,", ",")

    out = conv(body.strip().rstrip("."))
    if head.strip():
        out += r" \to " + conv(head.strip())
    return f"${out}$"


def _rule_lines(latex: str, max_chars: int, visible_len) -> "list[str]":
    """One \\mbox per line; a long rule breaks before '→ head' (as in the library box)."""
    if visible_len(latex) <= max_chars or r"\to" not in latex:
        return [rf"\mbox{{{latex}}}"]
    body, head = latex.rsplit(r"\to", 1)
    body = body.strip().lstrip("$").rstrip()
    # pack the body's atoms (split after a closing parenthesis) into lines that fit
    atoms = re.split(r"(?<=\)),\s*", body)
    lines, cur = [], ""
    for at in atoms:
        cand = f"{cur}, {at}" if cur else at
        if cur and visible_len(cand) > max_chars:
            lines.append(cur + ",")
            cur = at
        else:
            cur = cand
    lines.append(cur)
    out = [rf"\mbox{{${lines[0]}$}}"]
    out += [rf"\hspace*{{1.5em}}\mbox{{${l}$}}" for l in lines[1:]]
    out.append(rf"\hspace*{{3em}}\mbox{{$\to{head}}}")
    return out


def box(bodies, info: dict, per_model: dict, selected_only: bool,
        caption: str, label: str, limit: "int | None" = None) -> str:
    from tslearning.rules.tables import _BOX_MAX_CHARS, _visible_len

    lines = []
    models = list(dict.fromkeys(b["who"] for b in bodies))
    for mi, who in enumerate(models):
        rows = [b for b in bodies if b["who"] == who
                and (b.get("selected") or not selected_only)]
        if selected_only:     # in the order they were selected: by train lift
            rows.sort(key=lambda b: -(b["train_lift"] if b["train_lift"] is not None
                                     else float("-inf")))
        if limit:             # the box may show fewer rules than the comparison table
            rows = rows[:limit]
        if not rows:
            continue
        pm = per_model.get(who, {})
        temps = sorted({("default sampling" if t is None else f"temperature {t:g}")
                        for t in pm.get("temps", [])})
        meta = (f"{pm.get('runs', '?')} run{'s' if pm.get('runs') != 1 else ''}"
                + (f", {', '.join(temps)}" if temps else ""))
        name = who.removeprefix("LLM ").replace("_", r"\_")
        lines.append(rf"\textbf{{{name}}}\hspace{{1em plus 1fill}}"
                     rf"{{\footnotesize {meta}}}\\[2pt]")
        for j, b in enumerate(rows):
            i = info.get((who, b["id"]), {})
            tl = "n/a" if b["train_lift"] is None else f"{b['train_lift']:+.3f}"
            te = i.get("lift")
            te = "n/a" if te is None else f"{te:+.3f}"
            bits = [f"train lift {tl}", f"test lift {te}"]
            if i.get("verdict"):
                bits.append(i["verdict"])
            if i.get("about") == "no":
                bits.append("one series only")
            if b.get("flags"):
                bits.append("head reuses a vocabulary predicate"
                            if any("vocabulary" in f for f in b["flags"]) else "")
            desc = "; ".join(x for x in bits if x)
            lines.append(rf"\textbf{{{b['id']}}}\quad{{\footnotesize {desc}}}\\")
            parts = _rule_lines(rule_latex(b["rule"]), _BOX_MAX_CHARS - 4, _visible_len)
            last_rule = j == len(rows) - 1
            for q, part in enumerate(parts):
                end = q == len(parts) - 1
                gap = r"\\" if not end else (
                    "" if (last_rule and mi == len(models) - 1)
                    else (r"\\[8pt]" if last_rule else r"\\[4pt]"))
                lines.append(rf"\hspace*{{1.5em}}{part}{gap}")
    return "\n".join([
        r"\begin{figure}[t]", r"\centering",
        r"\fbox{\begin{minipage}{0.95\textwidth}\small",
        *lines,
        r"\end{minipage}}",
        rf"\caption{{\small{{{caption}}}}}", rf"\label{{{label}}}", r"\end{figure}"])


def write(out_dir, run_ids, bodies, info, per_model, top_k: int,
          box_top_k: "int | None" = None) -> "list[Path]":
    """Write the full and the _pub box; returns the paths written."""
    tag = "_".join(run_ids) if run_ids else "latest"
    note = (" Rules are written body $\\to$ head, with the window exactly as the model "
            "wrote it. Lifts, verdicts and the pair check use the templates' code.")
    out = []
    full = Path(out_dir) / f"llm_rules_box_{tag}.tex"
    full.write_text(box(bodies, info, per_model, False,
                        "Every distinct rule proposed by each language model, pooled over "
                        "its runs of the same prompt." + note,
                        "fig:llm-rules") + "\n")
    out.append(full)
    pub = Path(out_dir) / f"llm_rules_box_pub_{tag}.tex"
    pub.write_text(box(bodies, info, per_model, True,
                       (f"LLM rules in the comparison: per model, the {top_k} with the "
                        f"highest \\emph{{train}} lift over its runs."
                        if not box_top_k or box_top_k >= top_k else
                        f"LLM rules: per model, the {box_top_k} with the highest "
                        f"\\emph{{train}} lift of the {top_k} in the comparison.")
                       + note,
                       "fig:llm-rules-pub", limit=box_top_k) + "\n")
    out.append(pub)
    return out
