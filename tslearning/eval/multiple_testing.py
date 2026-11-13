"""
Multiple testing: could the significant rule(s) be luck, given how many
rules were scored?

For every rule scored on the test period (the rows with a lift in rule_results), the
SAME bootstrap as the results table (`scoring.bootstrap_lift`, same seed, same 2,000
resamples) is read twice:

  95% interval                 the usual one — identical to the table's CI column
  Bonferroni interval          the middle 1 - 0.05/m of the same resamples, m = number
                               of scored rules (Dunn 1961)

A rule "excludes zero" only if its interval is not degenerate (zero width — the
bootstrap failing at a boundary, see bootstrap_lift).

Output: results/multiple_testing[_<source>].{csv,md}, with the summary (m, expected
false positives m x 0.05, how many exclude zero at each level) in the .md.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from tslearning.config import CFG, PATHS


def run(source: str = "py", pairs=None, alpha: float = 0.05, out_dir=None,
        verbose: bool = True) -> dict:
    from tslearning.rules import scoring
    from tslearning.rules.generate import load_pair_facts, score_pair

    pairs = list(pairs or (CFG.get("paper") or {}).get("pairs") or [])
    scored = []
    for pair in pairs:
        facts = load_pair_facts(pair, source, verbose=False)
        test = scoring.split_facts(facts)["test"]
        for r in score_pair(pair, source, verbose=False):
            s = (r.scores or {}).get("test") or {}
            if r.accepted and s.get("lift") is not None and s.get("eligible", True):
                scored.append((pair, r, test))

    m = len(scored)
    a_bonf = alpha / m if m else alpha
    rows = []
    for pair, r, test in scored:
        row = {"Rule": r.template.split("_")[0], "template": r.template, "Pair": pair,
               "Lift": r.scores["test"]["lift"]}
        for tag, a in (("95", alpha), ("bonf", a_bonf)):
            b = scoring.bootstrap_lift(r, test, alpha=a) \
                if (r.parts or {}).get("kind") == "conditional" else {}
            lo, hi = b.get("ci_lo"), b.get("ci_hi")
            row[f"ci_lo_{tag}"], row[f"ci_hi_{tag}"] = lo, hi
            row[f"excludes_zero_{tag}"] = (lo is not None and not b.get("degenerate")
                                           and (lo > 0 or hi < 0))
            if tag == "95":
                row["degenerate"] = bool(b.get("degenerate"))
                row["method"] = b.get("method")
                row["episodes"] = b.get("n_episodes")
        rows.append(row)
    df = pd.DataFrame(rows)

    level = f"{100 * (1 - a_bonf):.2f}%"
    n95 = int(df["excludes_zero_95"].sum()) if m else 0
    nbf = int(df["excludes_zero_bonf"].sum()) if m else 0
    summary = (f"{m} rules scored on the test period ({source} labels). At {1 - alpha:.0%} "
               f"about {m * alpha:.2f} interval(s) would exclude zero by chance alone. "
               f"{n95} exclude zero at 95%; {nbf} still do at the Bonferroni level "
               f"{level} (alpha {alpha}/{m}).")

    out_dir = Path(out_dir or PATHS["results"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "multiple_testing" + ("" if source == "py" else f"_{source}")
    df.to_csv(out_dir / f"{stem}.csv", index=False)

    def _ci(lo, hi):
        return "—" if lo is None or pd.isna(lo) else f"[{lo:+.3f}, {hi:+.3f}]"
    md = ["# Multiple testing", "", summary, "",
          f"| Rule | Pair | Lift | 95% CI | Bonferroni {level} CI | excl. 0 (95%) | "
          f"excl. 0 (Bonf.) | note |", "|---|---|---|---|---|---|---|---|"]
    for _, x in df.iterrows():
        # a rule with no bootstrap (e.g. too few condition days) has no method/episodes
        has_ci = pd.notna(x.get("method")) and pd.notna(x.get("episodes"))
        note = "zero-width CI" if x["degenerate"] else (
            f"{int(x['episodes'])} episode(s), {x['method']} bootstrap" if has_ci else
            "no bootstrap CI")
        md.append(f"| {x['Rule']} | {x['Pair']} | {x['Lift']:+.3f} | "
                  f"{_ci(x['ci_lo_95'], x['ci_hi_95'])} | "
                  f"{_ci(x['ci_lo_bonf'], x['ci_hi_bonf'])} | "
                  f"{'yes' if x['excludes_zero_95'] else 'no'} | "
                  f"{'yes' if x['excludes_zero_bonf'] else 'no'} | {note} |")
    (out_dir / f"{stem}.md").write_text("\n".join(md) + "\n")
    if verbose:
        print(summary)
        print(f"wrote -> {out_dir / stem}.csv, .md")
    return {"m": m, "summary": summary, "table": df, "path": out_dir / f"{stem}.md"}
