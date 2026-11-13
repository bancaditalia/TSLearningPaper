#!/usr/bin/env python3
"""
One compact paper table with the rule results of BOTH label sources side by side:
exact (Python) labels and classifier labels.

    python scripts/combined_rule_table.py --run <run>

Reads the run's rule_results_pub.csv and rule_results_classifier_pub.csv (written by
`paper.py tables`) and recomputes nothing. Writes, next to them, without touching them:

    rule_results_combined_pub.{csv,md,tex}

Columns: Rule | Pair | constants | lift [95% CI] and verdict per label source. A rule
that appears in only one of the two pub tables shows "—" for the other. Significant
cells are in bold. The pair names drop "S&P 500 –" (every pair is the S&P 500 against
the asset named); the caption says so.
"""
from _cli import bootstrap, parser

bootstrap()

SHORT_VERDICT = {"significant": "sig.", "positive lift": "pos.", "negative": "neg.",
                 "negligible": "negl.", "ineligible": "inel.",
                 "near-vacuous (baseline high)": "vac.", "near-vacuous": "vac.",
                 "not scored": "n.s.", "not fitted": "n.f."}


def main(argv=None):
    p = parser(__doc__.strip().split("\n")[0], epilog=__doc__)
    p.add_argument("--fontsize", default="",
                   help="LaTeX size command without backslash, e.g. footnotesize "
                        "(default: none; the table is scaled to the text width)")
    p.add_argument("--dir", default=None,
                   help="folder holding the two pub tables, also where the combined one "
                        "is written (default: the run's results folder)")
    p.add_argument("--drop-empty", action="store_true",
                   help="leave out rules ineligible under both label sources (e.g. T5, "
                        "T6); default: keep every rule of the two pub tables")
    args = p.parse_args(argv)

    import pandas as pd
    from tslearning.config import PATHS
    from tslearning.rules.tables import _run_tag, _template_sort_key, periods, to_markdown

    from pathlib import Path
    from tslearning.config import REPO_ROOT
    res = Path(args.dir) if args.dir else PATHS["results"]
    res = res if res.is_absolute() else REPO_ROOT / res
    fp, fc = res / "rule_results_pub.csv", res / "rule_results_classifier_pub.csv"
    for f in (fp, fc):
        if not f.exists():
            p.error(f"{f} not found — run `paper.py tables` for this run first")
    py, cl = pd.read_csv(fp), pd.read_csv(fc)
    # The pub tables choose WHICH rules appear (a rule shown in either), but leave out
    # some rows (e.g. "ineligible", "not fitted"), which used to show as "—" here even
    # when the rule had been fitted. The cells therefore come from the FULL tables,
    # rule_results[_classifier].csv in the same folder, when they exist.
    full_p, full_c = res / "rule_results.csv", res / "rule_results_classifier.csv"
    fpy = pd.read_csv(full_p) if full_p.exists() else py
    fcl = pd.read_csv(full_c) if full_c.exists() else cl

    def verdict(v):
        v = "" if pd.isna(v) else str(v).split(":")[0].split(" (")[0].strip()
        return SHORT_VERDICT.get(v, v or "—")

    def lift_ci(r):
        if r is None or pd.isna(r.get("Lift")) or str(r.get("Lift")) == "—":
            return "—"
        ci = r.get("CI")
        ci = "" if pd.isna(ci) or str(ci) == "—" else f" {ci}"
        return f"{float(r['Lift']):+.3f}{ci}"

    shown = {(r.Rule, r.Pair) for d in (py, cl) for r in d.itertuples(index=False)}
    keyed_p = {(r.Rule, r.Pair): r._asdict() for r in fpy.itertuples(index=False)}
    keyed_c = {(r.Rule, r.Pair): r._asdict() for r in fcl.itertuples(index=False)}
    keys = sorted(shown, key=lambda k: (_template_sort_key(k[0]), k[1]))

    def const_of(r):
        c = r["Constants"] if r else None
        return None if c is None or pd.isna(c) or str(c) == "—" else c

    rows = []
    for k in keys:
        a, b = keyed_p.get(k), keyed_c.get(k)
        ca, cb = const_of(a), const_of(b)
        const = ca if (cb is None or ca == cb) else (cb if ca is None else f"{ca} / {cb}")
        rows.append({
            "Rule": k[0],
            "Pair": str(k[1]).replace("S&P 500 – ", ""),
            "Constants": "—" if const is None or pd.isna(const) else const,
            "Lift [CI], exact": lift_ci(a) if a else "—",
            "Verdict, exact": verdict(a["Verdict"]) if a else "—",
            "Lift [CI], classifier": lift_ci(b) if b else "—",
            "Verdict, classifier": verdict(b["Verdict"]) if b else "—",
        })
    out = pd.DataFrame(rows)
    # --drop-empty: rows with no scored result under either source (e.g. T5/T6 "inel."
    # twice) carry no number; leave them out to save space, and the caption says so
    dropped_note = ""
    if args.drop_empty:
        empty = {"—", "inel.", "n.s.", "n.f."}
        dropped = out[out["Verdict, exact"].isin(empty)
                      & out["Verdict, classifier"].isin(empty)]
        out = out.drop(dropped.index).reset_index(drop=True)
        dropped_note = (f" Not shown, ineligible under both label sources: "
                        f"{', '.join(sorted(set(dropped.Rule)))}." if len(dropped) else "")

    caption = periods(
        "Rule results, exact / classifier labels; pairs: S\\&P 500 vs the asset named; "
        "fitted on <TRAIN>, scored on <TEST>. \\emph{Lift} = test confidence $-$ base "
        "rate, 95\\% bootstrap CI. "
        "sig.: lift $\\geq 0.05$, CI excl.\\ zero; pos.: lift $\\geq 0.05$; negl.: "
        "$|$lift$| < 0.05$; neg.: lift $\\leq -0.05$; inel.: too few condition days; "
        "n.f.: not fitted."
        + dropped_note)

    out.to_csv(res / "rule_results_combined_pub.csv", index=False)
    (res / "rule_results_combined_pub.md").write_text(
        caption.replace("\\&", "&").replace("\\%", "%").replace("\\ ", " ")
        .replace("$\\geq", "≥").replace("$\\leq", "≤").replace("$-$", "−")
        .replace("$|$", "|").replace("$", "").replace("\\emph{Lift}", "Lift")
        .replace("---", "—")
        + "\n\n" + to_markdown(out) + "\n")

    def esc(s):
        s = str(s).replace("–", "--").replace("—", "--").replace("&", "\\&")
        s = s.replace("_", "\\_").replace("%", "\\%")
        import re
        return re.sub(r"([A-Za-z])⁻", r"$\1^-$", s)

    def cell(s, verdict_col):
        s = esc(s)
        return f"\\textbf{{{s}}}" if verdict_col == "sig." else s

    lines = [_run_tag().rstrip("\n")] if _run_tag() else []
    lines += ["\\begin{figure}[t]", "\\centering"]
    if args.fontsize:
        lines.append(f"\\{args.fontsize}")
    lines += ["\\resizebox{\\textwidth}{!}{%",
              "\\begin{tabular}{lllrlrl}", "\\toprule",
              " & & & \\multicolumn{2}{c}{Exact labels} & "
              "\\multicolumn{2}{c}{Classifier labels} \\\\",
              "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
              "Rule & Pair & Constants & Lift [95\\% CI] & Verdict & Lift [95\\% CI] & "
              "Verdict \\\\", "\\midrule"]
    def const_cell(s):
        # two label sources with different constants ("exact / classifier"): one per
        # line, so a long pair (T5/T6) does not widen the whole table. A nested tabular
        # aligned at its top line: plain LaTeX, no package, and the row grows downward
        # (\shortstack overlapped the rows above).
        # Also a new line after every 2 constants (T6 has three). The classifier's
        # constants, when different, start on their own line marked "/".
        # One continuous sequence, 2 constants per line: exact constants, then "/", then
        # the classifier's, e.g. "d=37, a=26," / "b=115 / d=2," / "a=238, b=247".
        tokens = []
        for i, src in enumerate(str(s).split(" / ")):
            consts = [c.strip() for c in src.split(",")]
            if i:
                consts[0] = "/ " + consts[0]
            tokens += consts

        def joined(ts):
            out_ = ts[0]
            for t_ in ts[1:]:
                out_ += (" " if t_.startswith("/ ") else ", ") + t_
            return out_

        lines_ = [joined(tokens[j:j + 2]) for j in range(0, len(tokens), 2)]
        lines_ = [ln + ("" if k == len(lines_) - 1 or tokens[2 * (k + 1)].startswith("/ ")
                        else ",") for k, ln in enumerate(lines_)]
        if len(lines_) == 1:
            return esc(lines_[0])
        return ("\\begin{tabular}[t]{@{}l@{}}" + " \\\\ ".join(esc(x) for x in lines_)
                + "\\end{tabular}")

    for r in out.itertuples(index=False):
        lines.append(" & ".join([
            esc(r[0]), esc(r[1]), const_cell(r[2]),
            cell(r[3], r[4]), cell(r[4], r[4]),
            cell(r[5], r[6]), cell(r[6], r[6])]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}",
              f"\\caption{{\\small{{{caption}}}}}", "\\label{fig:rule-results-combined}",
              "\\end{figure}"]
    (res / "rule_results_combined_pub.tex").write_text("\n".join(lines) + "\n")

    if not args.quiet:
        print(to_markdown(out))
    print(f"{len(out)} rows -> {res / 'rule_results_combined_pub'}.{{csv,md,tex}}")
    return out


if __name__ == "__main__":
    main()
