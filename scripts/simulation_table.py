#!/usr/bin/env python3
"""
Paper table of the simulation study: for each planted phenomenon and its two rule
templates (forward, backward), the share of replications flagged significant

  * with nothing planted (theta = 0): false alarms,
  * with the phenomenon planted (theta > 0): power,

at two sample lengths, next to the econometric test's rate on the same samples.

    python scripts/simulation_table.py --run <run>
    python scripts/simulation_table.py --run <run> --years 10,20,40

Reads <results>/simulation/summary.csv (scripts/simulate_rules.py) and recomputes
nothing. Writes simulation_pub.{csv,md,tex} into the same simulation/ folder.
"""
from _cli import bootstrap, csv_list, parser

bootstrap()

PHENOMENON = {"T1b": "Safe haven", "T2": "Contagion", "T3b": "Decoupling",
              "T4": "Lead under stress", "T7": "Macro (inflation)"}
ORDER = ["T1b", "T2", "T3b", "T4", "T7"]
DAYS_PER_YEAR = 252


def main(argv=None):
    p = parser(__doc__.strip().split("\n")[0], epilog=__doc__)
    p.add_argument("--years", type=csv_list, default=["10", "40"],
                   help="sample lengths in years, comma separated (default 10,40; "
                        "the datasets have 10, 20, 40)")
    args = p.parse_args(argv)

    import pandas as pd
    from tslearning.config import PATHS
    from tslearning.rules.tables import _run_tag, to_markdown

    sim = PATHS["results"] / "simulation"
    f = sim / "summary.csv"
    if not f.exists():
        p.error(f"{f} not found — run `simulate_rules.py --run <run>` first")
    s = pd.read_csv(f)
    years = [int(y) for y in args.years]
    ns = {y: y * DAYS_PER_YEAR for y in years}
    missing = [y for y, n in ns.items() if n not in set(s.n)]
    if missing:
        p.error(f"no simulated samples of {missing} years in {f} (have "
                f"{sorted({n // DAYS_PER_YEAR for n in s.n})})")

    def pct(x):
        return "—" if pd.isna(x) else f"{100 * x:.1f}"

    rows, planted = [], {}
    for st in ORDER:
        sub = s[s.sim_template == st]
        if sub.empty:
            continue
        th = sorted(sub.theta.unique())
        planted[st] = (th[-1], sub.rho0.iloc[0])
        for k, tpl in enumerate(sorted(sub.template.unique(),
                                       key=lambda t: ("_bw" in t, t))):
            row = {"Phenomenon": PHENOMENON.get(st, st) if k == 0 else "",
                   "Rule": tpl.split("_")[0]}
            for kind, theta in (("FA", 0.0), ("Power", th[-1])):
                for y in years:
                    r = sub[(sub.template == tpl) & (sub.theta == theta) & (sub.n == ns[y])]
                    row[f"{kind} {y}y rules"] = pct(r.detection_rate.iloc[0]) if len(r) else "—"
                    row[f"{kind} {y}y econ."] = (pct(r.econ_detection_rate.iloc[0])
                                                 if len(r) and k == 0 else "")
            rows.append(row)
    out = pd.DataFrame(rows)

    planted_txt = "; ".join(f"{PHENOMENON[k].lower()} $\\theta={t:g}$, $\\rho_0={r:g}$"
                            for k, (t, r) in planted.items())
    reps = int(s.reps.max())
    yrs = " and ".join(str(y) for y in years)
    caption = (
        f"Share (\\%) of {reps} simulated samples where the rule is significant "
        f"(lift $\\geq 0.05$, CI excl.\\ zero) vs the econometric test. False alarms: "
        f"nothing planted; power: phenomenon planted ({planted_txt}). 70/30 fit/test; "
        f"unscorable = not detected.")

    out.to_csv(sim / "simulation_pub.csv", index=False)
    (sim / "simulation_pub.md").write_text(
        caption.replace("\\%", "%").replace("\\emph{FA}", "FA")
        .replace("\\emph{Power}", "Power").replace("$\\theta", "θ").replace("$\\rho_0", "ρ0")
        .replace("$\\geq", "≥").replace("$", "") + "\n\n" + to_markdown(out) + "\n")

    ncol = 2 * len(years)
    spec = "ll" + "rr" * ncol
    head1 = " & & " + " & ".join(
        [f"\\multicolumn{{{ncol * 1}}}{{c}}{{False alarms (\\%)}}",
         f"\\multicolumn{{{ncol * 1}}}{{c}}{{Power (\\%)}}"]) + " \\\\"
    c1 = 3
    rule1 = (f"\\cmidrule(lr){{{c1}-{c1 + ncol - 1}}}"
             f"\\cmidrule(lr){{{c1 + ncol}-{c1 + 2 * ncol - 1}}}")
    head2 = " & & " + " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{y} years}}" for _ in range(2) for y in years) + " \\\\"
    rule2 = "".join(f"\\cmidrule(lr){{{c1 + 2 * i}-{c1 + 2 * i + 1}}}"
                    for i in range(2 * len(years)))
    head3 = "Phenomenon & Rule & " + " & ".join(["rules & econ."] * (2 * len(years))) + " \\\\"
    lines = [_run_tag().rstrip("\n")] if _run_tag() else []
    # printed at 70% of the text width (chosen 2026-11-10)
    lines += ["\\begin{figure}[t]", "\\centering", "{\\setbox0=\\hbox{%",
              f"\\begin{{tabular}}{{{spec}}}", "\\toprule", head1, rule1, head2, rule2,
              head3, "\\midrule"]
    for r in out.itertuples(index=False):
        lines.append(" & ".join("--" if v == "—" else str(v) for v in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}%",
              "\\resizebox{0.7\\textwidth}{!}{\\box0}\\par}",   # fixed at 70% of the text width
              f"\\caption{{\\small{{{caption}}}}}",
              "\\label{fig:simulation}", "\\end{figure}"]
    (sim / "simulation_pub.tex").write_text("\n".join(lines) + "\n")

    if not args.quiet:
        print(to_markdown(out))
    print(f"{len(out)} rows -> {sim / 'simulation_pub'}.{{csv,md,tex}}")
    return out


if __name__ == "__main__":
    main()
