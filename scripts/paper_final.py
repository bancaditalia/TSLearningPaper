#!/usr/bin/env python3
"""
Build the paper's final tables for one run and pair list, and copy them into one folder
you can copy from without touching anything else.

    python scripts/paper_final.py            # that's all: run and pairs from the config

The run and the pairs come from `paper_final:` in configs/config.yaml; `--run` and `-p`
override them. Recomputes nothing: first `paper.py tables` for the run (skip with
--no-run-tables), then the tables restricted to the pairs:
  1. results/runs/<run>/paper/ : rule tables (exact and classifier labels), the combined
     table, the Bonferroni check, fidelity and speed tables, for these pairs only.
  2. results/runs/<run>/simulation/simulation_pub.* : the simulation table.
  3. paper_tables/ (repo root, or -o): a COPY of the .tex files the paper uses, plus
     SOURCES.md saying where each came from and when. Only these files are replaced.

Rerun it whenever the run's results change (new engine run, LLM runs, timing).
"""
import os
import sys
from pathlib import Path

# No --run on the command line and no TSL_RUN: use the paper's run from the config, so
# the bare command always builds the paper's tables (never the old base run).
if not any(a == "--run" or a.startswith("--run=") for a in sys.argv[1:]) \
        and not os.environ.get("TSL_RUN"):
    import yaml
    _cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" /
                           "config.yaml").read_text())
    _run = (_cfg.get("paper_final") or {}).get("run")
    if not _run:
        sys.exit("no --run given and no paper_final.run in configs/config.yaml")
    sys.argv += ["--run", _run]

from _cli import bootstrap, csv_list, parser  # noqa: E402

bootstrap()

FINAL = [  # (file in the build folder, what it is)
    ("rule_results_combined_pub.tex", "rule results, exact and classifier labels side by side"),
    ("rule_results_pub.tex", "rule results, exact labels"),
    ("rule_results_classifier_pub.tex", "rule results, classifier labels"),
    ("rule_library_box_pub.tex", "rule library (templates)"),
    ("rule_comparison_pub.tex", "template rules vs LLM-proposed rules (S&P 500 – gold)"),
    ("llm_rules_box_pub.tex", "the LLM rules shown in the comparison, per model"),
    ("distillation.tex", "classifier fidelity vs the reasoner"),
    ("speedup.tex", "reasoner vs classifier time"),
    ("speedup_pub.tex", "reasoner vs classifier time: clean timing run + all-runs average"),
    ("classifier_pub.tex", "classifier fidelity (a) and speed (b), side by side"),
    ("multiple_testing.md", "Bonferroni check, exact labels (numbers for the text)"),
    ("multiple_testing_classifier.md", "Bonferroni check, classifier labels"),
    ("verification_summary_filtered.md", "rule verification, Python vs Vadalog, reported pairs only"),
    ("agreement_summary_filtered.md", "label agreement, Python vs reasoner, reported pairs only"),
]


def main(argv=None):
    p = parser(__doc__.strip().split("\n")[0], epilog=__doc__)
    p.add_argument("-p", "--pairs", type=csv_list, default=None,
                   help="the pairs the paper reports (default: paper_final.pairs in "
                        "configs/config.yaml)")
    p.add_argument("--no-run-tables", action="store_true",
                   help="don't run `paper.py tables` for the run first")
    p.add_argument("-o", "--out", default="paper_tables",
                   help="folder for the copies, relative to the repo root (default "
                        "paper_tables/)")
    p.add_argument("--years", default="10,40",
                   help="sample lengths of the simulation table (default 10,40)")
    args = p.parse_args(argv)

    import datetime as dt

    import pandas as pd
    import shutil
    import subprocess

    from tslearning.config import CFG, PATHS, REPO_ROOT
    from tslearning.eval import report
    from tslearning.rules.tables import pair_label, to_latex, to_markdown

    run = CFG["run"]["name"]
    if args.pairs is None:
        args.pairs = list((CFG.get("paper_final") or {}).get("pairs") or [])
        if not args.pairs:
            p.error("no -p given and no paper_final.pairs in configs/config.yaml")
    print(f"paper tables · run {run} · pairs {', '.join(args.pairs)}")
    res = PATHS["results"]
    build = res / "paper"
    build.mkdir(parents=True, exist_ok=True)
    P = ",".join(args.pairs)
    here = Path(__file__).resolve().parent

    def step(script, *a):
        cmd = [sys.executable, str(here / script), *a, "-q", "--run", run]
        print("  $", " ".join(cmd[1:]))
        r = subprocess.run(cmd)
        if r.returncode:
            sys.exit(f"stopped: {script} failed (exit {r.returncode})")

    # 0. the run's own tables (all its pairs), so the inputs below are current
    if not args.no_run_tables:
        cmd = [sys.executable, str(here / "paper.py"), "tables", "--run", run]
        print("  $", " ".join(cmd[1:]))
        if subprocess.run(cmd).returncode:
            print("  WARNING: paper.py tables reported an ERROR above (consistency "
                  "checks) — read it before using the tables")

    # 1. rule tables, Bonferroni, combined table — restricted to the pairs
    step("rule_table.py", "-p", P, "-o", str(build), "--in-dir", str(res))
    step("rule_table.py", "-p", P, "--tag", "classifier", "-o", str(build), "--in-dir", str(res))
    step("multiple_testing.py", "-p", P, "-o", str(build))
    step("multiple_testing.py", "--source", "classifier", "-p", P, "-o", str(build))
    step("combined_rule_table.py", "--dir", str(build))

    # the LLM rule box is named after the runs it pools (llm_rules_box_pub_<run ids>.tex),
    # so older boxes stay in the folder; copy the one just written under a fixed name
    boxes = sorted(build.glob("llm_rules_box_pub_*.tex"), key=lambda f: f.stat().st_mtime)
    if boxes:
        shutil.copy2(boxes[-1], build / "llm_rules_box_pub.tex")

    # fidelity and speed tables, rows of the dropped pairs removed
    keep_labels = {pair_label(q) for q in args.pairs}
    dropped = [q for q in (CFG.get("paper") or {}).get("pairs", []) if q not in args.pairs]
    df, cap = report.distillation_table(res)
    if not df.empty:
        df = df[df.Pair.isin(keep_labels)]
        (build / "distillation.tex").write_text(to_latex(df, cap, "fig:distillation") + "\n")
        (build / "distillation.md").write_text(to_markdown(df) + "\n")
    df, cap = report.speedup_table(res)
    if not df.empty:
        df = df[~df.Run.apply(lambda r: any(q in r for q in dropped))]
        (build / "speedup.tex").write_text(to_latex(df, cap, "fig:speedup") + "\n")
        (build / "speedup.md").write_text(to_markdown(df) + "\n")
    # the two-row version: the clean timing run and the all-runs average
    df, cap = report.speedup_pub_table(res)
    if not df.empty:
        (build / "speedup_pub.tex").write_text(
            report.unscaled(to_latex(df, cap, "fig:speedup-pub")) + "\n")
        (build / "speedup_pub.md").write_text(to_markdown(df) + "\n")
    # fidelity (a) and speed (b) side by side in one figure
    dist, _ = report.distillation_table(res)
    spd, _ = report.speedup_pub_table(res)
    if not dist.empty and not spd.empty:
        dist = dist[dist.Pair.isin(keep_labels)]
        rt = pd.read_csv(res / "reasoner_timing.csv") \
            if (res / "reasoner_timing.csv").exists() else None
        w = str(rt["classifier_weights"].iloc[0]) \
            if rt is not None and "classifier_weights" in rt.columns else "?"
        (build / "classifier_pub.tex").write_text(
            report.classifier_side_by_side(dist, spd, w) + "\n")

    # verification counts for the text, restricted to the reported pairs
    from tslearning.eval import verification_summary
    # written into the run's own folder (next to the unfiltered files), then copied
    verification_summary.run(args.pairs, res, res, verbose=False)
    shutil.copy2(res / "verification_summary_filtered.md", build / "verification_summary_filtered.md")
    if verification_summary.agreement(args.pairs, res, res):
        shutil.copy2(res / "agreement_summary_filtered.md", build / "agreement_summary_filtered.md")

    # 2. simulation table (independent of the pairs)
    if (res / "simulation" / "summary.csv").exists():
        step("simulation_table.py", "--years", args.years)

    # 3. copies
    out = Path(args.out)
    out = out if out.is_absolute() else REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    files = [(build / f, what) for f, what in FINAL] + [
        (res / "simulation" / "simulation_pub.tex", "simulation study (false alarms, power)")]
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# Paper tables — copied {now}", "",
             f"Run `{run}`, pairs: {', '.join(args.pairs)}.",
             "Rebuild with:", "",
             f"    python scripts/paper_final.py --run {run} -p {P}", "",
             "| file | what | copied from |", "|---|---|---|"]
    for src, what in files:
        if not src.exists():
            lines.append(f"| {src.name} | {what} | MISSING — not produced for this run |")
            continue
        shutil.copy2(src, out / src.name)
        lines.append(f"| {src.name} | {what} | `{src.relative_to(REPO_ROOT)}` |")
    (out / "SOURCES.md").write_text("\n".join(lines) + "\n")
    print(f"\npaper tables -> {out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out}/"
          f"  (see SOURCES.md)")


if __name__ == "__main__":
    main()
