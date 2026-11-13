"""
Robustness table: the same rules scored in two (or more) runs, side by side.

    python scripts/compare_runs.py --runs main_minw0,robust_2016-2024
    python scripts/compare_runs.py --runs base,<run> --source classifier
    python scripts/compare_runs.py --runs <a>,<b> --all      # also rules scored in only one run

Reads each run's results/runs/<run>/rule_results[_classifier].csv (results/ for `base`);
recomputes nothing. One row per (rule, pair) scored on the test period in EVERY run
(--all: in at least one), with each run's lift [CI] and verdict, and whether the
lift has the same sign in all runs. The caption is built from the runs' own
parameters (train/test periods, thresholds), so it cannot drift from what was run.

Writes results/compare/robustness[_classifier].{csv,md,tex} and a compare.json
naming the runs; -o for another folder.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _cli import bootstrap, csv_list, parser  # noqa: E402

bootstrap()


def _years(span) -> str:
    a, b = str(span[0])[:4], str(span[1])[:4]
    return a if a == b else f"{a}–{b}"


def main(argv=None):
    p = parser(__doc__.strip().split("\n")[0], epilog=__doc__)
    p.add_argument("--runs", type=csv_list, required=True,
                   help="two or more run names, comma separated ('base' = the paper run "
                        "in results/); the first is the reference")
    p.add_argument("--source", default="py", choices=["py", "classifier"],
                   help="which label source's rule table to compare (default: py)")
    p.add_argument("--all", action="store_true",
                   help="include rules scored in only some of the runs")
    p.add_argument("-o", "--out-dir", default=None,
                   help="default: results/compare/ (of the base run)")
    args = p.parse_args(argv)
    if len(args.runs) < 2:
        p.error("give at least two runs")

    import json
    import pandas as pd
    from tslearning.config import BASE_RUN, REPO_ROOT, load_config
    from tslearning.rules.tables import (_template_sort_key, to_latex, to_markdown)

    suffix = "" if args.source == "py" else "_classifier"
    runs, frames = [], []
    for name in args.runs:
        c = load_config(run=name)
        f = c["paths"]["results"] / f"rule_results{suffix}.csv"
        if not f.exists():
            p.error(f"{f} not found — run `paper.py tables --run {name}` first")
        thr = (c.get("labels") or {}).get("thresholds", "train")
        win = c["windows"]["financial"]["length"]
        min_w = int((c.get("rules") or {}).get("min_w", 0))
        # numbered, and naming window length and min_w: runs with the same periods
        # (e.g. window 25 / 50 / 100) must not share a column name
        col = (f"({len(runs) + 1}) test {_years(c['split']['test'])}, "
               f"train {_years(c['split']['train'])}, {thr} thr., "
               f"window {win}, min_w {min_w}")
        runs.append({"name": name, "file": str(f.relative_to(REPO_ROOT)),
                     "train": c["split"]["train"], "test": c["split"]["test"],
                     "thresholds": thr, "window": win, "min_w": min_w, "column": col})
        d = pd.read_csv(f)
        lift = pd.to_numeric(d["Lift"], errors="coerce")
        d = d.assign(_lift=lift, _scored=lift.notna())
        frames.append(d[["Rule", "Pair", "_lift", "CI", "Verdict", "_scored"]])

    m = frames[0]
    for i, d in enumerate(frames[1:], start=1):
        m = m.merge(d, on=["Rule", "Pair"], how="outer", suffixes=("", f"_{i}"))
    m = m.rename(columns={"_lift": "_lift_0", "CI": "CI_0", "Verdict": "Verdict_0",
                          "_scored": "_scored_0"})
    scored = m[[f"_scored_{i}" for i in range(len(runs))]].fillna(False).astype(bool)
    keep = scored.any(axis=1) if args.all else scored.all(axis=1)
    m = m[keep].copy()

    def cell(i, r):
        lift = r[f"_lift_{i}"]
        if pd.isna(lift):
            v = r.get(f"Verdict_{i}")
            return "—" if pd.isna(v) else str(v).split(":")[0].split(" (")[0]
        ci = r[f"CI_{i}"]
        return f"{lift:+.3f} {ci}" if isinstance(ci, str) and ci != "—" else f"{lift:+.3f}"

    rows = []
    for _, r in m.iterrows():
        row = {"Rule": r["Rule"], "Pair": r["Pair"]}
        for i, run in enumerate(runs):
            row[run["column"]] = cell(i, r)
        for i, run in enumerate(runs):
            v = r.get(f"Verdict_{i}")
            row[f"verdict {i + 1}"] = "—" if pd.isna(v) else str(v).split(":")[0].split(" (")[0]
        lifts = [r[f"_lift_{i}"] for i in range(len(runs))]
        signs = {(l > 0) - (l < 0) for l in lifts if pd.notna(l)}
        row["same sign"] = ("—" if len([l for l in lifts if pd.notna(l)]) < 2
                            else "yes" if len(signs) == 1 else "no")
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Rule", "Pair"], key=lambda s: s.map(
            _template_sort_key) if s.name == "Rule" else s, kind="stable")

    desc = "; ".join(f"({i + 1}) train {_years(r['train'])}, test {_years(r['test'])}, "
                     f"tau/qLow {'rolling (past 2 years)' if r['thresholds'] == 'rolling' else 'fitted on train'}, "
                     f"{r['window']}-day windows, min. w {r['min_w']}"
                     for i, r in enumerate(runs))
    caption = (f"The rule results in {len(runs)} settings ({args.source} labels). "
               f"Test-period lift with its 95\\% bootstrap CI in each setting: {desc}. "
               f"Constants are refitted in each setting on its own training period. "
               + ("Rules scored in only some settings are included; the others show their "
                  "verdict. " if args.all else
                  "Only rules scored on the test period in every setting are shown. ")
               + "Pairs absent from a setting (e.g.\\ Bitcoin before 2017) do not appear.")

    base = load_config(run=BASE_RUN)
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir = out_dir if out_dir.is_absolute() else REPO_ROOT / out_dir
    else:
        out_dir = base["paths"]["results"] / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"robustness{suffix}"
    out.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.md").write_text(caption.replace("\\%", "%").replace("e.g.\\ ", "e.g. ")
                                        + "\n\n" + to_markdown(out) + "\n")
    (out_dir / f"{stem}.tex").write_text(
        to_latex(out, caption, f"fig:robustness{suffix}") + "\n")
    (out_dir / "compare.json").write_text(json.dumps(
        {"source": args.source, "all": args.all, "runs": runs}, indent=2, default=str))
    if not args.quiet:
        print(to_markdown(out))
    dest = out_dir / stem
    dest = dest.relative_to(REPO_ROOT) if dest.is_relative_to(REPO_ROOT) else dest
    print(f"{len(out)} rule(s) compared across {len(runs)} runs -> {dest}.{{csv,md,tex}}")
    return out


if __name__ == "__main__":
    main()
