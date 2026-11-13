"""
List and create RUNS — robustness variants of the paper's setup (other periods, window
lengths, strides), each with its own folders and a record of its parameters.

    python scripts/runs.py                       # every run: periods, window, stride, folders
    python scripts/runs.py show <name>           # one run's full parameters + its YAML
    python scripts/runs.py new --train 2005-01-01:2014-12-31 --test 2015-01-01:2021-12-31
    python scripts/runs.py new --window 75 --trend 0.0245 --max-w 38 --min-days 45
    python scripts/runs.py new --stride 5 --name my_run

`new` writes configs/runs/<name>.yaml with ONLY the values that differ from the base
config. Without --name the name is built from the parameters, so it says what the run is:
    tr2005-2014_te2015-2021_w50_s1_gold-nasdaq      (+ the series paired with the base)

Then run any script with `--run <name>` (or TSL_RUN=<name>): its inputs and outputs live
in data/processed/runs/<name>/ and results/runs/<name>/, each with a run.json, and the
base run's folders are never touched. See configs/runs/README.md for the order of steps.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _cli import bootstrap, csv_list, parser  # noqa: E402

bootstrap()


def _span(s: str) -> list:
    a, b = s.split(":")
    return [a, b]


def _auto_name(cfg: dict) -> str:
    tr, te = cfg["split"]["train"], cfg["split"]["test"]
    w = cfg["windows"]["financial"]["length"]
    st = (cfg.get("real_pairs") or {}).get("stride", 1)
    name = f"tr{tr[0][:4]}-{tr[1][:4]}_te{te[0][:4]}-{te[1][:4]}_w{w}_s{st}"
    if (cfg.get("labels") or {}).get("thresholds", "train") == "rolling":
        name += "_roll"
    # the series paired with the base, so two runs differing only in pairs differ in name
    from tslearning.datasets.pairs import describe
    others = [describe(p)["b"].lower() for p in (cfg.get("paper") or {}).get("pairs") or []]
    return name + ("_" + "-".join(others) if others else "")


def _check_coverage(cfg: dict) -> None:
    """Warn if the run's period starts or ends outside the data on disk (the financial
    download is fixed at 2005-2024 in ingest_financial.py; Bitcoin starts in 2017)."""
    import pickle
    import pandas as pd
    from tslearning.datasets.pairs import describe
    f = cfg["paths"]["base_processed"] / "transformed.pkl"
    if not f.exists():
        return
    series = pickle.loads(f.read_bytes())
    lo, hi = pd.Timestamp(cfg["split"]["train"][0]), pd.Timestamp(cfg["split"]["test"][1])
    for pair in (cfg.get("paper") or {}).get("pairs") or []:
        for side in ("a", "b"):
            name = describe(pair)[side]
            s = series.get(name)
            if s is None:
                continue
            s0, s1 = s.dropna().index.min(), s.dropna().index.max()
            slack = pd.Timedelta(days=10)          # holidays at the edges
            tr_end = pd.Timestamp(cfg["split"]["train"][1])
            if s0 > tr_end or s1 < hi - slack:
                print(f"WARNING: {name} covers {s0.date()}..{s1.date()}, the run needs "
                      f"data in {lo.date()}..{hi.date()} (pair {pair}). Drop the pair "
                      f"with --pairs, or change the period.")
            elif s0 > lo + slack:
                print(f"note: {name} starts {s0.date()}, so for {pair} the train period "
                      f"is effectively {s0.date()}..{tr_end.date()}.")


def cmd_list() -> None:
    from tslearning.runs import list_runs
    for r in list_runs():
        if "error" in r:
            print(f"{r['name']:<46} ERROR: {r['error']}")
            continue
        p = r["parameters"]
        tr, te = p["split"]["train"], p["split"]["test"]
        print(f"{r['name']:<46} train {tr[0]}..{tr[1]}  test {te[0]}..{te[1]}  "
              f"window {p['window']['length']:>3}  stride {p['real_pairs']['stride']}  "
              f"{'outputs' if r['has_outputs'] else 'no outputs yet'}")
        from tslearning.config import REPO_ROOT
        rel = lambda q: Path(q).relative_to(REPO_ROOT)
        print(f"{'':<46} data {rel(r['data'])}   results {rel(r['results'])}")


def cmd_show(name: str) -> None:
    import json
    from tslearning.config import load_config, run_file
    from tslearning.runs import fingerprint, parameters
    c = load_config(run=name)
    print(json.dumps({"run": name, "fingerprint": fingerprint(parameters(c)),
                      "parameters": parameters(c),
                      "data": str(c["paths"]["data_processed"]),
                      "results": str(c["paths"]["results"])}, indent=2, default=str))
    f = run_file(name)
    if f.exists():
        print(f"\n# {f}\n{f.read_text()}")


def cmd_new(a) -> None:
    import yaml
    from tslearning.config import BASE_RUN, RUNS_DIR, _deep_merge, load_config, run_file

    over = {}
    if a.train:
        over.setdefault("split", {})["train"] = _span(a.train)
    if a.test:
        over.setdefault("split", {})["test"] = _span(a.test)
    if a.train or a.test:
        base = load_config(run=BASE_RUN)
        tr = over.get("split", {}).get("train", base["split"]["train"])
        te = over.get("split", {}).get("test", base["split"]["test"])
        # the real pairs are labelled over train + test: keep them in step
        over["real_pairs"] = {"date_from": tr[0], "date_to": te[1]}
    if a.window:
        over["windows"] = {"financial": {"length": a.window}}
    if a.stride:
        over.setdefault("real_pairs", {})["stride"] = a.stride
    tail = a.tail_min if a.tail_min in (None, "auto") else int(a.tail_min)
    lab = {k: v for k, v in (("trend", a.trend), ("high_vol_pct", a.high_vol),
                             ("low_vol_pct", a.low_vol), ("tail_min", tail),
                             ("thresholds", a.thresholds))
           if v is not None}
    if lab:
        over["labels"] = lab
    rules = {k: v for k, v in (("max_w", a.max_w), ("min_w", a.min_w),
                               ("min_condition_days", a.min_days))
             if v is not None}
    if rules:
        over["rules"] = rules
    if a.pairs:
        over["paper"] = {"pairs": a.pairs}
        base_fid = (load_config(run=BASE_RUN).get("paper") or {}).get("fidelity") or {}
        dropped = {p: None for p in base_fid if p not in a.pairs}   # null = remove
        if dropped:
            over["paper"]["fidelity"] = dropped
    if not over:
        sys.exit("nothing differs from the base run — pass at least one option")

    merged = _deep_merge(load_config(run=BASE_RUN), over)
    _check_coverage(merged)
    name = a.name or _auto_name(merged)
    f = run_file(name)
    if f.exists() and not a.force:
        sys.exit(f"{f} already exists (--force to replace it; a run with outputs refuses "
                 f"a change of parameters anyway)")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    f.write_text(f"# run '{name}' — created by scripts/runs.py; overrides configs/config.yaml\n"
                 + yaml.safe_dump(over, sort_keys=False))
    print(f"wrote {f}\n\n{f.read_text()}\nuse it with:  --run {name}")
    if a.window and not (lab or rules):
        print("NOTE: the window changed but the labelling cut-offs (trend, tail_min) and "
              "the rule limits (max_w, min_condition_days) did not — they were chosen for "
              "50 days. See configs/runs/README.md before running.")


def main(argv=None):
    p = parser(__doc__.strip().split("\n")[0], epilog=__doc__)
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("list")
    s = sub.add_parser("show")
    s.add_argument("name")
    n = sub.add_parser("new")
    n.add_argument("--name", default=None, help="default: built from the parameters")
    n.add_argument("--train", default=None, help="START:END, e.g. 2005-01-01:2014-12-31")
    n.add_argument("--test", default=None, help="START:END")
    n.add_argument("--window", type=int, default=None, help="window length (days)")
    n.add_argument("--stride", type=int, default=None, help="real-pair labelling stride")
    n.add_argument("--trend", type=float, default=None)
    n.add_argument("--high-vol", type=float, default=None)
    n.add_argument("--low-vol", type=float, default=None)
    n.add_argument("--tail-min", default=None,
                   help="co-crash days for tailDependent: a number, or auto (derived from "
                        "the window, see config labels.tail_min)")
    n.add_argument("--thresholds", default=None, choices=["train", "rolling"],
                   help="tau/qLow: fitted on the train period (paper) or rolling over "
                        "the 2 years before each window (labels.thresholds)")
    n.add_argument("--max-w", type=int, default=None)
    n.add_argument("--min-w", type=int, default=None, help="rules.min_w (config: 1)")
    n.add_argument("--min-days", type=int, default=None, help="rules.min_condition_days")
    n.add_argument("--pairs", type=csv_list, default=None,
                   help="paper.pairs for this run, e.g. sp500_gold,sp500_nasdaq")
    n.add_argument("--force", action="store_true")
    ac = sub.add_parser("accept-rules",
                        help="let an existing run take a changed rules.* setting from the "
                             "config (e.g. rules.min_w); then rerun its rule steps")
    ac.add_argument("name")
    a = p.parse_args(argv)
    if a.cmd in (None, "list"):
        return cmd_list()
    if a.cmd == "show":
        return cmd_show(a.name)
    if a.cmd == "accept-rules":
        from tslearning.runs import accept_rules_change
        ch = accept_rules_change(a.name)
        print(f"run '{a.name}': " + ("accepted " + "; ".join(ch) if ch else "nothing to accept")
              + ("\nits rule outputs are stale until you rerun: "
                 f"python scripts/paper.py results --no-engine --run {a.name}" if ch else ""))
        return ch
    return cmd_new(a)


if __name__ == "__main__":
    main()
