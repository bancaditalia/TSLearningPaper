#!/usr/bin/env python3
"""
R1.3 — verify mined rules on held-out data.

Fits each template on the TRAINING period, freezes the constants, then re-measures the
rule on the TEST period. Two engines:

    --engine python    scoring over the interval facts already on disk. No reasoner.
    --engine vadalog   compose preLabeling.vada + the frozen rule, run it, and compute
                       confidence from the engine's OWN derivations.
    --engine both      run both and report the difference — agreement is itself a result
                       (the Python labeler faithfully reproduces the declarative
                       semantics); disagreement localises a bug.

The composed program strips preLabeling's own @output annotations and keeps only the
predicates being verified, so a run writes 2-3 CSVs instead of 25. The EDB is a
CONTINUOUS timeline (not the disjoint blocks used for training), so `/\\50` slides and the
rule's temporal operators can reach across windows.

Examples
--------
    # no reasoner needed
    python scripts/verify_rules.py -p sp500_gold --engine python

    # write the .vada + EDB for inspection, don't execute
    python scripts/verify_rules.py -p sp500_gold --engine vadalog --dry-run

    # the real thing (reasoner must be running)
    python scripts/verify_rules.py -p sp500_gold --engine vadalog

    # cross-check the two
    python scripts/verify_rules.py -p sp500_gold,sp500_bitcoin --engine both

    # one template, verify on the training period instead (sanity check)
    python scripts/verify_rules.py -p sp500_gold -t T1_safeHaven --period train

Rules are verified with VARIABLES (X, Y) by default, like generate_rules.py; --ground
substitutes the series names. The rules are REFIT here (not read from rules.csv) with the
same --pct/--max-w/--max-gap as step 8, and checked against the step-8 table.

Artifacts land in data/processed/verification/ (`<tag>.vada`, `<tag>_edb.csv`, and the
derived predicate CSVs). The summary table goes to results/verification.csv for --source py,
results/verification_<source>.csv otherwise.
"""
from _cli import bootstrap, csv_list, parser, stage

bootstrap()


def _fitted_constants(path) -> "dict | None":
    """{(template, pair): {constant: value}} from a generate_rules.py table, or None."""
    import pandas as pd
    if not path.exists():
        return None
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c.startswith("const_")]
    out = {}
    for _, r in df[df["accepted"]].iterrows():
        out[(r["template"], r["pair"])] = {c[6:]: r[c] for c in cols if pd.notna(r[c])}
    return out


def _check_constants(rules, fitted, quiet: bool) -> None:
    """Warn when a refit rule's constants differ from the ones in the rules table: then the
    rule being verified is not the rule the tables report (different step-8 settings,
    or labels changed since step 8 ran)."""
    if fitted is None:
        return
    for r in rules:
        want = fitted.get((r.template, r.pair))
        if not r.accepted:
            if want is not None:
                print(f"  WARNING {r.template} {r.pair}: fitted in the rules table "
                      f"{want} but not here ({r.reason}) — pass the same "
                      f"--pct/--max-w/--max-gap as step 8, or rerun step 8")
            continue
        got = {k: v for k, v in r.constants.items()}
        if want is None:
            print(f"  WARNING {r.template} {r.pair}: fitted here but not in the rules "
                  f"table — rerun step 8")
        elif any(float(want.get(k, float("nan"))) != float(v) for k, v in got.items()):
            print(f"  WARNING {r.template} {r.pair}: constants {got} differ from the rules "
                  f"table {want} — pass the same --pct/--max-w/--max-gap as step 8, "
                  f"or rerun step 8")


def main(argv=None):
    p = parser(__doc__.split("Examples")[0].strip(), epilog=__doc__)
    p.add_argument("-p", "--pairs", type=csv_list, default=["all"],
                   help="pairs to verify, or 'all' (default: all)")
    p.add_argument("-t", "--templates", type=csv_list, default=None,
                   help="templates to verify (default: all registered)")
    p.add_argument("--source", default="py", choices=["py", "reasoner", "classifier"],
                   help="which labelling the rules are fitted from (default: py)")
    p.add_argument("--engine", default=None, choices=["python", "vadalog", "both"],
                   help="how to verify (default: python, or vadalog if --dry-run)")
    p.add_argument("--period", default="test", choices=["test", "train"],
                   help="period to verify on (default: test)")
    p.add_argument("--dry-run", action="store_true",
                   help="write the .vada program and EDB without executing them. "
                        "Implies --engine vadalog, since there is nothing to generate "
                        "for the python engine")
    p.add_argument("--ground", action="store_true",
                   help="verify with the actual series names substituted instead of the "
                        "variables X,Y. Off by default since 2026-08-05 — rules carry "
                        "variables. NOTE the engine binds X,Y to whatever pairs the fact "
                        "base contains, so a per-pair EDB gives the same answer either "
                        "way; a multi-pair EDB would additionally show which pairs "
                        "satisfy the rule, which is support rather than verification")
    p.add_argument("--keep-body-outputs", action="store_true",
                   help="also emit the rule's body atoms as @output. Off by default: the "
                        "engine derives them internally to fire the rule, but they aren't "
                        "needed downstream — the confidence denominator uses the exact "
                        "Python-side condition. Turn on to cross-check the derivations")
    # The same fitting settings as generate_rules.py: verification REFITS the rules (it
    # doesn't read rules.csv), so non-default step-8 settings must be passed here too.
    p.add_argument("--pct", type=int, default=25,
                   help="percentile for duration templates T5/T6 (default: 25) — use "
                        "the value given to generate_rules.py")
    p.add_argument("--max-w", type=int, default=None,
                   help="upper end of the w grid (default: config rules.max_w, 25) — as in generate_rules.py")
    p.add_argument("--max-gap", type=float, default=float("inf"),
                   help="T6: largest gap between the two regimes (default: unlimited) — "
                        "as in generate_rules.py")
    p.add_argument("--out", default=None,
                   help="summary table (default: results/verification.csv for --source "
                        "py, results/verification_<source>.csv otherwise)")
    p.add_argument("--tag", default=None,
                   help="suffix for the output file: verification_<tag>.csv, matching "
                        "generate_rules.py --tag and rule_table.py --tag. Defaults to "
                        "the source for any source other than py; a tag naming a "
                        "different source is refused. Ignored if --out is given")
    args = p.parse_args(argv)

    # --tag names the output file, --source picks the rules. A tag that is itself a
    # source name but differs from --source is always a mistake: `--tag classifier`
    # alone used to verify the Python-labeller rules and file them as the classifier's.
    _sources = ("py", "reasoner", "classifier")
    if args.tag in _sources and args.tag != args.source:
        p.error(f"--tag {args.tag} with --source {args.source}: the file would be named "
                f"after one label source and hold another's rules. Use --source {args.tag}")

    import pandas as pd
    from pathlib import Path
    from tslearning.config import PATHS
    from tslearning.rules import generate as G, verify as V
    from tslearning.rules.templates import TEMPLATES

    # --dry-run exists to produce the .vada; the python engine produces none, so asking
    # for both together would silently do nothing (which it used to).
    engine = args.engine
    if engine is None:
        engine = "vadalog" if args.dry_run else "python"
    elif engine == "python" and args.dry_run:
        print("note: --dry-run has no effect with --engine python (no program to "
              "generate) — switching to --engine vadalog")
        engine = "vadalog"

    pairs = G.discover_pairs(args.source) if args.pairs == ["all"] else args.pairs
    if not pairs:
        p.error(f"no labelled pairs under --source {args.source}")
    if args.templates:
        unknown = [t for t in args.templates if t not in TEMPLATES]
        if unknown:
            p.error(f"unknown template(s) {unknown}")

    # The rules table step 8 wrote for this source, to check the refit reproduces it
    rules_name = (f"rules_{args.tag}.csv" if args.tag else
                  "rules.csv" if args.source == "py" else f"rules_{args.source}.csv")
    fitted = _fitted_constants(PATHS["results"] / rules_name)
    if fitted is None and not args.quiet:
        print(f"note: {rules_name} not found — the refit rules can't be checked against it")

    # one folder per run (and per rule inside it), so a rerun never overwrites the exact
    # program and facts an earlier verification was based on
    from datetime import datetime
    run_dir = None
    if engine in ("vadalog", "both"):
        run_dir = PATHS["data_processed"] / "verification" / datetime.now().strftime("%Y%m%d_%H%M%S")

    from tslearning.rules.fitting import MIN_W, W_GRID   # config rules.min_w / max_w
    w_grid = W_GRID if args.max_w is None else range(MIN_W, args.max_w + 1)
    rows = []
    with stage(f"verify · {engine} · {args.period} · {','.join(pairs)}", args.quiet):
        for pair in pairs:
            if not args.quiet:
                print(f"\n{pair}")
            # score_pair fits on train and freezes — exactly what verification needs
            rules = G.score_pair(pair, args.source, args.templates, split=True,
                                 verbose=False, pct=args.pct,
                                 w_grid=w_grid, max_gap=args.max_gap)
            _check_constants(rules, fitted, args.quiet)
            for rule in rules:
                if not rule.accepted:
                    if not args.quiet:
                        print(f"  {rule.template:22s} not fitted — {rule.reason}")
                    continue
                if not args.quiet:
                    print(f"  {rule.template}  {rule.render(ground=args.ground)}")
                res = V.verify(rule, pair=pair, engine=engine, period=args.period,
                               ground=args.ground, run=not args.dry_run,
                               keep_body_outputs=args.keep_body_outputs, run_dir=run_dir,
                               verbose=not args.quiet)
                row = {"template": rule.template, "pair": pair, "period": args.period,
                       "rule": res["rule"],
                       "train_confidence": (rule.scores.get("train") or {}).get("confidence"),
                       "scored_confidence": (rule.scores.get(args.period) or {}).get("confidence")}
                for eng in ("python", "vadalog"):
                    if eng in res:
                        row[f"{eng}_confidence"] = res[eng].get("confidence")
                        row[f"{eng}_condition_days"] = res[eng].get("condition_days")
                        row[f"{eng}_reason"] = res[eng].get("reason")
                if "agreement" in res:
                    row["agreement_abs_diff"] = res["agreement"]
                if run_dir is not None:
                    row["artifacts"] = str((run_dir / res["vadalog"]["tag"]).relative_to(
                        PATHS["data_processed"])) if "vadalog" in res else None
                rows.append(row)

    table = pd.DataFrame(rows)
    if args.out:
        dest = Path(args.out)
    elif args.tag:
        dest = PATHS["results"] / f"verification_{args.tag}.csv"
    elif args.source != "py":
        # same rule as generate_rules.py: only the exact labeller writes verification.csv
        dest = PATHS["results"] / f"verification_{args.source}.csv"
    else:
        dest = PATHS["results"] / "verification.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(dest, index=False)
    print(f"\n{len(table)} rules verified ({engine}, {args.period}) -> {dest}")

    col = f"{'python' if engine != 'vadalog' else 'vadalog'}_confidence"
    if col in table.columns and table[col].notna().any():
        print(f"\ntrain -> {args.period} (frozen constants):")
        for _, r in table[table[col].notna()].sort_values(col, ascending=False).iterrows():
            tr = r.get("train_confidence")
            print(f"  {r['template']:22s} {r['pair']:28s} "
                  f"{'n/a' if pd.isna(tr) else f'{tr:.2f}'} -> {r[col]:.2f}")
    return table


if __name__ == "__main__":
    main()
