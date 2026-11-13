#!/usr/bin/env python3
"""
Stage 8 — rule generation. Fit the template constants and emit DatalogMTL rules.



Label source — the templates don't care which produced the facts, the CSVs are identical:

    --source py          real_pairs_regime_py/<pair>/     exact Python labeler (default)
    --source reasoner    real_pairs_regime_reasoner/<pair>/  Vadalog, recalibrated
    --source classifier  real_pairs_regime_classifier/<pair>/  the distilled classifier
                                                            (regimes predicted, flags copied)

Pairs: name them, or use `all` for every pair labelled under that source.

Grounding: rules are rendered with VARIABLES (X, Y) by default since 2026-08-05,
reversing the 2026-07-31 grounding decision.

The reasoning: syntax and evidence are different things. A rule is a logical object and
carries variables; WHICH ground pairs satisfy it is an empirical question answered by the
per-pair scores in the results table, not by baking one pair into the rule text. That also
turns the gold / bitcoin / nasdaq contrast into support and non-support for ONE rule,
rather than three unrelated single-pair claims.

The original worry — "fake generality", asserting something about pairs never examined —
is answered by reporting every pair the frozen rule was evaluated on, including the ones
where it fails. `--ground` restores the old behaviour.

Examples
--------
    python scripts/generate_rules.py --source py -p all
    python scripts/generate_rules.py --source py -p sp500_gold,sp500_bitcoin --score
    python scripts/generate_rules.py --source py -p all -t T1_safeHaven,T5_hedgeHorizon
    python scripts/generate_rules.py --source classifier -p sp500_bitcoin
    python scripts/generate_rules.py --list-templates
    python scripts/generate_rules.py --list-pairs --source py

    python scripts/generate_rules.py --source py -p all --score --ci     # the paper's run

Writes results/rules.csv for --source py, results/rules_<source>.csv otherwise (one row
per template x pair, including rejected ones — "the rule does not hold for this pair" is a
result, and the contrast between pairs that pass and fail is itself a finding).

With --score: constants fitted on the train period, frozen, and scored on train and test
(confidence, baseline, lift; --ci adds the bootstrap interval on the test lift).
Verification against the engine is the next step: verify_rules.py.
"""
from _cli import bootstrap, csv_list, parser, stage

bootstrap()


def main(argv=None):
    p = parser(__doc__.split("Examples")[0].strip(), epilog=__doc__)
    p.add_argument("--source", default="py", choices=["py", "reasoner", "classifier"],
                   help="which labelling produced the facts (default: py)")
    p.add_argument("-p", "--pairs", type=csv_list, default=["all"],
                   help="pairs to mine, comma separated, or 'all' (default: all)")
    p.add_argument("-t", "--templates", type=csv_list, default=None,
                   help="templates to run (default: all registered)")
    p.add_argument("--ground", action="store_true",
                   help="render with the actual series names instead of variables X,Y. "
                        "OFF by default since 2026-08-05: a rule is a logical object and "
                        "carries variables; which ground pairs satisfy it is reported by "
                        "the per-pair scores, not by baking the pair into the syntax")
    p.add_argument("--ci", action="store_true",
                   help="bootstrap a 95%% confidence interval on the test LIFT. Resamples\n"
                        "condition EPISODES where there are enough of them, otherwise\n"
                        "50-day BLOCKS within them — and reports which was used, because\n"
                        "the test period often holds a single episode and an interval\n"
                        "built from blocks of one event is weaker evidence than it looks")
    p.add_argument("--pct", type=int, default=25,
                   help="percentile for duration templates T5/T6 (default: 25)")
    p.add_argument("--max-w", type=int, default=None,
                   help="upper end of the w grid (default: config rules.max_w, 25)")
    p.add_argument("--max-gap", type=float, default=float("inf"),
                   help="T6: largest gap between the two regimes (default: unlimited)")
    g = p.add_argument_group("scoring (R1)")
    g.add_argument("--score", action="store_true",
                   help="compute temporal confidence: |condition AND conclusion| / "
                        "|condition|, counting condition-days only")
    g.add_argument("--in-sample", action="store_true",
                   help="with --score, score over the whole timeline instead of fitting "
                        "on train and scoring on test. Quick look only — the constants "
                        "were fitted on that same data, so it is not a result")
    g.add_argument("--breaks", action="store_true",
                   help="also print confidence per calendar year (R2 break detection)")

    p.add_argument("--out", default=None,
                   help="results table (default: results/rules.csv for --source py, "
                        "results/rules_<source>.csv otherwise)")
    p.add_argument("--tag", default=None,
                   help="suffix for the output file: rules_<tag>.csv. Defaults to the "
                        "source for any source other than py, so --source classifier "
                        "writes rules_classifier.csv and never replaces rules.csv. "
                        "Ignored if --out is given explicitly")
    p.add_argument("--labels-dir", default=None,
                   help="read this --source's label folders from DIR instead of the default "
                        "(e.g. a candidate classifier's infer.py --out-root); use with --tag "
                        "so the paper's rules files are not touched")
    p.add_argument("--list-templates", action="store_true")
    p.add_argument("--list-pairs", action="store_true")
    args = p.parse_args(argv)

    # --tag names the output file, --source picks the rules. A tag that is itself a
    # source name but differs from --source is always a mistake: `--tag classifier`
    # alone used to verify the Python-labeller rules and file them as the classifier's.
    _sources = ("py", "reasoner", "classifier")
    if args.tag in _sources and args.tag != args.source:
        p.error(f"--tag {args.tag} with --source {args.source}: the file would be named "
                f"after one label source and hold another's rules. Use --source {args.tag}")

    from tslearning.rules import generate as G
    from tslearning.rules.templates import TEMPLATES

    if args.labels_dir:
        if not args.tag or args.tag in _sources:
            p.error("--labels-dir needs its own --tag (e.g. --tag preview_v8p3), so the "
                    "paper's rules files are not overwritten")
        G.SOURCE_DIR_OVERRIDE[args.source] = args.labels_dir

    if args.list_templates:
        print(f"{'template':24s} needs")
        for name in sorted(TEMPLATES):
            print(f"{name:24s} {', '.join(TEMPLATES[name]['needs'])}")
            print(f"{'':24s} {TEMPLATES[name]['doc']}")
        return None

    if args.list_pairs:
        pairs = G.discover_pairs(args.source)
        print(f"pairs under --source {args.source} ({G.source_dir(args.source)}):")
        for q in pairs:
            print("   ", q)
        return pairs

    if args.templates:
        unknown = [t for t in args.templates if t not in TEMPLATES]
        if unknown:
            p.error(f"unknown template(s) {unknown}. Known: {sorted(TEMPLATES)}")

    from tslearning.rules.fitting import MIN_W, W_GRID   # config rules.min_w / max_w
    w_grid = W_GRID if args.max_w is None else range(MIN_W, args.max_w + 1)
    label = "rules+scores" if args.score else "rules"
    with stage(f"{label} · {args.source} · {','.join(args.pairs)}", args.quiet):
        out = G.generate(pairs=args.pairs, source=args.source, templates=args.templates,
                         ground=args.ground, ci=args.ci, verbose=not args.quiet,
                         score=args.score, split=not args.in_sample,
                         pct=args.pct, w_grid=w_grid,
                         max_gap=args.max_gap)

    table = out["table"]
    from pathlib import Path
    from tslearning.config import PATHS
    if args.out:
        dest = Path(args.out)
    elif args.tag:
        dest = PATHS["results"] / f"rules_{args.tag}.csv"
    elif args.source != "py":
        # rules.csv is the exact-labeller table the paper reports; another source must
        # never land there by accident (it used to, whenever --tag was forgotten)
        dest = PATHS["results"] / f"rules_{args.source}.csv"
    else:
        dest = PATHS["results"] / "rules.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(dest, index=False)

    ok = int(table["accepted"].sum())
    print(f"\n{ok} rules from {len(out['pairs'])} pair(s), "
          f"{len(table) - ok} rejected -> {dest}")

    if args.score and not args.in_sample and "test_confidence" in table.columns:
        held = table[table["accepted"] & table["test_confidence"].notna()]
        if len(held):
            print("\ntrain -> test confidence (frozen constants):")
            for _, r in held.sort_values("test_confidence", ascending=False).iterrows():
                elig = "" if r.get("test_eligible", True) else "  (too few condition days)"
                print(f"  {r['template']:22s} {r['pair']:30s} "
                      f"{r['train_confidence']:.2f} -> {r['test_confidence']:.2f}{elig}")

    if args.breaks:
        from tslearning.rules import scoring
        print("\nconfidence per year (frozen constants) — a collapse flags a break:")
        for rule in out["rules"]:
            if not rule.accepted or not rule.parts:
                continue
            facts = G.load_pair_facts(rule.pair, args.source, verbose=False)
            df = scoring.score_by_slice(rule, facts)
            df = df[df["confidence"].notna()]
            if df.empty:
                continue
            print(f"  {rule.template} · {rule.pair}: " + "  ".join(
                f"{r['slice'][:4]}={r['confidence']:.2f}" for _, r in df.iterrows()))
    return out


if __name__ == "__main__":
    main()
