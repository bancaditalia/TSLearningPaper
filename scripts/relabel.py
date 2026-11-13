#!/usr/bin/env python3
"""
Re-derive the correlation regime from percentages the reasoner already wrote.

Replaces the two ABSOLUTE cuts in preLabeling.vada (same% > 55 / diff% > 45) with
RELATIVE DOMINANCE (score = same% - diff%, compared against a deadband). Runs no
reasoning: the concordance was already computed, only the threshold applied afterwards
is being changed. 21 batches take about a second.

Why: on real daily data the absolute bars are effectively unreachable — measured max
diff% is 32 against a bar of 45 — so isNegativeCorrelation never fires and every real
window comes out "no correlation". A classifier distilled against that learns a
degenerate function. See tslearning/labeling/relabel.py for the full argument.

The output directory is a drop-in replacement: everything but the three regime CSVs is
copied through, so distill.py and eval.py read it via --results with no other change.

Examples
--------
    # the synthetic transfer set (what the classifier trains on)
    python scripts/relabel.py --results data/processed/batched_results

    # the real test set (what fidelity is measured against)
    python scripts/relabel.py --results data/processed/real_test_results

    # try a different deadband
    python scripts/relabel.py --results data/processed/batched_results --deadband 5
"""
from _cli import bootstrap, parser, stage

bootstrap()


def main(argv=None):
    p = parser(__doc__.strip().split("\n")[0], epilog=__doc__)
    p.add_argument("--results", required=True,
                   help="folder of reasoner output to relabel "
                        "(e.g. data/processed/batched_results)")
    p.add_argument("-o", "--out-dir", default=None,
                   help="where to write (default: <results>_dominance)")
    p.add_argument("--deadband", type=float, default=None,
                   help="margin in percentage points before committing to a regime "
                        "(default: 10)")
    p.add_argument("--no-copy", action="store_true",
                   help="write only the regime CSVs, do not copy the other predicates "
                        "(the output is then NOT a drop-in replacement)")
    p.add_argument("--all-tags", action="store_true",
                   help="keep non-numeric batch tags. Default skips them: "
                        "batched_results holds a stray *_test.csv set that reuses "
                        "window indices from batch 000 with different values, which "
                        "puts two contradictory regimes on the same window")
    args = p.parse_args(argv)

    with stage(f"relabel · {args.results}", args.quiet):
        from tslearning.labeling import relabel
        r = relabel.run(args.results, out_dir=args.out_dir,
                        deadband=args.deadband if args.deadband is not None
                        else relabel.DEFAULT_DEADBAND,
                        copy_others=not args.no_copy, all_tags=args.all_tags,
                        verbose=not args.quiet)

    n = sum(r["after"].values())
    print(f"{len(r['tags'])} batches, {n} cross-series windows -> {r['out_dir']}")
    return r


if __name__ == "__main__":
    main()
