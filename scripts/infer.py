#!/usr/bin/env python3
"""
Stage 6 — inference. Annotate the real timeline with the trained student.

This is where the distillation pays off: the reasoner would need days for the full
history times a lag scan; the student needs one forward pass per lag.

Default mode (since 2026-09-02) — one folder per pair, same layout as the Python path:

    data/processed/real_pairs_regime_classifier/<pair>/
        isPositiveCorrelation_<pair>.csv     ┐ the three regime predicates the
        isNegativeCorrelation_<pair>.csv     │ student PREDICTS
        isNoCorrelation_<pair>.csv           ┘
        predictions_lag0.pkl                 softmax probabilities (for HMM smoothing)
        highVol_… trendDown_… tailDependent_… leads*_… us_inflation*_… <pair>_dates.csv
                                             copied verbatim from real_pairs_regime_py/<pair>/
                                             — exact computations, no network involved

So `generate_rules.py --source classifier` and `--source py` read identical layouts and
differ ONLY in who produced the regime predicates. That is the comparison the paper
needs: same conditions, same scoring, conclusion from the exact labeller vs. from the
distilled student.

Requires the pair to be Python-labelled first (it supplies the calendar and the exact
flags): `prep_real_pairs.py -p <pair> --engine python --stride 1`.

Weights default to `classifier.weights` in configs/config.yaml — the paper's classifier.
The pre-fix weights (classifier_v7_corr.pt) are kept on disk for the ablation row only.

`--legacy-bundle` restores the old flat `temporal/` output for nb_10_part_2 Part B.

Examples
--------
    python scripts/infer.py -p sp500_gold
    python scripts/infer.py -p sp500_gold,sp500_bitcoin,sp500_nasdaq
    python scripts/infer.py -p all
    python scripts/infer.py -p sp500_bitcoin --classifier data/processed/classifier_v7_corr.pt \\
           --out-root data/processed/real_pairs_regime_classifier_prefix     # ablation
    python scripts/infer.py -p sp500_bitcoin --legacy-bundle --lags 0,1,2,3,4,5

Next step: `generate_rules.py --source classifier -p all --score`.
"""
from _cli import bootstrap, csv_list, int_list, parser, stage

bootstrap()


def main(argv=None):
    p = parser(__doc__.split("Examples")[0].strip(), epilog=__doc__)
    p.add_argument("-p", "--pairs", type=csv_list, default=["sp500_bitcoin"],
                   help="pairs as <a>_<b>, comma separated, or 'all' for every pair "
                        "labelled under real_pairs_regime_py (default: sp500_bitcoin)")
    p.add_argument("--classifier", default=None,
                   help="weights to load (default: classifier.weights in configs/config.yaml)")
    p.add_argument("--stride", type=int, default=1,
                   help="window stride on the real timeline (default: 1, matching the "
                        "Python labelling the rules were built from)")
    p.add_argument("--out-root", default=None,
                   help="parent folder for the per-pair output "
                        "(default: data/processed/real_pairs_regime_classifier)")
    p.add_argument("--no-copy-exact", action="store_true",
                   help="do not copy the exact-flag CSVs from the Python labelling "
                        "(the folder will then hold regime predicates only)")
    g = p.add_argument_group("legacy single-pair bundle (nb_10_part_2 Part B)")
    g.add_argument("--legacy-bundle", action="store_true",
                   help="old behaviour: flat data/processed/temporal/ from the aligned "
                        "pickle, sp500_bitcoin only")
    g.add_argument("--lags", type=int_list, default=[0],
                   help="legacy only — lags to predict (default: 0)")
    g.add_argument("--out-dir", default=None, help="legacy only — flat output folder")
    g.add_argument("--no-flags", action="store_true",
                   help="legacy only — skip the exact flag CSVs")
    args = p.parse_args(argv)

    from tslearning.distill import inference

    if args.legacy_bundle:
        pair = args.pairs[0]
        with stage(f"infer (legacy bundle) · {pair} · lags {args.lags}", args.quiet):
            r = inference.run(pair=pair, weights=args.classifier, lags=args.lags,
                              out_dir=args.out_dir, write_flags=not args.no_flags,
                              verbose=not args.quiet)
        n = len(r["bundles"][args.lags[0]]["regime"])
        print(f"{pair}: {n} timepoints x {len(args.lags)} lag(s) -> {r['out_dir']}")
        return r

    pairs = list(args.pairs)
    if pairs == ["all"]:
        from tslearning.rules.generate import discover_pairs
        pairs = discover_pairs("py")
        if not pairs:
            p.error("no pairs under real_pairs_regime_py — run prep_real_pairs.py first")

    results = []
    with stage(f"infer · {','.join(pairs)}", args.quiet):
        for pair in pairs:
            results.append(inference.run_pair(
                pair, weights=args.classifier, out_root=args.out_root,
                stride=args.stride, copy_exact=not args.no_copy_exact,
                verbose=not args.quiet))

    print()
    for r in results:
        c = r["counts"]
        print(f"  {r['pair']:16s} {r['n_windows']:5d} windows  "
              f"pos {c.get('isPositiveCorrelation', 0):5d}  "
              f"neg {c.get('isNegativeCorrelation', 0):5d}  "
              f"none {c.get('isNoCorrelation', 0):5d}")
    print(f"\n{len(results)} pair(s) -> {results[0]['out_dir'].parent}")
    return results


if __name__ == "__main__":
    main()
