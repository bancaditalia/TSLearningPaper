#!/usr/bin/env python3
"""
Stage 8 — evaluation. The two tables that justify the pipeline.

  --metric fidelity   does the student agree with the reasoner on data it never
        trained on? The number that counts is on held-out REAL windows, not
        synthetic ones — synthetic-to-real is exactly where the student was
        observed to collapse. Every accuracy the notebooks print is training fit;
        this is not.
  --metric speed      reasoner seconds vs student seconds per pair. This is the
        whole justification for having a neural stage, and belongs in the abstract.
  --metric all        fidelity + speed (speed needs --reasoner-seconds or --measure-live).

Labelling real test windows needs a reasoner run, and the reasoner is slow, so the
real test set is SUBSAMPLED (--step 10 -> every 10th window -> ~70 windows ~= 3
batches). That run doubles as the source of real training labels for fixing the
collapse.

Examples
--------
    # build + label the subsampled real test set, then score it
    python scripts/eval.py --metric fidelity --dataset real --label-first

    # score against labels already on disk
    python scripts/eval.py --metric fidelity --dataset real
    python scripts/eval.py --metric fidelity --dataset synthetic

    # speed table using a batch you already timed (preferred for the paper:
    # a fresh single-batch timing includes JVM warm-up and understates the cost)
    python scripts/eval.py --metric speed --reasoner-seconds 300 --reasoner-pairs 25

    # or time a live batch now (needs the engine running)
    python scripts/eval.py --metric speed --measure-live

Writes results/fidelity_real_<other side>.csv (or --save); results/speed_agreement.csv
only for --metric speed/all with --reasoner-seconds or --measure-live.
"""
from _cli import bootstrap, parser, stage

bootstrap()


def default_save(kind: str, dataset: str, pair, weights) -> "Path":
    """Where a fidelity/agreement row goes when --save isn't given.

    Keeps the names already used by the paper and the README —
    fidelity_real_gold.csv, fidelity_real_bitcoin.csv, fidelity_synthetic.csv,
    agreement_gold.csv — named after the NON-S&P side of the pair. Weights other than the
    config default add their file stem, so comparing two classifiers writes two files
    instead of one overwriting the other.
    """
    from pathlib import Path
    from tslearning.config import PATHS
    from tslearning.distill.classifier import default_weights

    other = "bitcoin"
    if pair:
        try:
            from tslearning.datasets.pairs import resolve_pair
            other = resolve_pair(pair)[1].lower()
        except Exception:
            other = str(pair).split("_", 1)[-1]
    if kind == "agreement":
        name = f"agreement_{other}"
    else:
        name = ("fidelity_synthetic" if dataset == "synthetic" else
                f"fidelity_valid_{other}" if dataset == "valid" else f"fidelity_real_{other}")
    if kind == "fidelity" and weights and \
            Path(weights).resolve() != default_weights().resolve():
        name += f"_{Path(weights).stem}"
    return PATHS["results"] / f"{name}.csv"


def main(argv=None):
    p = parser(__doc__.split("Examples")[0].strip(), epilog=__doc__)
    p.add_argument("--metric", default="fidelity",
                   choices=["fidelity", "speed", "agreement", "all"],
                   help="which table to produce (default: fidelity; `all` = fidelity + "
                        "speed). `agreement` compares "
                        "py_labeler against the reasoner at the PREDICATE level — the "
                        "link rule verification does not cover")
    p.add_argument("--dataset", default="real", choices=["real", "synthetic", "valid"],
                   help="fidelity only: which held-out set (default: real = the 2022-2024 "
                        "test period). valid = real windows of the TRAINING years with exact "
                        "Python labels, for choosing between classifiers (use --pair and "
                        "--step; step default 10, the paper's comparison uses 5)")
    p.add_argument("--classifier", default=None,
                   help="weights to evaluate "
                        "(default: classifier.weights in configs/config.yaml)")
    p.add_argument("--results", default=None,
                   help="folder with the reasoner's predicate CSVs for this set")
    p.add_argument("--save", default=None,
                   help="write the row here. Default for fidelity/agreement: "
                        "results/fidelity_real_<other side>.csv, "
                        "results/fidelity_synthetic.csv or results/agreement_<other side>.csv, "
                        "plus _<weights> when --classifier isn't the config default")

    g = p.add_argument_group("fidelity")
    g.add_argument("--step", type=int, default=10,
                   help="subsampling stride for the real test set (default: 10)")
    g.add_argument("--label-first", action="store_true",
                   help="build the real-test batches and run the reasoner before scoring")
    g.add_argument("--train-pairs", default=None,
                   help="--dataset synthetic: the pairs the classifier was trained on, in "
                        "training order (default: paper.pairs). The held-out part of that "
                        "training split is scored. For v7's flat set pass "
                        "--results data/processed/batched_results instead")
    g.add_argument("--test-size", type=float, default=0.2,
                   help="--dataset synthetic: the held-out fraction used in training "
                        "(distill.py --test-size, default 0.2)")
    g.add_argument("--per-pair-route", action="store_true",
                   help="with --pair sp500_bitcoin: label/score bitcoin through the per-pair "
                        "route (real_test_sp500_bitcoin/), like the other pairs, instead of "
                        "the older aligned-pickle route. Needed for --metric agreement")
    g.add_argument("--pair", default=None,
                   help="which pair to measure fidelity on (default: sp500_bitcoin, "
                        "via the aligned_windows_real pickle). Any other pair is "
                        "aligned on the fly and gets its own results folder. Use "
                        "sp500_gold to score the NEGATIVE class — bitcoin has only "
                        "1 negative window in 71, so its macro-F1 is uninformative")

    g = p.add_argument_group("speed")
    g.add_argument("--measure-live", action="store_true",
                   help="time one live reasoner batch (needs the engine running)")
    g.add_argument("--reasoner-seconds", type=float, default=None,
                   help="wall-clock of a batch you already ran. Required for the speed "
                        "table unless --measure-live: there is no default, because the "
                        "old one (300 s = 12 s/window-pair) was an assumption, not a "
                        "measurement. The measured route is classifier_timing.py + "
                        "timing_table.py")
    g.add_argument("--reasoner-pairs", type=int, default=25,
                   help="pairs in that batch (default: 25)")
    g.add_argument("--no-agreement", action="store_true",
                   help="speed only, skip the agreement columns")

    args = p.parse_args(argv)
    if args.metric in ("speed", "all") and args.reasoner_seconds is None \
            and not args.measure_live:
        p.error("the speed table needs the reasoner's time: pass --reasoner-seconds "
                "(and --reasoner-pairs) from a batch you timed, or --measure-live. "
                "For the paper, use classifier_timing.py + timing_table.py instead")
    verbose = not args.quiet
    out = {}

    if args.metric in ("fidelity", "all"):
        from tslearning.eval import fidelity

        # bitcoin goes through the older aligned-pickle route unless --per-pair-route
        generic_pair = args.pair and (args.pair not in ("sp500_bitcoin",)
                                      or args.per_pair_route)

        if args.label_first:
            if args.dataset != "real":
                p.error("--label-first only applies to --dataset real")
            label = f"fidelity · label the real test set" + (
                f" ({args.pair})" if generic_pair else "")
            with stage(label, args.quiet):
                if generic_pair:
                    fidelity.label_real_pair_test_set(args.pair, step=args.step,
                                                      result_dir=args.results,
                                                      verbose=verbose)
                else:
                    # same folder as the scoring below reads (it used to ignore --results)
                    fidelity.label_real_test_set(step=args.step, result_dir=args.results,
                                                 verbose=verbose)

        with stage(f"fidelity · {args.dataset}", args.quiet):
            out["fidelity"] = fidelity.run(dataset=args.dataset,
                                           weights=args.classifier,
                                           results_dir=args.results, step=args.step,
                                           # a --label-first run scores against
                                           # labels not yet relabelled (bitcoin):
                                           # only saved if asked explicitly
                                           save=args.save or (None if args.label_first
                                                else default_save(
                                               "fidelity", args.dataset, args.pair,
                                               args.classifier)),
                                           pair=args.pair,
                                           train_pairs=(args.train_pairs.split(",")
                                                        if args.train_pairs else None),
                                           test_size=args.test_size, verbose=verbose)
        row = out["fidelity"]["row"]
        print(f"fidelity ({row['dataset']}, n={row['n_windows']}): "
              f"accuracy {row['accuracy']:.3f}, macro-F1 {row['macro_f1']:.3f}")

    if args.metric == "agreement":
        from tslearning.eval import agreement
        with stage(f"label agreement · {args.pair or 'sp500_bitcoin'}", args.quiet):
            out["agreement"] = agreement.run(pair=args.pair or "sp500_bitcoin",
                                             step=args.step,
                                             results_dir=args.results,
                                             save=args.save or default_save(
                                                 "agreement", "real", args.pair, None),
                                             verbose=verbose)
        r = out["agreement"]["row"]
        print(f"label agreement ({r['pair']}, n={r['n_scored']}): {r['agreement']:.3f} "
              f"| {r['n_disagree']} disagree, {r['n_unlabelled_by_engine']} unlabelled")
        return out

    if args.metric in ("speed", "all"):
        from tslearning.eval import speed
        with stage("speed + agreement", args.quiet):
            out["speed"] = speed.run(weights=args.classifier, results_dir=args.results,
                                     measure_live=args.measure_live,
                                     reasoner_seconds=args.reasoner_seconds,
                                     reasoner_pairs=args.reasoner_pairs,
                                     with_agreement=not args.no_agreement,
                                     save=args.save if args.metric == "speed" else None,
                                     verbose=verbose)
        print(f"speedup: {out['speed']['speedup']:,.0f}x "
              f"({out['speed']['reasoner_s_per_pair']:.3f} s/pair reasoner vs "
              f"{out['speed']['clf_s_per_pair']*1e3:.4f} ms/pair student)")

    return out


if __name__ == "__main__":
    main()
