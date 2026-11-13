#!/usr/bin/env python3
"""
The orchestrator — run the pipeline end to end, or just the tail of it.

Every stage is also a standalone script (see scripts/README.md or the --help of
each); this file only sequences them and stops at the first failure, so a broken
stage never feeds garbage to the next.

Stages, in dependency order:

    1 data_prep         nb_00 / nb_00b     download, transform, window, fingerprint
    2 synthesize        nb_06 / nb_06b     train a DDPM per series, sample from it
    3 transfer_set      nb_10 (prep)       build + batch the pairs to be labelled
    4 label             nb_10 (vada)       the reasoner annotates them  [SLOW]
    5 distill           nb_10 (V7)         train the student to imitate the reasoner
    6 infer             nb_10_part_2 B     annotate the real timeline
    7 segment           nb_11 (first half) smooth + coalesce into intervals

Two modes:

  --stage full     all seven. Expect this to take days, because stage 4 is the
        reasoner and there is no way around that. Use
        --engine python to substitute the validated Python labeler and finish in
        minutes, at the cost of the "labels came from a declarative program" claim.

  --stage predict  stages 6-7 only. Assumes a trained classifier and existing labels
        are already on disk. This is the loop you actually iterate in while working
        on segmentation and rules — it is seconds, not days.

Rule generation is deliberately NOT a stage here: the template library, scoring and
break detection are still in development. Stage 7 hands it
a clean intervals CSV; the notebook takes it from there.

Examples
--------
    # the full thing, S&P x Bitcoin, reasoner labels
    python scripts/pipeline.py --stage full -d SP500,Bitcoin -p sp500_bitcoin

    # full, but skip the days-long reasoner run
    python scripts/pipeline.py --stage full -d SP500,Gold -p sp500_gold --engine python

    # just re-annotate and re-segment with a different smoothing
    python scripts/pipeline.py --stage predict -p sp500_bitcoin --smoothing majority

    # see the plan without running anything
    python scripts/pipeline.py --stage full -p sp500_bitcoin --dry-run

    # resume: data + synthesis already done
    python scripts/pipeline.py --stage full -p sp500_bitcoin --skip data_prep,synthesize
"""
from _cli import bootstrap, csv_list, int_list, parser, stage

bootstrap()

FULL_STAGES = ["data_prep", "synthesize", "transfer_set", "label", "distill",
               "infer", "segment"]
PREDICT_STAGES = ["infer", "segment"]


def main(argv=None):
    p = parser(__doc__.split("Examples")[0].strip(), epilog=__doc__)
    p.add_argument("--stage", default="full", choices=["full", "predict"],
                   help="full = all stages; predict = infer + segment only")
    p.add_argument("-d", "--datasets", type=csv_list, default=["SP500", "Bitcoin"],
                   help="series to prepare and synthesize (default: SP500,Bitcoin)")
    p.add_argument("-p", "--pair", default="sp500_bitcoin",
                   help="pair to label, distill, infer and segment "
                        "(default: sp500_bitcoin)")
    p.add_argument("--engine", default="vadalog", choices=["vadalog", "python"],
                   help="labelling engine for stage 4 (default: vadalog)")
    p.add_argument("--mode", default="engineered", choices=["engineered", "plain"],
                   help="transfer-set generator (default: engineered)")
    p.add_argument("--epochs", type=int, default=None,
                   help="synthesizer epochs (per series)")
    p.add_argument("--distill-epochs", type=int, default=1500)
    p.add_argument("--n-pairs", type=int, default=None,
                   help="pairs in the transfer set")
    p.add_argument("--lags", type=int_list, default=[0], help="lags to infer")
    p.add_argument("--smoothing", default=None, choices=["hmm", "majority", "none"])
    p.add_argument("--skip", type=csv_list, default=[],
                   help="stages to skip, comma separated (e.g. data_prep,synthesize)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and exit")
    args = p.parse_args(argv)

    stages = FULL_STAGES if args.stage == "full" else PREDICT_STAGES
    stages = [s for s in stages if s not in args.skip]
    verbose = not args.quiet

    if args.dry_run:
        print(f"plan ({args.stage}): pair={args.pair}, datasets={','.join(args.datasets)}, "
              f"engine={args.engine}")
        for i, s in enumerate(stages, 1):
            print(f"  {i}. {s}")
        if args.skip:
            print(f"  skipping: {', '.join(args.skip)}")
        return None

    results = {}

    if "data_prep" in stages:
        from tslearning.datasets import ingest_financial
        with stage("1/7 data prep", args.quiet):
            results["data_prep"] = ingest_financial.run(
                series=args.datasets, verbose=verbose)

    if "synthesize" in stages:
        from tslearning.synth import ddpm
        for d in args.datasets:
            with stage(f"2/7 synthesize · {d}", args.quiet):
                results.setdefault("synthesize", {})[d] = ddpm.run(
                    d, epochs=args.epochs, verbose=verbose)

    if "transfer_set" in stages:
        from tslearning.labeling import transfer_set
        with stage(f"3/7 transfer set · {args.pair}", args.quiet):
            results["transfer_set"] = transfer_set.run(
                pair=args.pair, mode=args.mode, n_pairs=args.n_pairs, verbose=verbose)

    if "label" in stages:
        with stage(f"4/7 label · {args.pair} · {args.engine}", args.quiet):
            results["label"] = _label(args, verbose)

    if "distill" in stages:
        from tslearning.distill import train
        with stage(f"5/7 distill · {args.pair}", args.quiet):
            results["distill"] = train.run(pair=args.pair, epochs=args.distill_epochs,
                                           verbose=verbose)

    if "infer" in stages:
        from tslearning.distill import inference
        with stage(f"6/7 infer · {args.pair}", args.quiet):
            # per-pair layout (real_pairs_regime_classifier/<pair>/), readable by
            # generate_rules.py --source classifier. Needs the pair Python-labelled first.
            results["infer"] = inference.run_pair(args.pair, verbose=verbose)

    if "segment" in stages:
        from tslearning.segment import predicate_io
        with stage(f"7/7 segment · {args.pair}", args.quiet):
            results["segment"] = predicate_io.run(pair=args.pair,
                                                  smoothing=args.smoothing,
                                                  verbose=verbose)

    print(f"\npipeline ({args.stage}) finished: {', '.join(stages)}")
    if "segment" in results:
        print(f"intervals -> {results['segment']['out_path']}")
        print("next: nb_11's rule-mining cells, pointed at that file")
    return results


def _label(args, verbose):
    """Stage 4, either engine. The Python engine labels the real timeline directly,
    so it also stands in for stage 6's inputs when no reasoner is available."""
    from pathlib import Path
    from tslearning.config import CFG, PATHS

    data_dir = PATHS["data_processed"]

    if args.engine == "vadalog":
        from tslearning.labeling import reasoner_client
        paths = sorted(Path(data_dir).glob("batch_*.csv"))
        if not paths:
            raise FileNotFoundError("no batch_*.csv — stage 3 must run first")
        return reasoner_client.run_batches(
            paths, data_dir / "batched_results",
            tags=[p.stem.split("_")[-1] for p in paths], verbose=verbose)

    from tslearning.datasets.data_utils import load_transformed
    from tslearning.labeling.py_labeler import label_pair
    from tslearning.labeling.transfer_set import load_thresholds

    a_name, b_name = args.pair.split("_", 1)
    lut = {n.lower(): n for n in CFG["series_freq"]}
    transformed = load_transformed()
    tau, qlow = load_thresholds()
    return label_pair(args.pair, transformed[lut[a_name]], transformed[lut[b_name]],
                      b_name, tau, qlow,
                      out_dir=data_dir / "real_pairs_regime_py",
                      lo=CFG["real_pairs"]["date_from"], hi=CFG["real_pairs"]["date_to"],
                      stride=CFG["real_pairs"].get("stride", 1), base_name=a_name,
                      seq_len=CFG["windows"]["financial"]["length"],
                      lags=CFG["real_pairs"].get("lags"))


if __name__ == "__main__":
    main()
