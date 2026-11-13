#!/usr/bin/env python3
"""
Stage 4b — real pairs. Prepare and label real market pairs end to end.

This is the multi-pair path:
a rule confirmed on ONE pair may be an accident, whereas the same rule holding for
equity-crypto and failing for equity-gold IS the finding. So this script takes
several pairs at once and gives each its own predicate set on the real timeline.

Two tracks, mirroring the notebook:

  --engine python    (Track A, fast) fit thresholds, then label every pair in
        Python. Seconds. Validated against the reasoner at 96.5% regime agreement.
  --engine vadalog   (Track B, slow) fit thresholds, build disjoint-block batches,
        run the reasoner over them (cached + resumable), then rescore by dominance.
        Hours to days, but the labels are the declarative ones.

Financial or macro, same command. A pair is two series of the SAME frequency:

    -p sp500_gold                          financial x financial (daily, 50-day windows)
    -p US_Inflation:US_Unemployment        macro x macro (monthly, 12-month windows)
    -p DE_Inflation:IT_Inflation           cross-country macro

Window length and stride come from the series' configured frequency, so nothing
needs saying on the command line. Use the colon form for macro (their names contain
underscores); the underscore form still works and is resolved against the known
series list, so `us_inflation_us_unemployment` parses correctly too.

Mixing frequencies (SP500 x US_Inflation) is refused with an explanation — that is
the T7 macro-conditioning template, a different computation.

Examples
--------
    # fast path, three pairs
    python scripts/prep_real_pairs.py -p sp500_gold,sp500_nasdaq,sp500_bitcoin

    # macro pairs
    python scripts/prep_real_pairs.py -p US_Inflation:US_Unemployment
    python scripts/prep_real_pairs.py -p DE_Inflation:IT_Inflation,DE_GDP:IT_GDP

    # reasoner path
    python scripts/prep_real_pairs.py -p sp500_gold --engine vadalog

    # only build the batch CSVs now, run the reasoner later/elsewhere
    python scripts/prep_real_pairs.py -p sp500_gold --engine vadalog --batches-only
    python scripts/label.py -p sp500_gold --track real --engine vadalog

    # tighter subsampling and a wider dead band
    python scripts/prep_real_pairs.py -p sp500_gold --stride 5 --deadband 8

Writes:
    thresholds/tau.csv, thresholds/qlow.csv        (fitted on the TRAIN period)
    real_pairs_batches/<pair>_batch_NNN.csv        (vadalog track only)
    real_pairs_results/                            (reasoner predicate CSVs)
    real_pairs_regime_py/<pair>/                   (python track output)
    real_pairs_regime_reasoner/<pair>/             (recalibrated reasoner output)

Next step: `generate_rules.py --source py -p <pair> --score --ci`.
"""
from _cli import bootstrap, csv_list, int_list, parser, stage

bootstrap()


def main(argv=None):
    p = parser(__doc__.split("Examples")[0].strip(), epilog=__doc__)
    p.add_argument("-p", "--pairs", type=csv_list, default=["sp500_gold"],
                   help="comma-separated pairs. Underscore form (sp500_gold) or "
                        "colon form (US_Inflation:US_Unemployment). Both sides must "
                        "share a frequency")
    p.add_argument("--engine", default="python", choices=["python", "vadalog"],
                   help="python = fast validated bypass (default); "
                        "vadalog = the reasoner (slow, cached, resumable)")
    p.add_argument("--stride", type=int, default=None,
                   help="take every Nth window (default: real_pairs.stride in config, 1)")
    p.add_argument("--batch-size", type=int, default=25,
                   help="window pairs per reasoner batch (default: 25)")
    p.add_argument("--deadband", type=float, default=None,
                   help="dominance margin in percentage points. Default comes from "
                        "configs/config.yaml labels.deadband — the same value the "
                        "reasoner receives as ${Deadband}")
    p.add_argument("--date-from", default=None,
                   help="first date (default: real_pairs.date_from in config, 2015-01-01). "
                        "For monthly macro pairs pass a wider range, e.g. 1996-01-01")
    p.add_argument("--date-to", default=None,
                   help="last date (default: real_pairs.date_to in config)")
    p.add_argument("--seq-len", type=int, default=None,
                   help="window length (default: from the pair's configured "
                        "frequency — 50 for daily, 12 for monthly, 8 for quarterly)")
    p.add_argument("--lags", type=csv_list, default=None,
                   help="python engine: lead-lag predicates at these lags, e.g. 1,2,3 "
                        "(default: real_pairs.lags in config, 1-5; 'config' = the same)")
    p.add_argument("--no-lags", action="store_true",
                   help="python engine: skip the lead-lag predicates (T4/T4b then "
                        "report missing predicates)")
    p.add_argument("--keep-old", action="store_true",
                   help="python engine: don't move the pair's previous label folder "
                        "aside first (default: move it to data/processed/_stale/)")
    p.add_argument("--no-macro", action="store_true",
                   help="python engine: don't recompute the macro state "
                        "(real_pairs.macro_states) after labelling")
    p.add_argument("--no-fit-thresholds", action="store_true",
                   help="reuse the existing thresholds/*.csv")
    p.add_argument("--batches-only", action="store_true",
                   help="vadalog track: write the batch CSVs and stop")
    p.add_argument("--skip-recalibrate", action="store_true",
                   help="vadalog track: don't rescore by dominance afterwards")
    args = p.parse_args(argv)

    from pathlib import Path
    from tslearning.config import CFG, PATHS
    from tslearning.datasets.pairs import describe, load_series
    from tslearning.labeling import transfer_set

    verbose = not args.quiet
    data_dir = PATHS["data_processed"]
    # the paper's settings live in config `real_pairs`; flags override them
    rp = CFG.get("real_pairs") or {}
    args.date_from = args.date_from or rp.get("date_from", "2015-01-01")
    args.date_to = args.date_to or rp.get("date_to", "2024-12-31")
    args.stride = args.stride or rp.get("stride", 1)
    if args.no_lags:
        lags = None
    elif args.lags in (None, ["config"]):
        lags = rp.get("lags") or CFG["predicates"]["lags"]
    else:
        lags = [int(v) for v in args.lags]
    if verbose:
        print(f"  settings: {args.date_from}..{args.date_to}, stride {args.stride}, "
              f"lags {lags or 'off'}")

    # Resolve every pair up front, so a typo or a frequency mismatch fails now
    # rather than halfway through a long run.
    specs = [describe(spec) for spec in args.pairs]
    series_names = sorted({s for d in specs for s in (d["a"], d["b"])})
    if verbose:
        for d in specs:
            print(f"  {d['slug']:34s} {d['a']} x {d['b']}  ({d['freq']}, "
                  f"window={d['seq_len']})")

    if not args.no_fit_thresholds:
        with stage("thresholds (tau, qLow) on the TRAIN period", args.quiet):
            transfer_set.fit_thresholds(series_names, verbose=verbose)
    tau, qlow = transfer_set.load_thresholds()

    transformed = load_series(series_names)
    out = {}

    for d in specs:
        pair, a_name, b_name = d["slug"], d["a_key"], d["b_key"]
        seq_len = args.seq_len or d["seq_len"]

        if args.engine == "python":
            from tslearning.labeling.py_labeler import label_pair, retire_pair_dir
            if not args.keep_old:
                retire_pair_dir(data_dir / "real_pairs_regime_py", pair, verbose)
            with stage(f"real pairs · {pair} · python", args.quiet):
                s = label_pair(pair, transformed[d["a"]], transformed[d["b"]],
                               b_name, tau, qlow,
                               out_dir=data_dir / "real_pairs_regime_py",
                               lo=args.date_from, hi=args.date_to,
                               stride=args.stride, seq_len=seq_len,
                               deadband=args.deadband, base_name=a_name, lags=lags)
            print(f"{pair}: {s['windows']} windows -> {s['regime_counts']}")
            if s.get("lead_counts"):
                for lag, c in sorted(s["lead_counts"].items()):
                    mark = " <- strongest" if lag == s["strongest_lag"] else ""
                    print(f"    lag {lag}: {dict(c)}{mark}")
            if not args.no_macro:
                from tslearning.labeling.macro_state import run_configured
                run_configured(pair, source="py", verbose=verbose)
            from tslearning.distill.inference import warn_if_stale
            warn_if_stale(pair, "after relabelling")
            out[pair] = s
            continue

        # ── vadalog track ──────────────────────────────────────────────────────
        with stage(f"real pairs · {pair} · build batches", args.quiet):
            info = transfer_set.build_real_batches(
                pair, {a_name: transformed[d["a"]], b_name: transformed[d["b"]]},
                b_name, args.date_from, args.date_to, args.stride, args.batch_size,
                out_dir=data_dir / "real_pairs_batches", seq_len=seq_len, base=a_name,
                verbose=verbose)
        out[pair] = info

        if args.batches_only:
            print(f"{pair}: {info['batches']} batches ready in {info['out_dir']} — "
                  f"run `label.py -p {pair} --track real --engine vadalog`")
            continue

        from tslearning.labeling import reasoner_client
        with stage(f"real pairs · {pair} · reasoner", args.quiet):
            info["reasoner"] = reasoner_client.run_pair_batches(
                pair, data_dir / "real_pairs_batches", data_dir / "real_pairs_results",
                verbose=verbose)

        if not args.skip_recalibrate:
            from tslearning.labeling.recalibrate import recalibrate_pair
            with stage(f"real pairs · {pair} · recalibrate", args.quiet):
                try:
                    _, s = recalibrate_pair(
                        pair, in_dir=data_dir / "real_pairs_results",
                        out_dir=data_dir / "real_pairs_regime_reasoner",
                        deadband=args.deadband, seq_len=seq_len)
                    info["recalibrated"] = s
                    print(f"{pair}: {s['regime_counts']}  "
                          f"score quartiles {s['score_quartiles']}")
                except FileNotFoundError:
                    print(f"{pair}: no reasoner output yet — the batches above must "
                          "finish before recalibrating")
    return out


if __name__ == "__main__":
    main()
