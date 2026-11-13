#!/usr/bin/env python3
"""
Stage 4 — symbolic labelling. Annotate window pairs with the predicate vocabulary.



Two engines produce the SAME predicate CSVs, so everything downstream is
interchangeable:

  --engine vadalog  the reasoner. This is the auditable definition — the labels the
        paper claims are derived from a declarative DatalogMTL program, not from
        code. It is also slow (a full run is hours to days), so it is cached and
        resumable: re-running skips batches whose output already exists.
  --engine python   labeling/py_labeler.py, the reasoner bypass. Validated against
        the reasoner on sp500_gold (171 windows): same-sign concordance exact,
        diff-sign within 2pp, regime label agreement 96.5%. Use it when the engine
        isn't running, or to iterate on thresholds in seconds instead of days.

Examples
--------
    # label the synthetic transfer set with the reasoner (the distillation targets)
    python scripts/label.py -p sp500_bitcoin --engine vadalog

    # same, but keep going past a failing batch
    python scripts/label.py -p sp500_bitcoin --engine vadalog --continue-on-error

    # real timeline, no reasoner needed
    python scripts/label.py -p sp500_gold --track real --engine python

    # real timeline via the reasoner, then rescore by dominance
    python scripts/label.py -p sp500_gold --track real --engine vadalog
    python scripts/label.py -p sp500_gold --track real --engine recalibrate

`--engine recalibrate` is the third, cheap option: it does NOT re-run anything. It
re-derives the regime from the concordance percentages the reasoner already wrote,
by relative dominance (score = sameSign% - diffSign%, compared against --deadband)
instead of the fixed absolute bar. On real data the absolute bar never fires, which
is why every real window came out "no correlation".

--track train : synthetic transfer-set batches (batch_NNN.csv) -> transfer_<pair>/ (same folder)
--track real  : real aligned windows                           -> real_pairs_regime_*/
"""
from _cli import bootstrap, parser, stage

bootstrap()


def _label_train_vadalog(args):
    from pathlib import Path
    from tslearning.config import PATHS
    from tslearning.labeling import reasoner_client

    from tslearning.datasets.pairs import resolve_pair

    data_dir = PATHS["data_processed"]
    # default: the pair's transfer folder holds both the batches and the labels
    # (build_transfer_set.py writes there). v7's flat set: --batch-dir data/processed
    # --results data/processed/batched_results
    tdir = data_dir / f"transfer_{resolve_pair(args.pair)[2]}"
    batch_dir = Path(args.batch_dir or tdir)
    result_dir = Path(args.results or tdir)

    paths = sorted(batch_dir.glob("batch_*.csv"))
    if args.n_batches:
        paths = paths[:args.n_batches]
    if not paths:
        raise FileNotFoundError(
            f"no batch_*.csv in {batch_dir} — run build_transfer_set.py first")

    return reasoner_client.run_batches(
        paths, result_dir, tags=[p.stem.split("_")[-1] for p in paths],
        ruleset=args.ruleset or reasoner_client.default_ruleset(),
        endpoint=args.endpoint,
        stop_on_error=not args.continue_on_error, verbose=not args.quiet)


def _label_real_python(args):
    from pathlib import Path
    from tslearning.config import PATHS
    from tslearning.datasets.pairs import describe, load_series
    from tslearning.labeling.py_labeler import label_pair
    from tslearning.labeling.transfer_set import load_thresholds

    data_dir = PATHS["data_processed"]
    out_dir = Path(args.out_dir or (data_dir / "real_pairs_regime_py"))

    d = describe(args.pair)
    transformed = load_series([d["a"], d["b"]])
    tau, qlow = load_thresholds()
    for side in (d["a_key"], d["b_key"]):
        if side not in tau:
            raise KeyError(f"no fitted tau for '{side}'. Run: prep_real_pairs.py "
                           f"-p {args.pair}  (it fits thresholds for both sides first)")

    # same defaults as prep_real_pairs.py: the paper's settings from config `real_pairs`,
    # lead-lag included, previous folder moved aside, macro state recomputed
    from tslearning.config import CFG
    from tslearning.labeling.macro_state import run_configured
    from tslearning.labeling.py_labeler import retire_pair_dir
    rp = CFG.get("real_pairs") or {}
    retire_pair_dir(out_dir, d["slug"], verbose=not args.quiet)
    r = label_pair(d["slug"], transformed[d["a"]], transformed[d["b"]],
                   d["b_key"], tau, qlow, out_dir=out_dir,
                   lo=args.date_from or rp.get("date_from", "2015-01-01"),
                   hi=args.date_to or rp.get("date_to", "2024-12-31"),
                   stride=args.stride or rp.get("stride", 1),
                   seq_len=args.seq_len or d["seq_len"],
                   deadband=args.deadband, base_name=d["a_key"],
                   lags=rp.get("lags") or CFG["predicates"]["lags"])
    if out_dir == PATHS["data_processed"] / "real_pairs_regime_py":
        run_configured(d["slug"], source="py", verbose=not args.quiet)
        from tslearning.distill.inference import warn_if_stale
        warn_if_stale(d["slug"], "after relabelling")
    return r


def _label_real_vadalog(args):
    from pathlib import Path
    from tslearning.config import PATHS
    from tslearning.labeling import reasoner_client

    data_dir = PATHS["data_processed"]
    batch_dir = Path(args.batch_dir or (data_dir / "real_pairs_batches"))
    result_dir = Path(args.results or (data_dir / "real_pairs_results"))
    if not list(batch_dir.glob(f"{args.pair}_batch_*.csv")):
        raise FileNotFoundError(
            f"no {args.pair}_batch_*.csv in {batch_dir} — run "
            f"`prep_real_pairs.py -p {args.pair} --batches-only` first")
    return reasoner_client.run_pair_batches(
        args.pair, batch_dir, result_dir, n_batches=args.n_batches,
        ruleset=args.ruleset or reasoner_client.default_ruleset(),
        endpoint=args.endpoint,
        stop_on_error=not args.continue_on_error, verbose=not args.quiet)


def _label_recalibrate(args):
    from pathlib import Path
    from tslearning.config import PATHS
    from tslearning.labeling.recalibrate import recalibrate_pair

    data_dir = PATHS["data_processed"]
    in_dir = Path(args.results or (data_dir / "real_pairs_results"))
    out_dir = Path(args.out_dir or (data_dir / "real_pairs_regime_reasoner"))
    from tslearning.datasets.pairs import describe
    _, summary = recalibrate_pair(args.pair, in_dir=in_dir, out_dir=out_dir,
                                  deadband=args.deadband,
                                  seq_len=describe(args.pair)["seq_len"])
    return summary


def main(argv=None):
    p = parser(__doc__.split("Examples")[0].strip(), epilog=__doc__)
    p.add_argument("-p", "--pair", default="sp500_bitcoin",
                   help="pair: sp500_gold or US_Inflation:US_Unemployment "
                        "(default: sp500_bitcoin)")
    p.add_argument("--engine", default="vadalog",
                   choices=["vadalog", "python", "recalibrate"],
                   help="vadalog = the reasoner (slow, authoritative); "
                        "python = validated bypass; recalibrate = rescore existing "
                        "reasoner output by dominance")
    p.add_argument("--track", default="train", choices=["train", "real"],
                   help="train = synthetic transfer set; real = real timeline")

    g = p.add_argument_group("reasoner")
    g.add_argument("--ruleset", default=None,
                   help="the .vada program name, relative to the engine's repository "
                        "(default: derived from reasoner_repository in config.yaml)")
    g.add_argument("--endpoint", default=None,
                   help="override config reasoner_endpoint")
    g.add_argument("--n-batches", type=int, default=None,
                   help="only label the first N batches")
    g.add_argument("--batch-dir", default=None, help="where the batch CSVs are")
    g.add_argument("--results", default=None, help="where predicate CSVs go")
    g.add_argument("--continue-on-error", action="store_true",
                   help="keep going after a failed batch (default: stop)")

    g = p.add_argument_group("python / recalibrate")
    g.add_argument("--stride", type=int, default=None,
                   help="take every Nth window on the real timeline "
                        "(default: real_pairs.stride in config, 1)")
    g.add_argument("--deadband", type=float, default=None,
                   help="dominance margin in percentage points: score > +d -> "
                        "positive, < -d -> negative, else none. Default comes from "
                        "configs/config.yaml labels.deadband, which is also what the "
                        "reasoner receives as ${Deadband} — override here only to test "
                        "sensitivity, not to set the project value")
    g.add_argument("--date-from", default=None,
                   help="default: real_pairs.date_from in config (2015-01-01)")
    g.add_argument("--date-to", default=None,
                   help="default: real_pairs.date_to in config (2024-12-31)")
    g.add_argument("--seq-len", type=int, default=None,
                   help="window length (default: from the pair's frequency)")
    g.add_argument("--out-dir", default=None)

    args = p.parse_args(argv)

    if args.engine == "recalibrate" and args.track != "real":
        p.error("--engine recalibrate only applies to --track real")
    if args.engine == "python" and args.track == "train":
        p.error("--engine python labels the real timeline; the synthetic transfer set "
                "exists to be labelled by the reasoner (--engine vadalog)")

    label = {("train", "vadalog"): _label_train_vadalog,
             ("real", "vadalog"): _label_real_vadalog,
             ("real", "python"): _label_real_python,
             ("real", "recalibrate"): _label_recalibrate}[(args.track, args.engine)]

    with stage(f"label · {args.pair} · {args.track}/{args.engine}", args.quiet):
        r = label(args)

    if isinstance(r, dict) and "regime_counts" in r:
        print(f"{args.pair}: {r.get('windows', r.get('n_windows'))} windows "
              f"-> {r['regime_counts']}")
    elif isinstance(r, dict) and "done" in r:
        print(f"{args.pair}: {len(r['done'])} batches labelled, "
              f"{len(r['skipped'])} cached, {len(r['failed'])} failed")
    return r


if __name__ == "__main__":
    main()
