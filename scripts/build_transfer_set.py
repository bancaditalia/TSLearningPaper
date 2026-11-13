#!/usr/bin/env python3
"""
Stage 3 — transfer set. Build the window pairs the reasoner will label.

 The
notebooks are unchanged.

Two generators, and the choice matters:

  --mode engineered  (default) controlled positive / negative / no-correlation
        shapes, with volatility, trend and co-crash events injected on top, so
        highVol / trendDown / tailDependent actually fire. Writes
        engineered_manifest.csv, the INJECTED ground truth, for a known-answer
        check. This is nb_10-alt.
  --mode plain  independent DDPM samples paired up. This is nb_10-clean. Honest as
        a data-free-KD transfer set, but almost every pair comes out
        "no correlation" — it is what collapsed V1-V5.

Examples
--------
    python scripts/build_transfer_set.py -p sp500_bitcoin
    python scripts/build_transfer_set.py -p sp500_bitcoin --mode plain
    python scripts/build_transfer_set.py -p sp500_gold --n-pairs 250 --batch-size 25
    python scripts/build_transfer_set.py -p sp500_bitcoin --no-fit-thresholds

Writes batch_NNN.csv (the reasoner's input, disjoint-block layout),
engineered_<side>.npy, engineered_manifest.csv, and — unless
--no-fit-thresholds — thresholds/tau.csv + thresholds/qlow.csv, fitted on the
config's TRAIN period only.

Next step: `label.py -p <pair>`.
"""
from _cli import bootstrap, parser, stage

bootstrap()


def main(argv=None):
    p = parser(__doc__.split("Examples")[0].strip(), epilog=__doc__)
    p.add_argument("-p", "--pair", default="sp500_bitcoin",
                   help="pair as <a>_<b>, lowercase series names "
                        "(default: sp500_bitcoin)")
    p.add_argument("--mode", default="engineered", choices=["engineered", "plain"],
                   help="pair generator (default: engineered)")
    p.add_argument("--n-pairs", type=int, default=None,
                   help="how many pairs to build (default: 525, i.e. 21 batches)")
    p.add_argument("--batch-size", type=int, default=25,
                   help="pairs per reasoner batch (default: 25)")
    p.add_argument("--no-fit-thresholds", action="store_true",
                   help="reuse the existing thresholds/tau.csv + qlow.csv")
    p.add_argument("--out-dir", default=None,
                   help="where the batch CSVs, windows and manifest go "
                        "(default: data/processed/transfer_<pair>/)")
    args = p.parse_args(argv)

    from tslearning.labeling import transfer_set
    with stage(f"transfer set · {args.pair} ({args.mode})", args.quiet):
        r = transfer_set.run(pair=args.pair, mode=args.mode, n_pairs=args.n_pairs,
                             batch_size=args.batch_size,
                             fit_thr=not args.no_fit_thresholds,
                             out_dir=args.out_dir, verbose=not args.quiet)

    print(f"{r['pair']}: {r['n_pairs']} pairs -> {r['n_batches']} batches "
          f"of {r['batch_size']} in {r['out_dir']}")
    return r


if __name__ == "__main__":
    main()
