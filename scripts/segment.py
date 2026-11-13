#!/usr/bin/env python3
"""
Stage 7 — segmentation. Per-window facts -> smoothed regime INTERVALS.

py yet.

The problem: windows overlap (length 50, step 1), so ~50 windows cover every day and
they routinely disagree. Something must reduce that to one regime per day, then to
intervals.

  --smoothing hmm       (config default) Viterbi decoding with ONE interpretable
        parameter: stickiness p, where 1/(1-p) is the expected regime duration in
        days (p=0.98 -> ~50 days ~= one window). Isolated one-day contradictions get
        overridden; a sustained change still flips the regime. Yields exactly one
        regime per day, so the old "exclusive vs everything" conflict-resolution
        step disappears entirely.
  --smoothing majority  the rolling majority vote nb_11 currently uses. Two
        arbitrary knobs and the output moves when they move — kept because the paper
        wants it as an ablation row.
  --smoothing none      raw argmax. The ablation floor.

Smoothing only applies when there are per-timepoint probabilities to decode, i.e.
when reading a classifier bundle from temporal/. The python-labeler and recalibrated-
reasoner outputs are already one label per window, so they are coalesced as they are.

Examples
--------
    python scripts/segment.py -p sp500_bitcoin
    python scripts/segment.py -p sp500_bitcoin --smoothing majority --majority-window 30
    python scripts/segment.py -p sp500_bitcoin --smoothing hmm --stickiness 0.99
    python scripts/segment.py -p sp500_gold --source data/processed/real_pairs_regime_py/sp500_gold

Writes data/processed/intervals/<pair>_intervals.csv — predicate, key, [start, end),
length, and calendar start_date/end_date when a {pair}_dates.csv is available.
Intervals must be REPORTED in calendar days, not tick indices.

Next step: nb_11's rule-mining cells, pointed at that intervals CSV.
"""
from _cli import bootstrap, parser, stage

bootstrap()


def main(argv=None):
    p = parser(__doc__.split("Examples")[0].strip(), epilog=__doc__)
    p.add_argument("-p", "--pair", default="sp500_bitcoin",
                   help="pair as <a>_<b> (default: sp500_bitcoin)")
    p.add_argument("--source", default=None,
                   help="folder holding the predicate CSVs. Default: the first of "
                        "real_pairs_regime_py/<pair>, real_pairs_regime_reasoner/"
                        "<pair>, temporal/ that exists")
    p.add_argument("--smoothing", default=None, choices=["hmm", "majority", "none"],
                   help="default: config segmentation.method")
    p.add_argument("--majority-window", type=int, default=50,
                   help="timepoints in the majority vote (default: 50)")
    p.add_argument("--stickiness", type=float, default=None,
                   help="HMM stay probability p; 1/(1-p) is the expected regime "
                        "duration in days (default: config hmm_stickiness)")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--out-dir", default=None,
                   help="where the intervals CSV goes (default: data/processed/intervals)")
    args = p.parse_args(argv)

    from tslearning.segment import predicate_io
    with stage(f"segment · {args.pair}", args.quiet):
        r = predicate_io.run(pair=args.pair, source=args.source,
                             smoothing=args.smoothing,
                             majority_window=args.majority_window,
                             stickiness=args.stickiness, out_dir=args.out_dir,
                             seq_len=args.seq_len, verbose=not args.quiet)

    n_pred = sum(1 for ivs in r["predicate_intervals"].values() if ivs)
    print(f"{args.pair}: {len(r['frame'])} intervals across {n_pred} predicates "
          f"-> {r['out_path']}")
    return r


if __name__ == "__main__":
    main()
