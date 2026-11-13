#!/usr/bin/env python3
"""
Build a REALISTIC synthetic transfer set — correlations concentrated near
the decision threshold and real-like tails, still fully synthetic (no real pair data) —
and label it in Python in the reasoner's CSV format.

    python scripts/build_realistic_transfer.py -p sp500_gold
    python scripts/build_realistic_transfer.py -p sp500_bitcoin,sp500_gold,sp500_nasdaq --strength boundary

Writes data/processed/transfer_<pair>_boundary/ (default --strength boundary; _even/ and
_realistic/ for the other two modes), refused if not empty:
  engineered_<a>.npy / engineered_<b>.npy   the windows
  batch_NNN.csv                             reasoner input (for an engine spot-check)
  {regime}_NNN.csv, same%/diff% CSVs        Python labels, reasoner format
  engineered_manifest.csv                   per pair: target class, c, score, injections
  comparison.csv                            real 2015-2021 vs this set vs the current set
  generator.json                            settings + label counts

Train on it with `--transfer-suffix _boundary` (distill.py, distill_v8.py).
The original generator (build_transfer_set.py) is unchanged.
"""
from _cli import bootstrap, csv_list, parser

bootstrap()


def main(argv=None):
    p = parser(__doc__.strip().split("\n")[0], epilog=__doc__)
    p.add_argument("-p", "--pairs", type=csv_list, default=["sp500_gold"])
    p.add_argument("--n-pairs", type=int, default=525)
    p.add_argument("--strength", default="boundary", choices=["boundary", "even", "matched"],
                   help="boundary (default, data-free): most pairs just past the threshold, "
                        "fewer further out -> transfer_<pair>_boundary/. even (data-free, "
                        "rejected): spread evenly, mostly strong -> _even/. matched: copy real "
                        "2015-2021 pooled scores (uses real pair statistics, superseded) -> "
                        "_realistic/")
    p.add_argument("--extreme-frac", type=float, default=0.2,
                   help="matched mode only: share of positive/negative pairs that are "
                        "strong, |score| > 40 (default 0.2)")
    p.add_argument("--seed", type=int, default=None, help="default: config seed")
    p.add_argument("--out-dir", default=None, help="only with a single pair")
    args = p.parse_args(argv)
    if args.out_dir and len(args.pairs) > 1:
        p.error("--out-dir needs a single pair")

    from tslearning.labeling import transfer_realistic as T
    from tslearning.config import CFG
    ref = T.real_reference(seq_len=CFG["windows"]["financial"]["length"])  # real train-period windows
    out = {}
    for pair in args.pairs:
        out[pair] = T.run(pair, n_pairs=args.n_pairs, extreme_frac=args.extreme_frac,
                          out_dir=args.out_dir, seed=args.seed, ref=ref,
                          strength=args.strength,
                          verbose=not args.quiet)
    return out


if __name__ == "__main__":
    main()
