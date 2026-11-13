"""
Multiple testing: Bonferroni check on the rules scored on the test period.

    python scripts/multiple_testing.py                       # py labels, paper.pairs
    python scripts/multiple_testing.py --source classifier

Reads the same bootstrap as the results table (same seed), once at 95% and once at the
Bonferroni level 1 - 0.05/m (m = number of scored rules).
Writes results/multiple_testing[_<source>].{csv,md}. No engine needed; seconds.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _cli import bootstrap, csv_list, parser  # noqa: E402

bootstrap()


def main(argv=None):
    p = parser(__doc__.strip().split("\n")[0], epilog=__doc__)
    p.add_argument("--source", default="py", choices=["py", "reasoner", "classifier"])
    p.add_argument("-p", "--pairs", type=csv_list, default=None,
                   help="default: paper.pairs in configs/config.yaml")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("-o", "--out-dir", default=None)
    args = p.parse_args(argv)

    from tslearning.eval import multiple_testing as mt
    return mt.run(args.source, args.pairs, args.alpha, args.out_dir,
                  verbose=not args.quiet)


if __name__ == "__main__":
    main()
