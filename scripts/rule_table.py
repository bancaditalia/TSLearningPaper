#!/usr/bin/env python3
"""
Emit the two paper tables from the rule registry and the result CSVs.

  Table 1  results/rule_library.{csv,md,tex}   one row per template — the method section
  Table 2  results/rule_results.{csv,md,tex}   one row per template x pair, rejections
                                               included — the results section

Reads `results/rules.csv` (from generate_rules.py --score) and, if present,
`results/verification.csv` (from verify_rules.py) for the |delta| column. Nothing is
recomputed here, so the tables always match the numbers those two stages produced.

Examples
--------
  # everything, markdown + LaTeX
  python scripts/rule_table.py

  # just the three pairs in the paper
  python scripts/rule_table.py -p sp500_gold,sp500_bitcoin,sp500_nasdaq

  # the T1 exhibit alone, LaTeX only
  python scripts/rule_table.py -t T1_safeHaven --format tex
"""
from _cli import bootstrap, csv_list, parser, stage

bootstrap()


def main() -> None:
    p = parser(__doc__.strip().split("\n")[0], epilog=__doc__)
    p.add_argument("-p", "--pairs", type=csv_list, default=None,
                   help="restrict to these pairs (default: every pair in rules.csv)")
    p.add_argument("-t", "--templates", type=csv_list, default=None,
                   help="restrict to these templates, e.g. T1_safeHaven")
    p.add_argument("--period", default="test", choices=["train", "test"],
                   help="which scored period the results table reports (default: test)")
    p.add_argument("--format", default="both", choices=["md", "tex", "both"],
                   help="which renderings to write alongside the CSV")
    p.add_argument("--scored-only", action="store_true",
                   help="drop rules that could not be fitted or were ineligible. Off by "
                        "default: a template that cannot fire on a pair is a result "
                        "(sp500-nasdaq has no negative-correlation days at all, which is "
                        "the control). Scored rows are listed first either way")
    p.add_argument("-o", "--out-dir", default=None,
                   help="where the tables are written (default: results/)")
    p.add_argument("--in-dir", default=None,
                   help="where rules*.csv, verification*.csv and the llm_*.json runs are "
                        "read from (default: results/)")
    p.add_argument("--tag", default=None,
                   help="label source tag. Reads rules_<tag>.csv + verification_<tag>.csv "
                        "and writes rule_results_<tag>.*, so e.g. --tag classifier sits "
                        "beside the default (exact-labeller) tables instead of replacing them")
    p.add_argument("--llm-run", default=None,
                   help="select which LLM run(s) to include in the comparison table. "
                        "Matches as a substring of the filename, e.g. a timestamp like "
                        "20260922_143012 or 'run_01' for legacy files. Default: latest "
                        "run per (pair, provider, model)")
    args = p.parse_args()

    with stage(f"rule tables{' · ' + args.tag if args.tag else ''}", quiet=args.quiet):
        from tslearning.rules import tables
        r = tables.run(out_dir=args.out_dir, pairs=args.pairs,
                       templates=args.templates, fmt=args.format,
                       period=args.period, scored_only=args.scored_only,
                       tag=args.tag, llm_run=args.llm_run, in_dir=args.in_dir,
                       verbose=not args.quiet)

    print(f"{len(r['library'])} templates, {len(r['results'])} result rows -> "
          f"{len(r['written'])} files in "
          f"{r['written'][0].parent if r['written'] else '?'}")


if __name__ == "__main__":
    main()
