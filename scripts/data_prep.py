#!/usr/bin/env python3
"""
Stage 1 — data prep. Download, transform, window, fingerprint.

Examples
--------
    # everything (financial + macro), using cached raw CSVs where present
    python scripts/data_prep.py -d all

    # just the financial series
    python scripts/data_prep.py -d financial

    # add the earlier history (config data.financial_start) in front of the cached files
    python scripts/data_prep.py -d financial --extend

    # only two series, forcing a fresh download
    python scripts/data_prep.py -d financial --series SP500,Gold --force

    # macro, US only (skip Eurostat — no API key needed for FRED cache hits)
    python scripts/data_prep.py -d macro --skip-eu

    # align a different real pair for the classifier's input
    python scripts/data_prep.py -d financial --aligned-pair SP500,Gold

Writes to data/raw/ (downloads) and data/processed/ (transformed.pkl, windowed.pkl,
fingerprints.pkl, aligned_windows_real.pkl, and the _macro variants).
"""
from _cli import bootstrap, csv_list, parser, stage

bootstrap()


def main(argv=None):
    p = parser(__doc__.split("Examples")[0].strip(), epilog=__doc__)
    p.add_argument("-d", "--dataset", default="all",
                   choices=["financial", "macro", "all"],
                   help="which family to prepare (default: all)")
    p.add_argument("--series", type=csv_list, default=None,
                   help="restrict to these series, comma separated "
                        "(e.g. SP500,Bitcoin). Default: all in the catalog")
    p.add_argument("--force", action="store_true",
                   help="re-download instead of reading data/raw/*_raw.csv (replaces "
                        "the whole file, including the days the paper uses)")
    p.add_argument("--extend", action="store_true",
                   help="financial: add the days between config data.financial_start "
                        "and the start of each cached file; the rows already there are "
                        "kept byte for byte, and nothing changes if Yahoo's overlap differs")
    p.add_argument("--aligned-pair", type=csv_list, default=["SP500", "Bitcoin"],
                   help="pair to date-align into aligned_windows_real.pkl "
                        "(default: SP500,Bitcoin)")
    p.add_argument("--with-eu", action="store_true",
                   help="macro only: also download the Eurostat series (EU unemployment, "
                        "inflation, GDP). Off by default: no result of the paper uses them, "
                        "and they need the `eurostat` package and ~110 MB")
    p.add_argument("--skip-eu", action="store_true",
                   help="kept for old commands; the Eurostat series are skipped by default")
    p.add_argument("--countries", type=csv_list, default=None,
                   help="macro only: EU countries (default: DE,FR,IT)")
    p.add_argument("--skip-fingerprints", action="store_true",
                   help="skip the stylized-fact fingerprints (faster; nb_08 needs them)")
    args = p.parse_args(argv)

    verbose = not args.quiet
    results = {}

    if args.dataset in ("financial", "all"):
        from tslearning.datasets import ingest_financial
        with stage("data prep · financial", args.quiet):
            results["financial"] = ingest_financial.run(
                series=args.series, force=args.force, extend=args.extend,
                aligned_pair=tuple(args.aligned_pair),
                skip_fingerprints=args.skip_fingerprints, verbose=verbose)

    if args.dataset in ("macro", "all"):
        from tslearning.datasets import ingest_macro
        with stage("data prep · macro", args.quiet):
            results["macro"] = ingest_macro.run(
                series=args.series, force=args.force, countries=args.countries,
                skip_eu=not args.with_eu, skip_fingerprints=args.skip_fingerprints,
                verbose=verbose)

    for family, r in results.items():
        print(f"{family}: {len(r['windowed'])} series windowed -> "
              f"{', '.join(r['written'])}")
    return results


if __name__ == "__main__":
    main()
