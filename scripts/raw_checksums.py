#!/usr/bin/env python3
"""
Fingerprints of the raw series the paper used, so a fresh download can be checked
against them without redistributing the data.

    python scripts/raw_checksums.py --check     # compare data/raw/ with the paper's data
    python scripts/raw_checksums.py --record    # (authors) write data/raw_checksums.json

For each series: the download date, the date range, and one hash per calendar year of the
closing values (rounded to 6 decimals, so formatting differences do not matter). --check
reports, per series, whether the download matches and, if not, from which year it differs.
Known case: the Yahoo Finance gold futures series (GC=F) was rebuilt by the provider from
2020 onwards after the paper's download; see REPRODUCIBILITY.md.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "raw_checksums.json"

# series used by the paper's results, and the download date of the paper's copy
SERIES = {
    "SP500":        ("SP500_raw.csv",        "yahoo", "2026-07-10 (2002-2004 added 2026-10-20)"),
    "Gold":         ("Gold_raw.csv",         "yahoo", "2026-07-10"),
    "Bitcoin":      ("Bitcoin_raw.csv",      "yahoo", "2026-07-10"),
    "TLT":          ("TLT_raw.csv",          "yahoo", "2026-10-20"),
    "UST10Y":       ("UST10Y_raw.csv",       "yahoo", "2026-10-20"),
    "US_Inflation": ("US_Inflation_raw.csv", "fred",  "2026-07-30"),
}


def _series(fname: str, kind: str) -> pd.Series:
    f = RAW / fname
    if kind == "yahoo":                       # yfinance CSV: two extra header rows
        d = pd.read_csv(f, skiprows=[1, 2], index_col=0, parse_dates=True)
        s = d["Close"]
    else:                                     # FRED: date,value
        d = pd.read_csv(f, index_col=0, parse_dates=True)
        s = d.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce").dropna().sort_index()


def fingerprint(s: pd.Series) -> dict:
    years = {}
    for y, part in s.groupby(s.index.year):
        text = "\n".join(f"{d:%Y-%m-%d},{v:.6f}" for d, v in part.items())
        years[str(y)] = hashlib.sha256(text.encode()).hexdigest()[:16]
    return {"first": f"{s.index[0]:%Y-%m-%d}", "last": f"{s.index[-1]:%Y-%m-%d}",
            "rows": len(s), "years": years}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="compare data/raw/ with the record")
    g.add_argument("--record", action="store_true", help="write the record (authors only)")
    a = ap.parse_args(argv)

    if a.record:
        rec = {}
        for name, (fname, kind, date) in SERIES.items():
            rec[name] = {"file": fname, "source": kind, "downloaded": date,
                         **fingerprint(_series(fname, kind))}
        gz = RAW / "feds200628.csv.gz"
        if gz.exists():
            rec["GSW (feds200628)"] = {"file": gz.name, "included": True,
                                       "sha256": hashlib.sha256(gz.read_bytes()).hexdigest()}
        OUT.write_text(json.dumps(rec, indent=2))
        print(f"wrote {OUT.relative_to(ROOT)} ({len(rec)} series)")
        return

    rec = json.loads(OUT.read_text())
    for name, (fname, kind, _date) in SERIES.items():
        r = rec.get(name)
        if r is None:
            continue
        if not (RAW / fname).exists():
            print(f"{name:13s} missing: run scripts/data_prep.py first")
            continue
        now = fingerprint(_series(fname, kind))
        bad = sorted(y for y, h in r["years"].items() if now["years"].get(y) != h)
        if not bad:
            print(f"{name:13s} identical to the paper's data ({r['first']}..{r['last']})")
        else:
            print(f"{name:13s} DIFFERS from the paper's data in {len(bad)} year(s), from {bad[0]} "
                  f"(paper's copy downloaded {r['downloaded']})")


if __name__ == "__main__":
    main()
