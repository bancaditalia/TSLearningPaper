"""
Summary of rule verification (Python scorer vs Vadalog) for the pairs the paper reports.

Reads verification.csv and verification_classifier.csv of the current run, keeps only the
given pairs, and writes verification_summary.{md,csv}: per label source, how many rules
were verified, how many both engines can score, how many agree exactly, the largest
|Delta|, and the list of rules that differ. These are the numbers of the Verification
paragraph; nothing is recomputed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from tslearning.config import PATHS

SOURCES = (("exact labels", "verification.csv"),
           ("classifier labels", "verification_classifier.csv"))


def run(pairs, out_dir, results=None, suffix: str = "_filtered",
        verbose: bool = True) -> "Path | None":
    """Writes, in out_dir: verification<suffix>.csv and verification_classifier<suffix>.csv
    (the rows of the given pairs only) and verification_summary<suffix>.{md,csv}."""
    res = Path(results or PATHS["results"])
    rows, details = [], []
    for label, fname in SOURCES:
        f = res / fname
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[d["pair"].isin(pairs)]
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        d.to_csv(Path(out_dir) / fname.replace(".csv", f"{suffix}.csv"), index=False)
        both = d["python_confidence"].notna() & d["vadalog_confidence"].notna()
        delta = (d["python_confidence"] - d["vadalog_confidence"]).abs()
        equal = both & (delta < 1e-9)
        differ = d[both & ~equal]
        rows.append({"labels": label, "verified": len(d), "scored_by_both": int(both.sum()),
                     "identical": int(equal.sum()), "differ": len(differ),
                     "max_abs_delta": round(float(delta[both].max()), 4) if both.any() else None,
                     "source": fname.replace(".csv", f"{suffix}.csv")})
        for _, r in differ.iterrows():
            details.append(f"| {label} | {r['template']} | {r['pair']} | "
                           f"{r['python_confidence']:.4f} | {r['vadalog_confidence']:.4f} | "
                           f"{abs(r['python_confidence'] - r['vadalog_confidence']):.4f} |")
    if not rows:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"verification_summary{suffix}.csv", index=False)
    md = ["# Rule verification: Python scorer vs Vadalog", "",
          f"FILTERED to the pairs reported in the paper: {', '.join(pairs)}. "
          f"Rows from verification{suffix}.csv and verification_classifier{suffix}.csv "
          f"(the unfiltered files, all pairs of the run, are next to them).", "",
          "| labels | verified | scored by both | identical | differ | max |Δ| |",
          "|---|---|---|---|---|---|"]
    md += [f"| {r['labels']} | {r['verified']} | {r['scored_by_both']} | {r['identical']} | "
           f"{r['differ']} | {r['max_abs_delta']} |" for r in rows]
    if details:
        md += ["", "Rules that differ:", "",
               "| labels | template | pair | Python | Vadalog | |Δ| |", "|---|---|---|---|---|---|"]
        md += details
    (out_dir / f"verification_summary{suffix}.md").write_text("\n".join(md) + "\n")
    if verbose:
        print("\n".join(md[:2 + 2 + len(rows) + 2]))
    return out_dir / f"verification_summary{suffix}.md"


def agreement(pairs, out_dir, results=None, suffix: str = "_filtered") -> "Path | None":
    """Label agreement (Python vs reasoner, `eval.py --metric agreement`) for the given pairs: one row
    per pair from agreement_<side>.csv, plus the total. Pairs without a file are listed
    as missing. Writes agreement_summary<suffix>.{md,csv} in out_dir."""
    from tslearning.datasets.pairs import describe

    res = Path(results or PATHS["results"])
    rows, missing = [], []
    for p in pairs:
        f = res / f"agreement_{describe(p)['b'].lower()}.csv"
        if not f.exists():
            missing.append(p)
            continue
        r = pd.read_csv(f).iloc[0]
        row = {"pair": p, "windows": int(r["n_scored"]), "regime_agree": int(r["n_agree"])}
        for pred in ("highVol", "lowVol", "tailDependent"):
            if f"{pred}_agree" in r:
                row[f"{pred}_agree"] = f"{int(r[f'{pred}_agree'])}/{int(r[f'{pred}_windows'])}"
        rows.append(row)
    if not rows:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"agreement_summary{suffix}.csv", index=False)
    cols = list(df.columns)
    md = ["# Label agreement: Python labeler vs reasoner, real test windows", "",
          f"FILTERED to the pairs reported in the paper: {', '.join(pairs)}."
          + (f" No agreement file for: {', '.join(missing)}." if missing else ""), "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(v) for v in r.values()) + " |" for r in rows]
    md += ["", f"Correlation regime, total: {int(df.regime_agree.sum())} of "
               f"{int(df.windows.sum())} windows."]
    (out_dir / f"agreement_summary{suffix}.md").write_text("\n".join(md) + "\n")
    return out_dir / f"agreement_summary{suffix}.md"
