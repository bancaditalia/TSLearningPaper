#!/usr/bin/env python3
"""
Spot-check: do the Python labels of a synthetic transfer set equal the reasoner's?

    python scripts/label.py -p sp500_gold --track train --engine vadalog --n-batches 1 \
        --batch-dir data/processed/transfer_sp500_gold_boundary \
        --results data/processed/enginecheck_sp500_gold_boundary
    python scripts/check_engine_labels.py \
        --python data/processed/transfer_sp500_gold_boundary \
        --engine data/processed/enginecheck_sp500_gold_boundary

Compares, window by window, the regime and the same% / diff% that the engine wrote with
the ones written in Python (only for the batches the engine ran). A missing percentage
counts as 0 on both sides; the regime is re-derived from the engine's percentages by the
same dominance rule (the engine's own regime rule leaves a window unlabelled when one
percentage is 0). Writes <engine dir>/engine_check.json.
"""
import glob
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _cli import bootstrap, parser  # noqa: E402

bootstrap()

SAME = "magnitudeSameSignConsistencyPercentage"
DIFF = "magnitudeDiffSignConsistencyPercentage"


def _pct(folder: Path, pred: str, tags) -> dict:
    import pandas as pd
    out = {}
    for t in tags:
        p = folder / f"{pred}_{t}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        for x, y, v in zip(d["X"], d["Y"], d["Pct"]):
            if x != y:
                out[frozenset((x, y))] = float(v)
    return out


def main(argv=None):
    p = parser(__doc__.strip().split("\n")[0], epilog=__doc__)
    p.add_argument("--python", required=True, help="transfer set with the Python labels")
    p.add_argument("--engine", required=True, help="folder the engine wrote")
    args = p.parse_args(argv)

    from tslearning.labeling.py_labeler import default_deadband
    py, en = Path(args.python), Path(args.engine)
    tags = sorted({re.match(rf"{SAME}_(\d+)\.csv", Path(f).name).group(1)
                   for f in glob.glob(str(en / f"{SAME}_*.csv"))
                   if re.match(rf"{SAME}_(\d+)\.csv", Path(f).name)})
    if not tags:
        p.error(f"no {SAME}_NNN.csv in {en} — did the engine run?")
    windows = set()
    for t in tags:
        b = py / f"batch_{t}.csv"
        if b.exists():
            import pandas as pd
            names = pd.read_csv(b)["window_name"].unique()
            a = sorted(n for n in names if n.startswith("sp500"))
            o = sorted(n for n in names if not n.startswith("sp500"))
            idx = lambda n: int(n.rsplit("_", 1)[1])
            oi = {idx(n): n for n in o}
            windows |= {frozenset((n, oi[idx(n)])) for n in a if idx(n) in oi}
    db = default_deadband()
    ps, pdf = _pct(py, SAME, tags), _pct(py, DIFF, tags)
    es, edf = _pct(en, SAME, tags), _pct(en, DIFF, tags)

    def reg(s, d):
        sc = s - d
        return "pos" if sc > db else "neg" if sc < -db else "none"

    same_ok = diff_ok = reg_ok = 0
    bad = []
    for w in sorted(windows, key=lambda k: sorted(k)):
        a, b = ps.get(w, 0.0), es.get(w, 0.0)
        c, d = pdf.get(w, 0.0), edf.get(w, 0.0)
        same_ok += abs(a - b) < 1e-6
        diff_ok += abs(c - d) < 1e-6
        r_ok = reg(a, c) == reg(b, d)
        reg_ok += r_ok
        if not (abs(a - b) < 1e-6 and abs(c - d) < 1e-6):
            bad.append({"window": sorted(w), "python": [a, c], "engine": [b, d]})
    n = len(windows)
    res = {"batches": tags, "windows": n, "same_pct_equal": same_ok,
           "diff_pct_equal": diff_ok, "regime_equal": reg_ok, "mismatches": bad[:20]}
    (en / "engine_check.json").write_text(json.dumps(res, indent=2))
    print(f"{n} windows in batch(es) {', '.join(tags)}: same% equal {same_ok}/{n}, "
          f"diff% equal {diff_ok}/{n}, regime equal {reg_ok}/{n} -> {en / 'engine_check.json'}")
    return res


if __name__ == "__main__":
    main()
