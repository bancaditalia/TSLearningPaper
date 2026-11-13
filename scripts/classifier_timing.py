"""
Classifier timing table, on the SAME window-pairs the reasoner was timed on.

For every data/processed/<folder>/timing.json (the reasoner runs), the batches listed in
it are mapped back to their input CSVs:

    transfer_*   key "000"            -> <folder>/batch_000.csv
    _timing      key "sp500_gold_000" -> real_pairs_batches/sp500_gold_batch_000.csv

The classifier then labels all of that folder's window-pairs in one forward pass,
`--repeats` times after a warm-up pass; the table reports the mean. The ALL row is the
window-pair-weighted mean of the folders. The weights file is recorded in every row.

    python scripts/classifier_timing.py
    python scripts/classifier_timing.py --weights data/processed/classifier_v7b_dominance.pt

Writes results/classifier_timing.csv, which scripts/timing_table.py uses for the speedup.
"""
import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import bootstrap  # noqa: E402

bootstrap()  # repo on sys.path + --run
from tslearning.config import PATHS  # noqa: E402
from tslearning.datasets.data_utils import zscore_window  # noqa: E402
from tslearning.distill.classifier import load_classifier  # noqa: E402


def _series(name: str) -> str:
    for sep in ("_synth_", "_real_"):
        if sep in name:
            return name.split(sep)[0]
    return name


def batch_csv(folder: Path, key: str) -> Path:
    """Input CSV the reasoner was given for this timing.json batch key.

    Looked for where each kind of run writes its batches (first existing wins):
      transfer sets         key "000"           -> <folder>/batch_000.csv
      bitcoin fidelity set  key "0"             -> real_test_batch_000.csv
      fidelity sets         key "sp500_tlt_000" -> <folder>_batches/sp500_tlt_batch_000.csv
      timing run            key "sp500_gold_000"-> real_pairs_batches/sp500_gold_batch_000.csv
    (Before 2026-10-28 only the first and last were tried, so the fidelity folders were
    skipped, and real_test_sp500_gold was timed on the timing run's batches.)"""
    dp = PATHS["data_processed"]
    if key.isdigit():
        cands = [folder / f"batch_{key}.csv", folder / f"batch_{int(key):03d}.csv",
                 dp / f"real_test_batch_{int(key):03d}.csv"]
    else:
        pair, _, idx = key.rpartition("_")
        cands = [dp / f"{folder.name}_batches" / f"{pair}_batch_{idx}.csv",
                 dp / "real_pairs_batches" / f"{pair}_batch_{idx}.csv"]
    return next((c for c in cands if c.exists()), cands[0])


def window_pairs(path: Path) -> np.ndarray:
    """(N, 2, L) classifier input from one batch CSV: the i-th window of the first
    series paired with the i-th window of the second, each z-scored."""
    df = pd.read_csv(path)
    wins = {n: g["value"].to_numpy(float) for n, g in df.groupby("window_name", sort=False)}
    series = list(dict.fromkeys(_series(n) for n in wins))
    if len(series) != 2:
        raise ValueError(f"{path}: expected 2 series, found {series}")
    a = [w for n, w in wins.items() if _series(n) == series[0]]
    b = [w for n, w in wins.items() if _series(n) == series[1]]
    return np.stack([np.stack([zscore_window(x), zscore_window(y)])
                     for x, y in zip(a, b)]).astype(np.float32)


def time_model(model, X: np.ndarray, repeats: int) -> float:
    """Mean seconds per window-pair over `repeats` full passes, after one warm-up pass."""
    import torch
    Xt = torch.from_numpy(X)
    with torch.no_grad():
        model(Xt)                                   # warm-up
        runs = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(Xt)
            runs.append(time.perf_counter() - t0)
    return st.mean(runs) / len(X)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(PATHS["data_processed"]))
    ap.add_argument("--exclude", default="_warmup",
                    help="comma-separated folder names to skip (default: _warmup)")
    ap.add_argument("--weights", default=None,
                    help="classifier weights (default: classifier.weights in configs/config.yaml)")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--save", default=str(PATHS["results"] / "classifier_timing.csv"))
    args = ap.parse_args(argv)

    # resolve the weights actually loaded, so the CSV records them (default included)
    from tslearning.distill.classifier import default_weights
    weights = Path(args.weights) if args.weights else default_weights()
    model = load_classifier(weights)
    exclude = {e.strip() for e in args.exclude.split(",") if e.strip()}

    rows = []
    for f in sorted(Path(args.root).glob("*/timing.json")):
        folder = f.parent
        if folder.name in exclude:
            continue
        # *_dominance folders are relabelled COPIES of a transfer set (relabel.py): any
        # timing.json there is the original run's, so counting it would double it
        if folder.name.endswith("_dominance"):
            continue
        keys = list(json.loads(f.read_text()).get("batches", {}))
        if not keys:
            print(f"skipped {folder.name}: no timed batches")
            continue
        missing = [k for k in keys if not batch_csv(folder, k).exists()]
        if missing:
            print(f"skipped {folder.name}: input CSVs missing for batches {missing}")
            continue
        X = np.concatenate([window_pairs(batch_csv(folder, k)) for k in keys])
        rows.append({"folder": folder.name, "window_pairs": len(X),
                     "s_per_window_pair": time_model(model, X, args.repeats)})

    if not rows:
        sys.exit("nothing to time")

    # ALL = window-pair-weighted mean of the per-folder times, NOT one pass over everything:
    # a single large batch is cheaper per pair, which would inflate the overall speedup.
    n = sum(r["window_pairs"] for r in rows)
    rows.append({"folder": "ALL", "window_pairs": n,
                 "s_per_window_pair": sum(r["s_per_window_pair"] * r["window_pairs"]
                                          for r in rows) / n})
    table = pd.DataFrame(rows)
    table["weights"] = weights.name
    table["repeats"] = args.repeats
    table.to_csv(args.save, index=False)
    print(table.drop(columns=["weights", "repeats"]).to_markdown(index=False, floatfmt=".3g"))
    print(f"\nweights: {weights}  repeats: {args.repeats}  ->  {args.save}")


if __name__ == "__main__":
    main()
