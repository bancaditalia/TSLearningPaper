"""
Reasoner timing table, computed from every timing.json under data/processed/.

Each timing.json is written by reasoner_client._save_timing: {"batches": {name: seconds}}.
One batch = `--batch-size` window-pairs (25 by default, as in transfer_set.write_batches).

    python scripts/classifier_timing.py --weights <file>    # first: the classifier side
    python scripts/timing_table.py
    python scripts/timing_table.py --exclude _warmup,_timing   # transfer sets only

The speedup of each folder divides its reasoner time by the classifier time measured on
THAT folder's window-pairs (results/classifier_timing.csv). The ALL row uses the
window-pair-weighted mean of the included folders on both sides, so --exclude applies to
the reasoner and the classifier alike.

_warmup is excluded by default (REPRODUCIBILITY.md §2: the warm-up batch is discarded).
Folders whose timing.json has no batches (everything cached) are skipped.
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import bootstrap  # noqa: E402

bootstrap()  # repo on sys.path + --run
from tslearning.config import PATHS  # noqa: E402


def load_classifier_timing(path: Path) -> "tuple[dict, str | None]":
    """{folder: classifier s/window-pair} and the weights file name, from classifier_timing.py."""
    if not path.exists():
        return {}, None
    df = pd.read_csv(path)
    df = df[df["folder"] != "ALL"]
    weights = df["weights"].iloc[0] if "weights" in df.columns and len(df) else None
    return dict(zip(df["folder"], df["s_per_window_pair"])), weights


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(PATHS["data_processed"]))
    ap.add_argument("--exclude", default="_warmup",
                    help="comma-separated folder names to skip (default: _warmup)")
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--classifier-seconds", type=float, default=None,
                    help="one classifier s/window-pair for every folder, instead of the "
                         "per-folder values in results/classifier_timing.csv")
    ap.add_argument("--save", default=str(PATHS["results"] / "reasoner_timing.csv"))
    args = ap.parse_args(argv)

    root = Path(args.root)
    exclude = {e.strip() for e in args.exclude.split(",") if e.strip()}
    clf, weights = load_classifier_timing(PATHS["results"] / "classifier_timing.csv")
    if args.classifier_seconds is None and not clf:
        print("no results/classifier_timing.csv — run scripts/classifier_timing.py first "
              "(or pass --classifier-seconds); speedup column omitted")

    rows, all_batches = [], []
    for f in sorted(root.glob("*/timing.json")):
        name = f.parent.name
        if name in exclude:
            continue
        # *_dominance folders are relabelled COPIES of a transfer set (relabel.py): any
        # timing.json there is the original run's, so counting it would double it
        if name.endswith("_dominance"):
            continue
        batches = list(json.loads(f.read_text()).get("batches", {}).values())
        if not batches:
            print(f"skipped {name}: no timed batches")
            continue
        all_batches += batches
        c = args.classifier_seconds if args.classifier_seconds is not None else clf.get(name)
        if c is None and clf:
            print(f"no classifier timing for {name} — rerun classifier_timing.py "
                  f"with the same --exclude")
        rows.append({"folder": name, "batches": len(batches),
                     "window_pairs": len(batches) * args.batch_size,
                     "s_per_window_pair": st.mean(batches) / args.batch_size,
                     "classifier_s_per_window_pair": c})

    if not rows:
        sys.exit("no timing.json with batches found")

    # ALL: weighted mean over the included folders, on both sides
    n = sum(r["window_pairs"] for r in rows)
    timed = [r for r in rows if r["classifier_s_per_window_pair"] is not None]
    all_clf = (sum(r["classifier_s_per_window_pair"] * r["window_pairs"] for r in timed)
               / sum(r["window_pairs"] for r in timed)) if len(timed) == len(rows) else None
    rows.append({"folder": "ALL", "batches": len(all_batches), "window_pairs": n,
                 "s_per_window_pair": st.mean(all_batches) / args.batch_size,
                 "classifier_s_per_window_pair": all_clf})

    table = pd.DataFrame(rows)
    has_clf = table["classifier_s_per_window_pair"].notna()
    table["speedup"] = pd.NA
    table.loc[has_clf, "speedup"] = (table.loc[has_clf, "s_per_window_pair"]
                                     / table.loc[has_clf, "classifier_s_per_window_pair"]
                                     ).round(0).astype(int)
    table["classifier_weights"] = ("--classifier-seconds" if args.classifier_seconds
                                   is not None else weights)

    table.to_csv(args.save, index=False)
    show = table.drop(columns=["classifier_weights"])
    print(show.to_markdown(index=False, floatfmt=".3g"))
    if args.classifier_seconds is None and clf:
        print(f"\nclassifier weights: {weights or 'unknown (old classifier_timing.csv — rerun it)'}")
    print(f"excluded: {', '.join(sorted(exclude)) or 'none'}  ->  {args.save}")


if __name__ == "__main__":
    main()
