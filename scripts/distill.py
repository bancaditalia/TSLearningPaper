#!/usr/bin/env python3
"""
Stage 5 — distillation. Train the fast student to imitate the slow reasoner.

Earlier classifier versions (V1-V6) are the paper's negative ablation, not part of this code.

The student predicts ONLY the correlation regime. Trend / volatility / tail come
from labeling/exact_flags.py, computed exactly in Python — per-window
standardization (which is what makes the student transfer across series) deletes the
scale information those flags need, so asking the student for them is asking for
something it cannot see. That is a design decision, not a limitation to fix.

What this adds over the notebook: a seeded, stratified train/test split, and
best-held-out checkpointing instead of last-epoch. Every accuracy the notebook
prints is training fit; the number here is held out.

Examples
--------
    python scripts/distill.py -p sp500_bitcoin
    python scripts/distill.py -p sp500_bitcoin --epochs 3000 --lr 5e-4
    python scripts/distill.py --resume data/processed/classifier_v7_corr.pt --epochs 500
    python scripts/distill.py --save data/processed/classifier_v8.pt --test-size 0.3

Reads windows and reasoner labels from data/processed/transfer_<pair>/. v7's original
training set is the flat layout: --results data/processed/batched_results, with windows
from data/processed/engineered_<side>.npy. Writes --save (an existing file is refused).

Next step: `infer.py -p <pair>` then `eval.py --metric fidelity`.
"""
from _cli import bootstrap, parser, stage

bootstrap()


def main(argv=None):
    p = parser(__doc__.split("Examples")[0].strip(), epilog=__doc__)
    p.add_argument("-p", "--pair", default="sp500_bitcoin",
                   help="pair as <a>_<b> (default: sp500_bitcoin). "
                        "Ignored when --pairs is given.")
    p.add_argument("--pairs", default=None,
                   help="comma-separated pair specs for multi-pair training, "
                        "e.g. sp500_bitcoin,sp500_gold,sp500_nasdaq. Each pair's "
                        "synthetic data is loaded from data/processed/transfer_<slug>/. "
                        "Build each with: build_transfer_set.py -p <pair> "
                        "--out-dir data/processed/transfer_<slug>, then label with: "
                        "label.py -p <pair> --batch-dir ... --results ...")
    p.add_argument("--results", default=None,
                   help="folder with the reasoner's predicate CSVs "
                        "(default: data/processed/transfer_<pair>/, windows and labels "
                        "together). Give data/processed/batched_results for v7's set. "
                        "Only used in single-pair mode.")
    p.add_argument("--epochs", type=int, default=1500,
                   help="full-batch gradient steps (default: 1500)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--test-size", type=float, default=0.2,
                   help="held-out fraction, stratified on the regime (default: 0.2)")
    p.add_argument("--save", default=None,
                   help="where to write the weights "
                        "(default: data/processed/classifier_v7_dominance.pt). "
                        "An existing file is never replaced unless --force is given")
    p.add_argument("--force", action="store_true",
                   help="allow --save to replace an existing weights file "
                        "(default: refuse, before any training is done)")
    p.add_argument("--resume", default=None,
                   help="continue training from these weights")
    p.add_argument("--class-weights", default=None,
                   choices=["balanced"],
                   help="'balanced' = inverse-frequency weighting on the regime "
                        "classes (upweights rare classes like isNegativeCorrelation)")
    p.add_argument("--mix-real", nargs="*", default=None,
                   metavar="PAIR",
                   help="mix real py-labeller windows into training. "
                        "Pass pair names (e.g. sp500_gold sp500_bitcoin) or "
                        "no arguments for both gold+bitcoin. Best combined "
                        "with --class-weights balanced.")
    p.add_argument("--date-cutoff", default=None,
                   help="only include real windows before this date (optional; "
                        "the primary leakage guard is leave-one-pair-out via "
                        "--mix-real with a single pair)")
    p.add_argument("--transfer-suffix", default="",
                   help="read data/processed/transfer_<pair><suffix>/ instead of "
                        "transfer_<pair>/, e.g. _dominance for the relabelled sets")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed for initialisation AND the held-out split "
                        "(default: config seed, 42). Recorded in the training record")
    args = p.parse_args(argv)

    # --mix-real with no arguments means both default pairs
    mix_real = args.mix_real
    if mix_real is not None and len(mix_real) == 0:
        mix_real = ["sp500_gold", "sp500_bitcoin"]

    # --pairs: comma-separated list for multi-pair training
    pairs = [p.strip() for p in args.pairs.split(",")] if args.pairs else None

    from pathlib import Path
    from tslearning.config import PATHS
    from tslearning.distill import train
    target = Path(args.save or (PATHS["data_processed"] / "classifier_v7_dominance.pt"))
    if target.exists() and not args.force:
        p.error(f"{target} already exists — pass a new file with --save, "
                f"or --force to replace it on purpose")
    label = ", ".join(pairs) if pairs else args.pair
    with stage(f"distill · {label} (V7)", args.quiet):
        r = train.run(pair=args.pair, pairs=pairs,
                      results_dir=args.results, save=args.save,
                      epochs=args.epochs, lr=args.lr, test_size=args.test_size,
                      resume=args.resume, class_weights=args.class_weights,
                      mix_real=mix_real, date_cutoff=args.date_cutoff,
                      overwrite=args.force, transfer_suffix=args.transfer_suffix,
                      seed=args.seed, verbose=not args.quiet)

    print(f"{label}: {r['n_pairs']} labelled pairs, best held-out accuracy "
          f"{r['best_acc']:.3f} -> {r['weights']}")
    return r


if __name__ == "__main__":
    main()
