#!/usr/bin/env python3
"""
Train V8 — V7 plus an auxiliary head trained on the reasoner's intermediate quantities
(same%, diff%, score). The auxiliary head is used only in training: the saved model
predicts the regime only and is used exactly like a V7 file (infer.py, eval.py, ...).
V7's own trainer is scripts/distill.py, unchanged.

Examples
--------
    python scripts/distill_v8.py --aux-weight 0.1 --save data/processed/classifier_v8_l01.pt
    python scripts/distill_v8.py --aux-weight 0.5 --save data/processed/classifier_v8_l05.pt

Same windows, split, seed and settings as v7c (3 transfer sets, no class weights,
1500 epochs, lr 1e-3), so v8 vs v7c isolates the auxiliary head. Writes the weights and
a <weights>.json training record (aux weight, targets, per-epoch history).
"""
from _cli import bootstrap, csv_list, parser, stage

bootstrap()


def main(argv=None):
    p = parser(__doc__.split("Examples")[0].strip(), epilog=__doc__)
    p.add_argument("--pairs", type=csv_list,
                   default=["sp500_bitcoin", "sp500_gold", "sp500_nasdaq"],
                   help="transfer sets to train on (default: the three, as v7c/v7b)")
    p.add_argument("--aux-weight", type=float, required=True,
                   help="λ: weight of the auxiliary loss (0 = v7c again)")
    p.add_argument("--aux", type=csv_list, default=["same", "diff", "score"],
                   help="auxiliary targets, from: same, diff, score (default: these three) "
                        "and sign, lag1xy, lag1yx (add information the label lacks)")
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--save", required=True, help="weights file to write (never replaced "
                                                 "unless --force)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed (default: config seed, 42); recorded")
    p.add_argument("--transfer-suffix", default="",
                   help="read data/processed/transfer_<pair><suffix>/, e.g. _dominance")
    args = p.parse_args(argv)

    from pathlib import Path
    if Path(args.save).exists() and not args.force:
        p.error(f"{args.save} already exists — pass a new file, or --force")
    from tslearning.distill.train_v8 import ALL_AUX
    bad = [a for a in args.aux if a not in ALL_AUX]
    if bad:
        p.error(f"unknown --aux target(s) {bad}; choose from {', '.join(ALL_AUX)}")

    from tslearning.distill import train_v8
    with stage(f"distill V8 · λ={args.aux_weight} · {', '.join(args.pairs)}", args.quiet):
        r = train_v8.run(args.pairs, args.aux_weight, args.save, aux=tuple(args.aux),
                         epochs=args.epochs, lr=args.lr, test_size=args.test_size,
                         overwrite=args.force, transfer_suffix=args.transfer_suffix,
                         seed=args.seed, verbose=not args.quiet)
    print(f"V8 λ={args.aux_weight}: {r['n_pairs']} labelled pairs, best held-out accuracy "
          f"{r['best_acc']:.3f} -> {r['weights']}")
    return r


if __name__ == "__main__":
    main()
