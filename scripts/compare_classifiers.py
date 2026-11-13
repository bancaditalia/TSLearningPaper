#!/usr/bin/env python3
"""
Compare classifier recipes on the VALIDATION set: real windows of
the training years (2015–2021), exact Python labels. Never the 2022–2024 test period.

    python scripts/compare_classifiers.py
    python scripts/compare_classifiers.py --families v7r,v8r,v7r2,v8r2 --seeds 42,1,2,3,4

Each family is data/processed/classifier_<family>_s<seed>.pt. For each model and each
pair (gold, bitcoin) it measures accuracy, macro-F1 and recall on isNegativeCorrelation.
Writes:
  results/classifier_validation.csv     one row per model x pair
  results/classifier_validation.md      mean ± sd (and min–max) per family and pair
A family is BETTER (criteria fixed in advance) if its average goes up AND its spread
across seeds goes down. The ensemble of a family can be added with --ensembles
(data/processed/classifier_<family>_ens.ens, if it exists).
"""
from _cli import bootstrap, csv_list, parser

bootstrap()


def main(argv=None):
    p = parser(__doc__.strip().split("\n")[0], epilog=__doc__)
    p.add_argument("--families", type=csv_list, default=["v7r", "v8r", "v7r2", "v8r2"])
    p.add_argument("--seeds", type=csv_list, default=["42", "1", "2", "3", "4"])
    p.add_argument("--pairs", type=csv_list, default=["sp500_gold", "sp500_bitcoin"])
    p.add_argument("--step", type=int, default=5,
                   help="one window every N days (default 5)")
    p.add_argument("--ensembles", action="store_true",
                   help="also score classifier_<family>_ens.ens where it exists")
    args = p.parse_args(argv)

    import pandas as pd
    from tslearning.config import PATHS
    from tslearning.eval import fidelity

    dp, rows = PATHS["data_processed"], []
    for fam in args.families:
        models = [(f"s{s}", dp / f"classifier_{fam}_s{s}.pt") for s in args.seeds]
        if args.ensembles:
            models.append(("ens", dp / f"classifier_{fam}_ens.ens"))
        for tag, w in models:
            if not w.exists():
                print(f"missing, skipped: {w.name}")
                continue
            for pair in args.pairs:
                r = fidelity._run("valid", weights=w, pair=pair, step=args.step,
                                  verbose=False)["row"]
                rows.append({"family": fam, "model": tag, **r})
                print(f"{fam:6s} {tag:4s} {pair:14s} acc {r['accuracy']:.3f}  "
                      f"macro-F1 {r['macro_f1']:.3f}  neg recall "
                      f"{r['neg_recall'] if r['neg_recall'] is None else round(r['neg_recall'], 3)}")
    if not rows:
        p.error("no models found")
    df = pd.DataFrame(rows)
    out = PATHS["results"] / "classifier_validation.csv"
    df.to_csv(out, index=False)

    seeds = df[df.model != "ens"]
    lines = ["# Classifier comparison on the VALIDATION set (real, training years)", "",
             f"Windows every {args.step} days; exact Python labels. Mean ± sd over seeds "
             f"({', '.join(args.seeds)}), min–max in brackets. Not the test period.", ""]
    for pair, g in seeds.groupby("pair"):
        b = g.iloc[0]
        lines += [f"## {pair} — {int(b['n_windows'])} windows, {b['period']}, "
                  f"{int(b['neg_support'])} negative; majority baseline "
                  f"{b['baseline_accuracy']:.2f} / {b['baseline_macro_f1']:.2f}", "",
                  "| family | accuracy | macro-F1 | neg. recall | ensemble acc. / F1 / neg. |",
                  "|---|---|---|---|---|"]
        for fam, h in g.groupby("family", sort=False):
            def f(c):
                v = h[c].astype(float)
                return f"{v.mean():.2f} ± {v.std():.2f} [{v.min():.2f}–{v.max():.2f}]"
            e = df[(df.family == fam) & (df.pair == pair) & (df.model == "ens")]
            ens = (f"{e.iloc[0]['accuracy']:.2f} / {e.iloc[0]['macro_f1']:.2f} / "
                   f"{e.iloc[0]['neg_recall']:.2f}" if len(e) else "—")
            lines.append(f"| {fam} | {f('accuracy')} | {f('macro_f1')} | "
                         f"{f('neg_recall')} | {ens} |")
        lines.append("")
        # per class: what each recipe gets right (recall) and how often its calls are
        # right (precision); "predicted" = how many windows it puts in the class
        lines += [f"### {pair} — per class (mean over seeds; ensemble in brackets)", "",
                  f"true counts: positive {int(b['pos_support'])}, negative "
                  f"{int(b['neg_support'])}, none {int(b['none_support'])}", "",
                  "| family | class | recall | precision | windows predicted |",
                  "|---|---|---|---|---|"]
        for fam, h in g.groupby("family", sort=False):
            e = df[(df.family == fam) & (df.pair == pair) & (df.model == "ens")]
            for cls in ("pos", "neg", "none"):
                def cell(c, fmt="{:.2f}"):
                    v = h[f"{cls}_{c}"].astype(float).mean()
                    s = fmt.format(v)
                    if len(e) and e.iloc[0][f"{cls}_{c}"] == e.iloc[0][f"{cls}_{c}"]:
                        s += " (" + fmt.format(float(e.iloc[0][f"{cls}_{c}"])) + ")"
                    return s
                lines.append(f"| {fam} | {cls} | {cell('recall')} | {cell('precision')} | "
                             f"{cell('predicted', '{:.0f}')} |")
        lines.append("")
    (PATHS["results"] / "classifier_validation.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote -> {out} and classifier_validation.md")
    return df


if __name__ == "__main__":
    main()
