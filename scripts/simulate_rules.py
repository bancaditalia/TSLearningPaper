#!/usr/bin/env python3
"""
Rule pipeline on the simulated series: how often each template detects a planted
phenomenon (power, theta > 0) or reports one that is not there (size, theta = 0).

Reads the datasets of `data_simulation_econ tests/simulation/datasets/core/` (made by
export_datasets.py). Every replication is labelled and scored exactly like a real pair
with the Python labeller: thresholds on the rep's train split, templates fitted on
train, scored on test with the bootstrap CI, and the paper's verdict. Detected =
"significant". No engine, no classifier. See tslearning/eval/simulation.py.

Examples
--------
    # all files, the two templates (forward, backward) of each planted phenomenon
    python scripts/simulate_rules.py --jobs 4

    # a quick check: one file, 10 reps
    python scripts/simulate_rules.py -f T2_theta0.4_rho0.3_n2520 --reps 10

    # every template on every file (does T1 fire where only contagion was planted?)
    python scripts/simulate_rules.py --templates all --jobs 4

    # rolling tau/qLow instead of train-fitted
    python scripts/simulate_rules.py --thresholds rolling --jobs 4

Writes results/simulation/  (results/runs/<run>/simulation/ with --run):
    reps_<file>.csv   one row per (rep, template): constants, test lift, CI, verdict
    summary.csv       one row per (file, template): fitted / testable share, detection
                      rate, and the econometric test's rate on the same configuration
    settings.json     window and cut-offs used
"""
from _cli import bootstrap, csv_list, parser, stage

bootstrap()


def main(argv=None):
    p = parser(__doc__.split("Examples")[0].strip(), epilog=__doc__)
    p.add_argument("-f", "--files", type=csv_list, default=None,
                   help="dataset files (names or stems), comma separated; default: all")
    p.add_argument("--dir", default=None,
                   help="folder of the datasets (default: data_simulation_econ tests/"
                        "simulation/datasets/core)")
    p.add_argument("--templates", type=csv_list, default=["matching"],
                   help="'matching' (default: the forward and backward template of the "
                        "planted phenomenon), 'all', or template names")
    p.add_argument("--reps", type=int, default=None,
                   help="only the first N replications of each file (default: all)")
    p.add_argument("--thresholds", default="train", choices=["train", "rolling"],
                   help="tau/qLow fitted on each rep's train split (default) or rolling")
    p.add_argument("-j", "--jobs", type=int, default=1,
                   help="parallel processes (default 1)")
    p.add_argument("--summary-only", action="store_true",
                   help="rebuild summary.csv from the reps_*.csv already saved; "
                        "nothing is rerun")
    args = p.parse_args(argv)

    import pandas as pd
    from tslearning.eval import simulation as S

    cols = ["sim_template", "theta", "rho0", "n", "template", "reps", "decided_share",
            "detection_rate", "detection_rate_decided", "econ_testable_share",
            "econ_detection_rate"]
    if args.summary_only:
        sf = S.write_summary(S.load_reps())
        summ = pd.read_csv(sf)
        print(summ[[c for c in cols if c in summ.columns]].to_string(index=False))
        print(f"\nsummary -> {sf}")
        return summ

    which = args.templates[0] if args.templates in (["matching"], ["all"]) else args.templates
    files = S.find_files(args.files, args.dir)
    parts = []
    for f in files:
        meta = S.parse_name(f)
        names = S.templates_for(meta["sim_template"], which)
        with stage(f"simulation · {f.name} · {','.join(names)}", args.quiet):
            parts.append(S.run_file(f, which, reps=args.reps, jobs=args.jobs,
                                    thresholds=args.thresholds, verbose=not args.quiet))
    reps = pd.concat(parts, ignore_index=True)
    sf = S.save(reps)

    summ = pd.read_csv(sf)
    summ = summ[summ["file"].isin(reps["file"].unique())]
    print()
    print(summ[[c for c in cols if c in summ.columns]].to_string(index=False))
    print(f"\n{len(reps)} rows -> {S.out_dir()}/reps_*.csv; summary -> {sf}")
    return reps


if __name__ == "__main__":
    main()
