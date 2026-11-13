#!/usr/bin/env python3
"""
Pin several trained classifiers into one ensemble manifest (<name>.ens).

    python scripts/make_ensemble.py \
        --members data/processed/classifier_v8r_s42.pt,data/processed/classifier_v8r_s1.pt,\
data/processed/classifier_v8r_s2.pt,data/processed/classifier_v8r_s3.pt,data/processed/classifier_v8r_s4.pt \
        --save data/processed/classifier_v8r_ens.ens

The manifest records each member's path, SHA-256 and training record. Every script that
takes a classifier (infer.py, eval.py, classifier_timing.py, `classifier.weights` in
the config) accepts the .ens file: the members' regime probabilities are averaged.
A member whose file changes afterwards is refused (SHA-256 check).
"""
from _cli import bootstrap, csv_list, parser

bootstrap()


def main(argv=None):
    p = parser(__doc__.strip().split("\n")[0], epilog=__doc__)
    p.add_argument("--members", type=csv_list, required=True,
                   help="comma-separated weight files (.pt)")
    p.add_argument("--save", required=True, help="manifest to write, ending in .ens")
    p.add_argument("--force", action="store_true", help="replace an existing manifest")
    args = p.parse_args(argv)

    from tslearning.distill import ensemble
    out = ensemble.write_manifest(args.members, args.save, overwrite=args.force)
    print(f"ensemble of {len(args.members)} -> {out}")
    return out


if __name__ == "__main__":
    main()
