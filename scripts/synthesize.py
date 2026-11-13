#!/usr/bin/env python3
"""
Stage 2 — synthesizer. Train a generator for one series and sample from it.

The per-series differences live in tslearning/synth/registry.py; everything else is
shared.

Examples
--------
    python scripts/synthesize.py -d SP500
    python scripts/synthesize.py -d Bitcoin --epochs 400 --n-samples 2000
    python scripts/synthesize.py -d Electricity            # per-window normalization
    python scripts/synthesize.py -d US_Inflation           # macro, reads windowed_macro
    python scripts/synthesize.py --list                    # what -d accepts

Writes data/processed/synthetic_ddpm_<series>.npy and ddpm_unet1d_<series>.pt —
the same filenames nb_06 wrote, so nb_10 and the transfer set keep working.

--model exists so a second synthesizer (Diffusion-TS, TimeGAN) can be
added behind the same CLI later. Only `ddpm` is implemented today.
"""
from _cli import bootstrap, parser, stage

bootstrap()


def main(argv=None):
    p = parser(__doc__.split("Examples")[0].strip(), epilog=__doc__)
    p.add_argument("-d", "--dataset", help="series to synthesize (e.g. SP500)")
    p.add_argument("--model", default="ddpm", choices=["ddpm"],
                   help="synthesizer family (default: ddpm)")
    p.add_argument("--epochs", type=int, default=None,
                   help="override the registry's epoch count")
    p.add_argument("--n-samples", type=int, default=None,
                   help="how many synthetic windows to draw")
    p.add_argument("--diffusion-steps", type=int, default=None,
                   help="T, the number of diffusion steps (default 500)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--no-save", action="store_true",
                   help="train and report, but don't write .npy/.pt")
    p.add_argument("--list", action="store_true",
                   help="list the datasets -d accepts, with their resolved settings")
    args = p.parse_args(argv)

    from tslearning.synth import registry

    if args.list:
        print(f"{'dataset':18s} {'source':10s} {'normalize':12s} {'epochs':>7s} {'samples':>8s}")
        for name in registry.known_datasets():
            s = registry.spec(name)
            print(f"{name:18s} {s['source']:10s} {s['normalize']:12s} "
                  f"{s['epochs']:>7d} {s['n_samples']:>8d}")
        return None

    if not args.dataset:
        p.error("-d/--dataset is required (or use --list)")

    if args.model != "ddpm":
        p.error(f"model {args.model!r} not implemented")

    from tslearning.synth import ddpm
    with stage(f"synthesize · {args.dataset} ({args.model})", args.quiet):
        r = ddpm.run(args.dataset, epochs=args.epochs, n_samples=args.n_samples,
                     diffusion_steps=args.diffusion_steps, batch_size=args.batch_size,
                     lr=args.lr, save=not args.no_save, verbose=not args.quiet)

    print(f"{args.dataset}: {r['samples'].shape[0]} synthetic windows of "
          f"{r['seq_len']} -> {r['paths'].get('samples', '(not saved)')}")
    return r


if __name__ == "__main__":
    main()
