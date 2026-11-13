"""
Speedup — reasoner seconds vs student seconds per pair.

This is the measurement that justifies the neural stage existing at all: the
reasoner is the auditable definition, the student is its fast copy, and "how much
faster" is a number that belongs in the abstract. Observed so
far: ~97% agreement, five to six orders of magnitude faster.

Timing the reasoner honestly is the awkward part, so both options from the notebook
are kept:
  --measure-live   time one real batch now (needs the engine running; costs minutes)
  --reasoner-seconds / --reasoner-pairs   supply the wall-clock of a batch you
                   already ran. Preferred for a paper table: a fresh single-batch
                   timing includes JVM warm-up and understates the real cost.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from tslearning.config import CFG, PATHS


def time_classifier(model, X, warmup: int = 8, repeats: int = 3) -> float:
    """Seconds per pair for the student, best of `repeats` after a warm-up pass."""
    import torch

    Xt = torch.tensor(np.asarray(X, np.float32))
    model.eval()
    with torch.no_grad():
        model(Xt[:min(warmup, len(Xt))])          # warm-up: allocator, kernels
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(Xt)
            best = min(best, time.perf_counter() - t0)
    return best / len(Xt)


def time_reasoner_live(windows_a, windows_b, n_pairs: int = 25,
                       verbose: bool = True) -> float:
    """Seconds per pair, measured by running the engine once on `n_pairs` pairs."""
    from tslearning.labeling import reasoner_client
    from tslearning.labeling.transfer_set import build_pairs_csv

    data_dir = PATHS["data_processed"]
    probe = data_dir / "speed_probe.csv"
    out = data_dir / "speed_probe_results"
    build_pairs_csv(windows_a[:n_pairs], windows_b[:n_pairs], probe)

    t0 = time.perf_counter()
    reasoner_client.evaluate(probe, "probe", out)
    elapsed = time.perf_counter() - t0

    if verbose:
        print(f"reasoner (measured): {n_pairs} pairs in {elapsed:.1f}s "
              f"-> {elapsed/n_pairs:.3f} s/pair")
    return elapsed / n_pairs


def build_table(reasoner_s_per_pair: float, clf_s_per_pair: float,
                agreement: "dict | None" = None, per_n: int = 10_000) -> pd.DataFrame:
    """The speed table, with the agreement columns folded in when available."""
    table = pd.DataFrame([
        {"method": "reasoner (Vadalog)", "s_per_pair": reasoner_s_per_pair,
         f"per_{per_n}_pairs": reasoner_s_per_pair * per_n},
        {"method": "classifier (V7)", "s_per_pair": clf_s_per_pair,
         f"per_{per_n}_pairs": clf_s_per_pair * per_n},
    ])
    table["speedup_x"] = reasoner_s_per_pair / clf_s_per_pair
    if agreement:
        table["agreement_acc"] = agreement.get("accuracy")
        table["agreement_macroF1"] = agreement.get("macro_f1")
    return table


# ══════════════════════════════════════════════════════════════════════════════
#  Stage runner
# ══════════════════════════════════════════════════════════════════════════════

def run(weights=None, results_dir=None, measure_live: bool = False,
        reasoner_seconds: "float | None" = None, reasoner_pairs: int = 25,
        with_agreement: bool = True, save=None, verbose: bool = True) -> dict:
    """Produce results/speed_agreement.csv.

    The student is timed on the engineered transfer set (the same inputs the
    reasoner was given, so the comparison is apples to apples).
    """
    from tslearning.distill.classifier import build_input, load_classifier
    from tslearning.labeling.predicate_targets import build_targets

    data_dir = PATHS["data_processed"]
    results_dir = Path(results_dir or (data_dir / "batched_results"))

    a = np.load(data_dir / "engineered_sp500.npy")
    b = np.load(data_dir / "engineered_bitcoin.npy")
    lookup = {f"sp500_synth_{i}": a[i] for i in range(len(a))}
    lookup.update({f"btc_synth_{i}": b[i] for i in range(len(b))})

    model = load_classifier(weights)

    tgt = build_targets(results_dir)
    pair_names = [p for p in tgt["pair_names"] if p[0] in lookup and p[1] in lookup]
    if pair_names:
        keep = [i for i, p in enumerate(tgt["pair_names"]) if p in set(pair_names)]
        X = build_input(pair_names, lookup, standardize=True)
        y_soft = tgt["y_soft"][keep]
    else:
        # No labels yet: still time the student on every available pair.
        n = min(len(a), len(b))
        pair_names = [(f"sp500_synth_{i}", f"btc_synth_{i}") for i in range(n)]
        X = build_input(pair_names, lookup, standardize=True)
        y_soft = None

    clf_s = time_classifier(model, X)
    if verbose:
        print(f"classifier: {len(X)} pairs -> {clf_s*1e3:.4f} ms/pair")

    if measure_live:
        reasoner_s = time_reasoner_live(a, b, n_pairs=reasoner_pairs, verbose=verbose)
    else:
        if reasoner_seconds is None:
            raise ValueError("reasoner_seconds is required unless measure_live=True — "
                             "there is no default reasoner time (the old 300 s was an "
                             "assumption)")
        reasoner_s = reasoner_seconds / reasoner_pairs
        if verbose:
            print(f"reasoner (from your run): {reasoner_s:.3f} s/pair "
                  f"({reasoner_seconds}s / {reasoner_pairs} pairs)")

    agreement = None
    if with_agreement and y_soft is not None:
        from tslearning.eval.fidelity import evaluate_regime
        agreement = evaluate_regime(model, X, y_soft, tgt["soft_cols"],
                                    title="agreement with reasoner", verbose=verbose)

    table = build_table(reasoner_s, clf_s, agreement)

    out = Path(save or (PATHS["results"] / "speed_agreement.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    if verbose:
        print("\n" + table.to_string(index=False))
        print(f"\nspeedup: {reasoner_s/clf_s:,.0f}x")
        print(f"labelling 10k pairs -> reasoner ~{reasoner_s*10000/3600:.1f} h  |  "
              f"classifier ~{clf_s*10000:.1f} s")
        print(f"wrote -> {out}")

    return {"table": table, "reasoner_s_per_pair": reasoner_s,
            "clf_s_per_pair": clf_s, "speedup": reasoner_s / clf_s,
            "agreement": agreement, "out_path": out}
