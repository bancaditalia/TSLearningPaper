"""
Distillation trainer — the library half of `scripts/distill.py`.

The teacher is the Vadalog reasoner; the student is V7. Training data is the
engineered transfer set (windows) plus the reasoner's correlation labels, read
through labeling/predicate_targets.py.

Two things the notebook version leaves implicit and this one makes explicit,
because the missing split is the real problem:
  * the split is stratified on the regime label and seeded from the config, so a
    re-run reproduces it;
  * the reported numbers come from the HELD-OUT slice only, and the checkpoint kept
    is the best held-out one, not the last epoch.

Full-batch gradient steps (no mini-batching) are kept from the notebook: the
transfer set is a few hundred pairs of 50 points, so the whole thing fits in one
forward pass and mini-batching would only add noise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from tslearning.config import CFG, PATHS


def load_real_training_data(pairs=("sp500_gold", "sp500_bitcoin"),
                            regime_dir=None, date_cutoff: "str | None" = None,
                            verbose: bool = True) -> dict:
    """Build (X, y) from real pair windows labelled by the Python labeller.

    Reads the sliding windows directly from the aligned return series (same
    slicing as py_labeler.label_pair) and the regime labels from the per-pair
    CSVs in real_pairs_regime_py/.  Returns the same dict shape as
    load_training_data so the two can be concatenated.

    date_cutoff: if set (e.g. "2022-01-01"), only include windows whose START
    date is strictly before the cutoff. This prevents leakage into the test
    period that the rule evaluation uses.
    """
    import pandas as pd
    from tslearning.datasets.data_utils import zscore_window
    from tslearning.datasets.pairs import describe, load_series

    data_dir = PATHS["data_processed"]
    regime_dir = Path(regime_dir or (data_dir / "real_pairs_regime_py"))
    softmax_group = list(CFG["classifier"]["softmax_group"])

    all_X, all_y, all_names = [], [], []
    skipped_after_cutoff = 0

    for pair in pairs:
        d = describe(pair)
        transformed = load_series([d["a"], d["b"]])
        sp = transformed[d["a"]]
        ot = transformed[d["b"]]
        seq_len = d["seq_len"]

        # Align — same logic as py_labeler._aligned
        # the period and stride the real pairs were LABELLED with (config real_pairs),
        # so day index t here is the same t as in the label CSVs
        _rp = CFG.get("real_pairs") or {}
        lo, hi = _rp.get("date_from", "2015-01-01"), _rp.get("date_to", "2024-12-31")
        idx = sp.dropna().sort_index().loc[lo:hi].index.intersection(
              ot.dropna().sort_index().loc[lo:hi].index).sort_values()
        sp_arr = sp.loc[idx].to_numpy(float)
        ot_arr = ot.loc[idx].to_numpy(float)

        # Date map for cutoff filtering
        dates = pd.DatetimeIndex(idx)
        cutoff_dt = pd.Timestamp(date_cutoff) if date_cutoff else None

        # Read regime labels from the py_labeler output
        pair_dir = regime_dir / d["slug"]
        regime_windows = {}  # t -> class index
        for ci, pred in enumerate(softmax_group):
            csv = pair_dir / f"{pred}_{d['slug']}.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            for _, row in df.iterrows():
                t = int(row["Start"])
                regime_windows[t] = ci

        # Build window pairs and labels
        stride = int(_rp.get("stride", 1))
        N = len(idx)
        tps = list(range(0, N - seq_len + 1, stride))

        for t in tps:
            if t not in regime_windows:
                continue
            if cutoff_dt is not None and dates[t] >= cutoff_dt:
                skipped_after_cutoff += 1
                continue
            wa = zscore_window(sp_arr[t:t + seq_len])
            wb = zscore_window(ot_arr[t:t + seq_len])
            all_X.append(np.stack([wa, wb], axis=0))
            y_row = np.zeros(len(softmax_group), dtype=np.float32)
            y_row[regime_windows[t]] = 1.0
            all_y.append(y_row)
            all_names.append((f"{d['a_key']}_real_{t}", f"{d['b_key']}_real_{t}"))

    X = np.stack(all_X).astype(np.float32)
    y_soft = np.stack(all_y)

    if verbose:
        dist = {c: int(y_soft[:, j].sum()) for j, c in enumerate(softmax_group)}
        cutoff_msg = f" (cutoff {date_cutoff}, {skipped_after_cutoff} skipped)" \
                     if date_cutoff else ""
        print(f"real training data: {len(all_names)} windows from {len(pairs)} pairs"
              f"{cutoff_msg} | label dist {dist}")

    return {"X": X, "y_soft": y_soft, "pair_names": all_names,
            "soft_cols": softmax_group}


def load_training_data(results_dir=None, windows_a=None, windows_b=None,
                       pair: str = "sp500_bitcoin", side_a: str = "sp500",
                       side_b: str = "btc", verbose: bool = True) -> dict:
    """Assemble (X, y) for distillation from the reasoner output + the pair windows.

    The window arrays default to the engineered_*.npy written by
    labeling/transfer_set.py, and pair name -> window mapping follows the
    "{side}_synth_{i}" convention the batch CSVs use, so index i of the array is
    window i in the reasoner's output.
    """
    from tslearning.distill.classifier import build_input
    from tslearning.labeling.predicate_targets import build_targets

    data_dir = PATHS["data_processed"]
    results_dir = Path(results_dir or (data_dir / "batched_results"))
    a_name, b_name = pair.split("_", 1)

    # Resolve through transfer_set, which owns the naming: the arrays and the window
    # ids use the SHORT name ("btc"), while the pair spec uses the series name
    # ("bitcoin"). side_a/side_b are already the short names — prefer them.
    from tslearning.labeling.transfer_set import engineered_path

    if windows_a is None:
        windows_a = np.load(engineered_path(side_a or a_name, data_dir))
    if windows_b is None:
        windows_b = np.load(engineered_path(side_b or b_name, data_dir))

    tgt = build_targets(results_dir, side_a=side_a, side_b=side_b)
    if not tgt["pair_names"]:
        raise FileNotFoundError(
            f"no reasoner output in {results_dir}. Run `label.py -p {pair} "
            "--engine vadalog` first (or point --results at the right folder).")

    lookup = {}
    for i in range(len(windows_a)):
        lookup[f"{side_a}_synth_{i}"] = windows_a[i]
    for i in range(len(windows_b)):
        lookup[f"{side_b}_synth_{i}"] = windows_b[i]

    keep = [i for i, (a, b) in enumerate(tgt["pair_names"])
            if a in lookup and b in lookup]
    if len(keep) < len(tgt["pair_names"]) and verbose:
        print(f"dropping {len(tgt['pair_names']) - len(keep)} labelled pairs with no "
              "matching window (transfer set and reasoner output are out of sync)")

    pair_names = [tgt["pair_names"][i] for i in keep]
    X = build_input(pair_names, lookup, standardize=True)
    y_soft = tgt["y_soft"][keep]

    if verbose:
        dist = {c: int(y_soft[:, j].sum()) for j, c in enumerate(tgt["soft_cols"])}
        print(f"training data: {len(pair_names)} pairs, X {X.shape} | label dist {dist}")
    return {"X": X, "y_soft": y_soft, "pair_names": pair_names,
            "soft_cols": tgt["soft_cols"]}


def load_mixed_training_data(real_pairs=("sp500_gold", "sp500_bitcoin"),
                             results_dir=None, pair: str = "sp500_bitcoin",
                             side_a: str = "sp500", side_b: str = "btc",
                             date_cutoff: "str | None" = None,
                             verbose: bool = True) -> dict:
    """Concatenate synthetic (reasoner-labelled) and real (py-labeller) training data.

    The combined set gives the classifier real examples of structured negative
    correlation (flight-to-safety patterns) that the independent-DDPM synthetic
    generator cannot produce, while keeping the synthetic set's balanced coverage
    of all three regimes.

    To avoid data leakage, use LEAVE-ONE-PAIR-OUT: train on real pair A, evaluate
    on real pair B. E.g. real_pairs=("sp500_gold",) then evaluate on bitcoin.
    date_cutoff is available but secondary — the pair split is the primary guard.
    """
    synth = load_training_data(results_dir=results_dir, pair=pair,
                               side_a=side_a, side_b=side_b, verbose=verbose)
    real = load_real_training_data(pairs=real_pairs, date_cutoff=date_cutoff,
                                  verbose=verbose)

    assert synth["soft_cols"] == real["soft_cols"], \
        f"column mismatch: {synth['soft_cols']} vs {real['soft_cols']}"

    X = np.concatenate([synth["X"], real["X"]], axis=0)
    y_soft = np.concatenate([synth["y_soft"], real["y_soft"]], axis=0)
    pair_names = synth["pair_names"] + real["pair_names"]

    if verbose:
        dist = {c: int(y_soft[:, j].sum()) for j, c in enumerate(synth["soft_cols"])}
        print(f"mixed training data: {len(pair_names)} total "
              f"({len(synth['pair_names'])} synth + {len(real['pair_names'])} real) "
              f"| label dist {dist}")

    return {"X": X, "y_soft": y_soft, "pair_names": pair_names,
            "soft_cols": synth["soft_cols"]}


def train(X, y_soft, soft_cols, epochs: int = 1500, lr: float = 1e-3,
          weight_decay: float = 1e-4, test_size: float = 0.2,
          eval_every: int = 100, resume=None, seed: "int | None" = None,
          class_weights: "str | list | None" = None,
          verbose: bool = True) -> dict:
    """Train V7 with a stratified held-out split and best-checkpoint selection.

    Returns {"model", "best_acc", "report", "confusion", "test_idx", "history"}.
    """
    import torch
    import torch.nn as nn
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    from tslearning.distill.classifier import build_classifier

    seed = CFG["seed"] if seed is None else seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    y_idx = np.asarray(y_soft).argmax(1)
    Xt = torch.tensor(np.asarray(X, np.float32))
    yt = torch.tensor(y_idx, dtype=torch.long)

    tr, te = train_test_split(np.arange(len(Xt)), test_size=test_size,
                              random_state=seed, stratify=y_idx)

    model = build_classifier(len(soft_cols))
    if resume:
        model.load_state_dict(torch.load(resume, map_location="cpu"))
        if verbose:
            print(f"resumed from {resume}")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Class weights: inverse frequency so rare classes get proportionally higher loss.
    # On the current transfer set (237/168/120 = pos/none/neg) this is mild; on a
    # mixed real+synthetic set with very few negatives it matters much more.
    if class_weights == "balanced":
        counts = np.bincount(y_idx, minlength=len(soft_cols)).astype(float)
        counts = np.maximum(counts, 1.0)  # avoid div-by-zero on empty classes
        w = len(y_idx) / (len(soft_cols) * counts)
        w = torch.tensor(w, dtype=torch.float32)
        if verbose:
            print(f"  class weights: {dict(zip(soft_cols, w.tolist()))}")
    elif class_weights is not None:
        w = torch.tensor(class_weights, dtype=torch.float32)
    else:
        w = None

    ce = nn.CrossEntropyLoss(weight=w)

    best_acc, best_state, history = -1.0, None, []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = ce(model(Xt[tr]), yt[tr])
        loss.backward()
        opt.step()

        if (epoch + 1) % eval_every == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(Xt[te]).argmax(1) == yt[te]).float().mean().item()
            history.append({"epoch": epoch + 1, "loss": float(loss.item()), "held_out_acc": acc})
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"  epoch {epoch+1:5d} | loss {loss.item():.4f} | held-out acc {acc:.3f}")

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred = model(Xt[te]).argmax(1).numpy()
    true = y_idx[te]
    labels = list(range(len(soft_cols)))
    report = classification_report(true, pred, labels=labels, target_names=soft_cols,
                                   digits=3, zero_division=0)
    cm = confusion_matrix(true, pred, labels=labels)

    if verbose:
        print(f"\nbest held-out accuracy {best_acc:.3f}")
        print(report)
        print("confusion (rows=true, cols=pred):\n", cm)

    return {"model": model, "best_acc": best_acc, "report": report,
            "confusion": cm, "test_idx": te, "history": history}


# ══════════════════════════════════════════════════════════════════════════════
#  Multi-pair synthetic loader
# ══════════════════════════════════════════════════════════════════════════════

def load_multi_pair_training_data(
        pairs: "list[str]",
        transfer_dirs: "list[str | Path] | None" = None,
        verbose: bool = True) -> dict:
    """Load and concatenate synthetic training data from multiple pairs.

    Each pair must have its own directory containing:
      - engineered_<side_a>.npy, engineered_<side_b>.npy  (the windows)
      - {predicate}_*.csv  (the Vadalog labels)

    pairs:         list of pair specs, e.g. ["sp500_bitcoin", "sp500_gold", "sp500_nasdaq"]
    transfer_dirs: per-pair directories. If None, defaults to
                   data/processed/transfer_<slug>/ for each pair.
    """
    from tslearning.datasets.pairs import describe
    from tslearning.labeling.transfer_set import engineered_name

    data_dir = PATHS["data_processed"]
    if transfer_dirs is None:
        transfer_dirs = [data_dir / f"transfer_{describe(p)['slug']}" for p in pairs]
    else:
        transfer_dirs = [Path(d) for d in transfer_dirs]

    assert len(pairs) == len(transfer_dirs), \
        f"need one directory per pair: {len(pairs)} pairs, {len(transfer_dirs)} dirs"

    all_X, all_y, all_names = [], [], []
    soft_cols = None

    for pair, tdir in zip(pairs, transfer_dirs):
        d = describe(pair)
        a_short = engineered_name(d["a"])
        b_short = engineered_name(d["b"])
        data = load_training_data(
            results_dir=tdir,
            windows_a=np.load(tdir / f"engineered_{a_short}.npy"),
            windows_b=np.load(tdir / f"engineered_{b_short}.npy"),
            pair=pair, side_a=a_short, side_b=b_short,
            verbose=verbose)

        if soft_cols is None:
            soft_cols = data["soft_cols"]
        assert data["soft_cols"] == soft_cols

        all_X.append(data["X"])
        all_y.append(data["y_soft"])
        all_names.extend(data["pair_names"])

    X = np.concatenate(all_X, axis=0)
    y_soft = np.concatenate(all_y, axis=0)

    if verbose:
        dist = {c: int(y_soft[:, j].sum()) for j, c in enumerate(soft_cols)}
        print(f"multi-pair training data: {len(all_names)} total from {len(pairs)} pairs "
              f"| label dist {dist}")

    return {"X": X, "y_soft": y_soft, "pair_names": all_names, "soft_cols": soft_cols}


def transfer_dirs_for(pairs, suffix: str = "") -> list:
    """data/processed/transfer_<slug><suffix>/ for each pair."""
    from tslearning.datasets.pairs import describe
    return [PATHS["data_processed"] / f"transfer_{describe(p)['slug']}{suffix}"
            for p in pairs]


# ══════════════════════════════════════════════════════════════════════════════
#  Stage runner
# ══════════════════════════════════════════════════════════════════════════════

def run(pair: str = "sp500_bitcoin", pairs: "list[str] | None" = None,
        results_dir=None, transfer_dirs: "list[str | Path] | None" = None,
        save=None, epochs: int = 1500,
        lr: float = 1e-3, test_size: float = 0.2, resume=None,
        class_weights: "str | list | None" = None,
        mix_real: "tuple | list | None" = None,
        date_cutoff: "str | None" = None,
        overwrite: bool = False,
        transfer_suffix: str = "",
        seed: "int | None" = None,
        verbose: bool = True) -> dict:
    """Train the student on labelled transfer set(s) and save the weights.

    transfer_suffix: read data/processed/transfer_<slug><suffix>/ instead of
    transfer_<slug>/ — e.g. "_dominance" for the sets relabelled by relabel.py.

    overwrite: allow `save` to replace an existing weights file. Off by default, and
    checked BEFORE training: an accidental run without --save used to replace
    classifier_v7_dominance.pt, the v7 weights every comparison depends on.

    pairs: list of pair specs for multi-pair training. When given, each pair's
    synthetic data is loaded from its own directory (transfer_dirs, or the default
    data/processed/transfer_<slug>/). The single-pair `pair` argument is ignored.

    mix_real: if given, a list of pair names whose real py-labeller output is
    mixed into the training set. Use leave-one-pair-out to avoid leakage:
    train on ["sp500_gold"], evaluate on bitcoin (or vice versa).
    Use together with class_weights="balanced" for best results on rare classes.

    date_cutoff: optional, only include real windows before this date.
    """
    # Default target is the dominance-label student. Never default to classifier_v7_corr.pt
    # again — those are the pre-fix weights and the ablation row needs them intact.
    save_path = Path(save or (PATHS["data_processed"] / "classifier_v7_dominance.pt"))
    if save_path.exists() and not overwrite:
        raise FileExistsError(
            f"{save_path} already exists and would be overwritten. Pass a new file "
            f"with --save, or --force to replace it on purpose.")

    import torch

    if transfer_suffix and transfer_dirs is None and (pairs or results_dir is None):
        transfer_dirs = transfer_dirs_for(pairs or [pair], transfer_suffix)
    if pairs and len(pairs) > 0:
        data = load_multi_pair_training_data(pairs=pairs, transfer_dirs=transfer_dirs,
                                             verbose=verbose)
    elif mix_real:
        data = load_mixed_training_data(
            real_pairs=mix_real, results_dir=results_dir, pair=pair,
            date_cutoff=date_cutoff, verbose=verbose)
    elif results_dir is None:
        # single pair, default layout: data/processed/transfer_<pair>/ holds the windows
        # AND the reasoner labels, exactly as for --pairs
        data = load_multi_pair_training_data(pairs=[pair], transfer_dirs=transfer_dirs,
                                             verbose=verbose)
    else:
        # legacy flat layout (v7's training set): labels in `results_dir`, windows in
        # data/processed/engineered_*.npy
        data = load_training_data(results_dir=results_dir, pair=pair, verbose=verbose)
    out = train(data["X"], data["y_soft"], data["soft_cols"], epochs=epochs, lr=lr,
                test_size=test_size, resume=resume, class_weights=class_weights,
                seed=seed, verbose=verbose)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out["model"].state_dict(), save_path)
    # record what the weights were trained on, so the held-out synthetic split can be
    # reproduced exactly later (eval.py --dataset synthetic reads this)
    import json
    from tslearning.config import CFG as _CFG
    save_path.with_suffix(".json").write_text(json.dumps({
        "pairs": list(pairs) if pairs else None,
        "pair": None if pairs else pair,
        "results_dir": None if pairs or results_dir is None else str(results_dir),
        "mix_real": list(mix_real) if mix_real else None,
        "transfer_dirs": [str(Path(d).name) for d in transfer_dirs] if transfer_dirs
                         else None,
        "test_size": test_size, "seed": _CFG["seed"] if seed is None else seed, "epochs": epochs, "lr": lr,
        "class_weights": class_weights, "n_pairs": len(data["pair_names"]),
        "seq_len": int(data["X"].shape[-1]), "run": _CFG["run"]["name"],
        "best_acc": float(out.get("best_acc", float("nan"))),
    }, indent=2))
    if verbose:
        print(f"wrote -> {save_path}")

    out["weights"] = save_path
    out["n_pairs"] = len(data["pair_names"])
    return out
