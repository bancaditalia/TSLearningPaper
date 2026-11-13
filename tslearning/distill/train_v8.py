"""
Train V8: V7's regime classifier + an auxiliary head trained on the reasoner's
intermediate quantities (classifier_v8.py). V7's trainer (train.py) is not touched.

Everything that decides the comparison with v7c is kept identical:
  * same windows, in the same order (train.load_multi_pair_training_data)
  * same stratified split, same seed, same test_size -> the SAME held-out windows
  * same optimiser, lr, weight decay, full-batch epochs, best checkpoint chosen on
    held-out REGIME accuracy (the auxiliary loss never picks the checkpoint)
The only difference is the loss:  CE(regime) + aux_weight * MSE(aux targets).

Auxiliary targets (default), per window, from the transfer sets' reasoner output:
  same    magnitudeSameSignConsistencyPercentage / 100
  diff    magnitudeDiffSignConsistencyPercentage / 100
  score   (same% - diff%) / 100     — the quantity the dominance rule thresholds
These three carry no information the regime label lacks (the label IS score cut at
±deadband). Optional targets that DO add information (2026-10-16), computed in Python
from the raw synthetic windows of the transfer set:
  sign    share of the 50 days on which the two series move in the same direction,
          ignoring the magnitude gate (the reasoner's percentagePositive, up to its
          distinct-value counting of identical products) / 100
  lag1xy  same% - diff% with Y one day after X, over the 49 day-pairs in the window
  lag1yx  the same with X one day after Y
Volatility is deliberately NOT a target: the classifier sees each window z-scored, so
the scale that defines "big" days is not in its input — it could not learn it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tslearning.config import CFG, PATHS

SAME = "magnitudeSameSignConsistencyPercentage"
DIFF = "magnitudeDiffSignConsistencyPercentage"
DEFAULT_AUX = ("same", "diff", "score")
EXTRA_AUX = ("sign", "lag1xy", "lag1yx")
ALL_AUX = DEFAULT_AUX + EXTRA_AUX


def _lag_score(a, b) -> float:
    """same% - diff% (magnitude-gated, as the reasoner) of a[t] vs b[t+1], /100."""
    x, y = np.asarray(a[:-1], float), np.asarray(b[1:], float)
    ax, ay = np.abs(x), np.abs(y)
    same = diff = 0
    for t in range(1, len(x)):
        if (ax[t] > ax[t - 1] and ay[t] > ay[t - 1]) or (ax[t] < ax[t - 1] and ay[t] < ay[t - 1]):
            s = x[t] * y[t]
            same += s > 0
            diff += s < 0
    return (same - diff) / float(len(a))   # share of the window (was /50.0)


def extra_targets(pairs, pair_names, transfer_dirs) -> dict:
    """{pair_key: {sign, lag1xy, lag1yx}} from the RAW engineered windows."""
    from tslearning.datasets.pairs import describe
    from tslearning.labeling.predicate_targets import pkey
    from tslearning.labeling.transfer_set import engineered_name

    raw = {}
    for pair, tdir in zip(pairs, transfer_dirs):
        d = describe(pair)
        a, b = engineered_name(d["a"]), engineered_name(d["b"])
        A, B = np.load(Path(tdir) / f"engineered_{a}.npy"), np.load(Path(tdir) / f"engineered_{b}.npy")
        for i in range(len(A)):
            raw[pkey(f"{a}_synth_{i}", f"{b}_synth_{i}")] = (A[i], B[i])
    out = {}
    for n in pair_names:
        k = pkey(*n)
        x, y = raw[k]
        out[k] = {"sign": float(np.mean(np.asarray(x) * np.asarray(y) > 0)),
                  "lag1xy": _lag_score(x, y), "lag1yx": _lag_score(y, x)}
    return out


def load_aux_targets(pairs, pair_names, aux=DEFAULT_AUX, transfer_dirs=None,
                     verbose: bool = True):
    """(N, len(aux)) targets aligned with `pair_names` (as returned by
    train.load_multi_pair_training_data for the same `pairs`)."""
    from tslearning.datasets.pairs import describe
    from tslearning.labeling.predicate_targets import build_targets, pkey
    from tslearning.labeling.transfer_set import engineered_name

    from tslearning.distill.train import transfer_dirs_for
    transfer_dirs = transfer_dirs or transfer_dirs_for(pairs)
    table = {}
    for pair, tdir in zip(pairs, transfer_dirs):
        d = describe(pair)
        a, b = engineered_name(d["a"]), engineered_name(d["b"])
        t = build_targets(tdir, side_a=a, side_b=b)
        ci = {c: i for i, c in enumerate(t["aux_cols"])}
        for (x, y), row in zip(t["pair_names"], t["y_aux"]):
            table[pkey(x, y)] = (row[ci[SAME]], row[ci[DIFF]])

    missing = [n for n in pair_names if pkey(*n) not in table]
    if missing:
        raise ValueError(f"{len(missing)} training windows have no auxiliary values, "
                         f"e.g. {missing[:3]}")
    bad = [k for k in aux if k not in ALL_AUX]
    if bad:
        raise ValueError(f"unknown auxiliary target(s) {bad}; choose from {ALL_AUX}")
    extra = (extra_targets(pairs, pair_names, transfer_dirs)
             if any(k in EXTRA_AUX for k in aux) else {})
    out = []
    for n in pair_names:
        s, df = table[pkey(*n)]
        vals = {"same": s / 100.0, "diff": df / 100.0, "score": (s - df) / 100.0,
                **extra.get(pkey(*n), {})}
        out.append([vals[k] for k in aux])
    y_aux = np.asarray(out, np.float32)
    if verbose:
        print(f"auxiliary targets {list(aux)}: mean {np.round(y_aux.mean(0), 3).tolist()}, "
              f"sd {np.round(y_aux.std(0), 3).tolist()}")
    return y_aux


def train_v8(X, y_soft, y_aux, soft_cols, aux_weight: float, epochs: int = 1500,
             lr: float = 1e-3, weight_decay: float = 1e-4, test_size: float = 0.2,
             eval_every: int = 100, seed: "int | None" = None,
             verbose: bool = True) -> dict:
    import torch
    import torch.nn as nn
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    from tslearning.distill.classifier_v8 import build_classifier_v8

    seed = CFG["seed"] if seed is None else seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    y_idx = np.asarray(y_soft).argmax(1)
    Xt = torch.tensor(np.asarray(X, np.float32))
    yt = torch.tensor(y_idx, dtype=torch.long)
    at = torch.tensor(np.asarray(y_aux, np.float32))

    # the same split as train.train(): same seed, same stratification
    tr, te = train_test_split(np.arange(len(Xt)), test_size=test_size,
                              random_state=seed, stratify=y_idx)

    model = build_classifier_v8(len(soft_cols), at.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()

    best_acc, best_state, history = -1.0, None, []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits, aux = model.forward_with_aux(Xt[tr])
        l_reg, l_aux = ce(logits, yt[tr]), mse(aux, at[tr])
        loss = l_reg + aux_weight * l_aux
        loss.backward()
        opt.step()

        if (epoch + 1) % eval_every == 0:
            model.eval()
            with torch.no_grad():
                lg, ax = model.forward_with_aux(Xt[te])
                acc = (lg.argmax(1) == yt[te]).float().mean().item()
                aux_te = mse(ax, at[te]).item()
            history.append({"epoch": epoch + 1, "loss_regime": float(l_reg.item()),
                            "loss_aux": float(l_aux.item()), "held_out_acc": acc,
                            "held_out_aux_mse": aux_te})
            if acc > best_acc:          # checkpoint chosen on the REGIME only
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"  epoch {epoch+1:5d} | regime loss {l_reg.item():.4f} | "
                      f"aux loss {l_aux.item():.4f} | held-out acc {acc:.3f} | "
                      f"held-out aux mse {aux_te:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xt[te]).argmax(1).numpy()
    labels = list(range(len(soft_cols)))
    report = classification_report(y_idx[te], pred, labels=labels, target_names=soft_cols,
                                   digits=3, zero_division=0)
    cm = confusion_matrix(y_idx[te], pred, labels=labels)
    if verbose:
        print(f"\nbest held-out accuracy {best_acc:.3f}")
        print(report)
        print("confusion (rows=true, cols=pred):\n", cm)
    return {"model": model, "best_acc": best_acc, "report": report, "confusion": cm,
            "test_idx": te, "history": history}


def run(pairs, aux_weight: float, save, aux=DEFAULT_AUX, epochs: int = 1500,
        lr: float = 1e-3, test_size: float = 0.2, overwrite: bool = False,
        transfer_suffix: str = "", seed: "int | None" = None,
        verbose: bool = True) -> dict:
    import torch
    from tslearning.distill.train import load_multi_pair_training_data, transfer_dirs_for

    save_path = Path(save)
    if save_path.exists() and not overwrite:
        raise FileExistsError(f"{save_path} already exists — pass a new file, "
                              f"or --force to replace it on purpose")

    tdirs = transfer_dirs_for(pairs, transfer_suffix)
    data = load_multi_pair_training_data(pairs=list(pairs), transfer_dirs=tdirs,
                                         verbose=verbose)
    y_aux = load_aux_targets(pairs, data["pair_names"], aux, transfer_dirs=tdirs,
                             verbose=verbose)
    out = train_v8(data["X"], data["y_soft"], y_aux, data["soft_cols"], aux_weight,
                   epochs=epochs, lr=lr, test_size=test_size, seed=seed,
                   verbose=verbose)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out["model"].state_dict(), save_path)
    # training record: what eval.py --dataset synthetic needs to rebuild the held-out
    # split, plus everything that makes this V8 run reproducible
    save_path.with_suffix(".json").write_text(json.dumps({
        "arch": "v8", "pairs": list(pairs), "pair": None, "results_dir": None,
        "mix_real": None, "transfer_dirs": [d.name for d in tdirs],
        "test_size": test_size, "seed": CFG["seed"] if seed is None else seed,
        "epochs": epochs, "lr": lr, "class_weights": None,
        "aux_weight": aux_weight, "aux_targets": list(aux),
        "n_pairs": len(data["pair_names"]), "best_acc": float(out["best_acc"]),
        "seq_len": int(data["X"].shape[-1]), "run": CFG["run"]["name"],
        "history": out["history"],
    }, indent=2))
    if verbose:
        print(f"wrote -> {save_path} (+ .json training record)")
    out["weights"] = save_path
    out["n_pairs"] = len(data["pair_names"])
    return out
