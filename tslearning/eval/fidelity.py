"""
Fidelity — does the student agree with the teacher?

This is the classifier's report card, and it is blunt about why it
matters: every accuracy number the notebooks currently print is training fit.
Fidelity is agreement with the REASONER on held-out data — and the honest version
of it is on held-out REAL windows, not synthetic ones, because synthetic-to-real is
exactly where the student was observed to collapse (predicting "no correlation" for
all 1962 real windows).

Two datasets, one function:
  synthetic  held-out slice of the engineered transfer set        -> cheap, optimistic
  real       reasoner-labelled real windows from the test period  -> the number to report

The real variant needs a reasoner run over a SUBSAMPLED real test set (every Nth
window), because labelling all of them takes days. That run doubles as the source of
real training labels for fixing the collapse, which is why
prep_real_pairs.py and this share the same batch machinery.

────────────────────────────────────────────────────────────────────────────────
TWO ROUTES TO A REAL TEST SET, AND WHY BOTH EXIST
────────────────────────────────────────────────────────────────────────────────

`build_real_test_set`      sp500 x bitcoin ONLY, from aligned_windows_real.pkl.
                           The original path; the pickle holds no other pair.
`build_real_pair_test_set` ANY pair, aligned on the fly from the transformed series,
                           using the `{side}_real_{t}` naming that build_real_batches
                           writes — so the same batches feed the reasoner and the
                           student.

The second exists because measuring fidelity on sp500-bitcoin alone cannot test the
class that matters. Over 2022-2024 that pair has ONE negative-correlation window in 71,
so `isNegativeCorrelation` gets precision/recall of 0.000 on a support of 1, and
macro-F1 is meaningless. sp500-gold carries 14.2% negative-correlation days against
bitcoin's 4.8% — it is the pair on which the negative class can actually be scored,
and the negative class is the one template T1 depends on.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tslearning.config import CFG, PATHS


def _metrics(true, pred, labels, target_names) -> dict:
    from sklearn.metrics import (classification_report, confusion_matrix, f1_score)

    return {
        "accuracy": float((np.asarray(pred) == np.asarray(true)).mean()),
        "macro_f1": float(f1_score(true, pred, average="macro", labels=labels,
                                   zero_division=0)),
        "report": classification_report(true, pred, labels=labels,
                                        target_names=target_names, digits=3,
                                        zero_division=0),
        "confusion": confusion_matrix(true, pred, labels=labels),
        "n": int(len(true)),
    }


def majority_baseline(y_soft, soft_cols) -> dict:
    """Scores of a constant predictor that always outputs the most frequent teacher class.

    The classifier's accuracy/macro-F1 only mean something relative to these.
    """
    true = np.asarray(y_soft).argmax(1)
    maj = int(np.bincount(true, minlength=len(soft_cols)).argmax())
    m = _metrics(true, np.full_like(true, maj), list(range(len(soft_cols))), soft_cols)
    return {"majority_class": soft_cols[maj], "baseline_accuracy": m["accuracy"],
            "baseline_macro_f1": m["macro_f1"]}


def evaluate_regime(model, X, y_soft, soft_cols, title: str = "eval",
                    verbose: bool = True) -> dict:
    """Compare the student's argmax against the reasoner's one-hot regime labels."""
    from tslearning.distill.classifier import predict_regime

    pred = predict_regime(model, X)
    true = np.asarray(y_soft).argmax(1)
    m = _metrics(true, pred, list(range(len(soft_cols))), soft_cols)
    if verbose:
        print(f"===== {title} · correlation regime (n={m['n']}) =====")
        print(m["report"])
        print("confusion (rows=true, cols=pred):\n", m["confusion"])
        print(f"accuracy {m['accuracy']:.3f} | macro-F1 {m['macro_f1']:.3f}")
    return m


# ══════════════════════════════════════════════════════════════════════════════
#  Real held-out test set
# ══════════════════════════════════════════════════════════════════════════════

def build_real_test_set(step: int = 10, aligned=None, verbose: bool = True) -> dict:
    """Slice the real timeline to the config's TEST dates, then subsample.

    `step` is the subsampling stride: the reasoner is too slow to label every test
    window, so every `step`-th one is taken. step=10 over 2022-2024 gives ~70
    windows ~= 3 batches, which is a few hours rather than days.
    """
    import pickle

    data_dir = PATHS["data_processed"]
    if aligned is None:
        with open(data_dir / "aligned_windows_real.pkl", "rb") as f:
            aligned = pickle.load(f)

    a = np.asarray(aligned["sp500_real"]["windows"], np.float32)
    b = np.asarray(aligned["bitcoin_real_paired"]["windows"], np.float32)
    tp2date = aligned["timepoint_to_date"]
    dates = pd.to_datetime([tp2date[i] for i in range(len(a))])

    lo, hi = CFG["split"]["test"]
    mask = (dates >= lo) & (dates <= hi)
    a_test, b_test = a[mask], b[mask]
    tp_test = np.arange(len(a))[mask]

    a_sub, b_sub, tp_sub = a_test[::step], b_test[::step], tp_test[::step]
    if verbose:
        print(f"real test windows {lo}..{hi}: {len(a_test)} of {len(a)}; "
              f"subsampled every {step} -> {len(a_sub)} "
              f"({(len(a_sub)+24)//25} batches of 25)")
    return {"windows_a": a_sub, "windows_b": b_sub, "timepoints": tp_sub,
            "n_before_subsample": int(len(a_test))}


def build_real_pair_test_set(pair: str, step: int = 10, lo=None, hi=None,
                             verbose: bool = True) -> dict:
    """Test-period windows for ANY pair, named as build_real_batches names them.

    Aligns the two transformed series on their common trading days, slides windows,
    then subsamples every `step`-th one. Window ids are `{side}_real_{t}` where `t` is
    the index into the ALIGNED day list — identical to what build_real_batches writes,
    so the reasoner's output and the student's inputs key off the same names.

    Note the difference from `build_real_test_set`, which slices a pre-built pickle and
    subsamples AFTER restricting to the test period. Here the alignment happens first
    and `t` is an index into the test-period days only, so the two are not
    interchangeable — do not mix their outputs in one results folder.
    """
    from tslearning.datasets.pairs import describe, load_series

    d = describe(pair)
    lo = lo or CFG["split"]["test"][0]
    hi = hi or CFG["split"]["test"][1]
    seq_len = d["seq_len"]

    series = load_series([d["a"], d["b"]])
    idx = None
    for s in (series[d["a"]], series[d["b"]]):
        i = s.dropna().sort_index().loc[lo:hi].index
        idx = i if idx is None else idx.intersection(i)
    idx = idx.sort_values()

    a = series[d["a"]].loc[idx].values.astype(np.float32)
    b = series[d["b"]].loc[idx].values.astype(np.float32)
    tps = list(range(0, len(idx) - seq_len + 1, step))

    wa = np.array([a[t:t + seq_len] for t in tps], dtype=np.float32)
    wb = np.array([b[t:t + seq_len] for t in tps], dtype=np.float32)
    if verbose:
        print(f"{pair} real test {lo}..{hi}: {len(idx)} aligned days -> {len(tps)} "
              f"windows (every {step}) ~= {(len(tps) + 24) // 25} batches of 25")
    return {"pair": d["slug"], "windows_a": wa, "windows_b": wb, "timepoints": tps,
            "base": d["a_key"], "other": d["b_key"], "seq_len": seq_len,
            "series": {d["a_key"]: series[d["a"]], d["b_key"]: series[d["b"]]},
            "lo": lo, "hi": hi, "dates": idx}


def label_real_pair_test_set(pair: str, step: int = 10, batch_size: int = 25,
                             batch_dir=None, result_dir=None, run_reasoner: bool = True,
                             lo=None, hi=None, verbose: bool = True) -> dict:
    """Write the test-period batches for `pair` and label them with the reasoner.

    Uses a PAIR-SPECIFIC result folder by default. That is not tidiness: every loader
    globs `{predicate}_*.csv`, so two pairs sharing one folder silently merge into each
    other's targets.
    """
    from tslearning.labeling import reasoner_client
    from tslearning.labeling.transfer_set import build_real_batches

    data_dir = PATHS["data_processed"]
    ts = build_real_pair_test_set(pair, step=step, lo=lo, hi=hi, verbose=verbose)
    slug = ts["pair"]
    batch_dir = Path(batch_dir or (data_dir / f"real_test_{slug}_batches"))
    result_dir = Path(result_dir or (data_dir / f"real_test_{slug}"))
    result_dir.mkdir(parents=True, exist_ok=True)

    build_real_batches(slug, ts["series"], ts["other"], ts["lo"], ts["hi"],
                       stride=step, batch_size=batch_size, out_dir=batch_dir,
                       seq_len=ts["seq_len"], base=ts["base"], verbose=verbose)

    outcome = None
    if run_reasoner:
        outcome = reasoner_client.run_pair_batches(slug, batch_dir, result_dir,
                                                   verbose=verbose)
    elif verbose:
        print(f"wrote batches to {batch_dir}; re-run without --no-reasoner to label")
    return {"batch_dir": batch_dir, "result_dir": result_dir, "test_set": ts,
            "reasoner": outcome}


def label_real_test_set(step: int = 10, batch_size: int = 25, result_dir=None,
                        run_reasoner: bool = True, verbose: bool = True) -> dict:
    """Write the real-test batch CSVs and (optionally) run the reasoner over them.

    Resumable — completed batches are skipped, so an interrupted run just continues.
    """
    from tslearning.labeling import reasoner_client
    from tslearning.labeling.transfer_set import build_pairs_csv

    data_dir = PATHS["data_processed"]
    result_dir = Path(result_dir or (data_dir / "real_test_results"))
    result_dir.mkdir(parents=True, exist_ok=True)

    ts = build_real_test_set(step=step, verbose=verbose)
    a, b = ts["windows_a"], ts["windows_b"]

    paths, tags = [], []
    n_batches = (len(a) + batch_size - 1) // batch_size
    for i in range(n_batches):
        s, e = i * batch_size, min((i + 1) * batch_size, len(a))
        p = data_dir / f"real_test_batch_{i:03d}.csv"
        build_pairs_csv(a[s:e], b[s:e], p, start_global=s)
        paths.append(p)
        tags.append(str(i))

    outcome = None
    if run_reasoner:
        outcome = reasoner_client.run_batches(paths, result_dir, tags=tags,
                                              verbose=verbose)
    elif verbose:
        print(f"wrote {len(paths)} batch CSVs; re-run without --no-reasoner to label them")

    return {"batches": paths, "result_dir": result_dir, "test_set": ts,
            "reasoner": outcome}


def _warn_if_degenerate_targets(y_soft, soft_cols, results_dir, dataset: str) -> None:
    """Shout if the TEACHER's labels are degenerate, before anyone reads the score.

    2026-09-03: bitcoin fidelity was measured against `real_test_results/`, which still
    holds labels from the ABSOLUTE-cut `preLabeling.vada` (same% > 55 / diff% < 45). Those
    bars are unreachable on real daily returns, so the folder's target distribution is
    4 / 0 / 67 — `isNegativeCorrelation` has zero support. Two students scored against it:
    the pre-fix one got 0.944 (= the majority-class baseline exactly, i.e. nothing) and the
    dominance-retrained one got 0.408, punished for NOT reproducing a broken teacher.

    Nothing in the pipeline complained. `build_targets` reads whatever regime CSVs are in
    the folder, and eval.py's default `--results` points at the un-relabelled one, so the
    run succeeds and prints a plausible number. Hence this check.
    """
    import numpy as np

    n = len(y_soft)
    if not n:
        return
    counts = {c: int(v) for c, v in zip(soft_cols, np.asarray(y_soft).sum(0))}
    empty = [c for c, v in counts.items() if v == 0]
    top = max(counts.values()) / n if n else 0.0
    if not empty and top <= 0.90:
        return

    dist = "  ".join(f"{c} {v}" for c, v in counts.items())
    print("\n" + "!" * 78)
    print("DEGENERATE TEACHER LABELS — the fidelity number below is not meaningful.")
    print(f"  targets from : {results_dir}")
    print(f"  distribution : {dist}   (n={n})")
    if empty:
        print(f"  zero support : {', '.join(empty)}")
    if top > 0.90:
        print(f"  majority class holds {top:.1%} — a constant predictor scores that much")
    print("\n  Most likely cause: these labels came from the ABSOLUTE-cut preLabeling")
    print("  rules, which cannot fire isNegativeCorrelation on real daily returns.")
    print("  Fix (no reasoner needed, seconds):")
    print(f"      python scripts/relabel.py --results {results_dir}")
    print(f"      python scripts/eval.py --metric fidelity --dataset {dataset} \\")
    print(f"             --results {results_dir}_dominance ...")
    print("!" * 78 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage runner
# ══════════════════════════════════════════════════════════════════════════════

# v7b predates the training sidecar: this is how the README trains it (step 5)
_V7B_TRAIN_PAIRS = ["sp500_bitcoin", "sp500_gold", "sp500_nasdaq"]


def _training_setup(weights) -> dict:
    """What the weights were trained on: the sidecar <weights>.json written by
    distill.train.run, or a best guess for weights trained before it existed."""
    import json
    side = Path(weights).with_suffix(".json")
    if side.exists():
        return {**json.loads(side.read_text()), "from": "sidecar"}
    name = Path(weights).name
    if "v7b" in name:
        return {"pairs": _V7B_TRAIN_PAIRS, "test_size": 0.2, "from": "assumed (v7b training recipe)"}
    # v7 was trained on the DOMINANCE labels (2026-08-05). batched_results holds the same
    # 525 windows with the OLD absolute-cut labels — grading v7 against those gave 0.30.
    return {"pairs": None,
            "results_dir": str(PATHS["data_processed"] / "transfer_dominance"),
            "test_size": 0.2, "from": "assumed (v7's flat set, dominance labels)"}


def _synthetic_heldout(results_dir=None, train_pairs=None, test_size: float = 0.2,
                       transfer_dirs=None, seed: "int | None" = None):
    """The held-out part of the classifier's OWN training split.

    Same data and the same seeded, stratified split as distill.train.train(), so the
    windows scored here are exactly the ones the classifier never trained on:

      results_dir=None   the transfer folders of `train_pairs` (default paper.pairs),
                         concatenated in that order — how v7b and later are trained
      results_dir=<dir>  the flat single-pair layout (v7: data/processed/batched_results)

    The split must match training: same pairs, same order, same test_size (distill.py's
    --test-size, default 0.2) and config seed.
    """
    from sklearn.model_selection import train_test_split
    from tslearning.distill import train as T

    if results_dir is None:
        train_pairs = list(train_pairs or (CFG.get("paper") or {}).get("pairs") or [])
        dirs = ([PATHS["data_processed"] / d for d in transfer_dirs]
                if transfer_dirs else None)
        data = T.load_multi_pair_training_data(pairs=train_pairs, transfer_dirs=dirs,
                                               verbose=False)
        what = ("transfer sets " + ",".join(transfer_dirs) if transfer_dirs
                else f"transfer sets {','.join(train_pairs)}")
    else:
        data = T.load_training_data(results_dir=results_dir, verbose=False)
        what = f"flat set {Path(results_dir).name}"
    y_idx = np.asarray(data["y_soft"]).argmax(1)
    _tr, te = train_test_split(np.arange(len(y_idx)), test_size=test_size,
                               random_state=CFG["seed"] if seed is None else seed,
                               stratify=y_idx)
    return {"X": np.asarray(data["X"], np.float32)[te], "y_soft": data["y_soft"][te],
            "soft_cols": data["soft_cols"], "n": len(te), "what": what}


def run(dataset: str = "real", weights=None, results_dir=None, step: int = 10,
        save=None, pair: "str | None" = None, train_pairs=None,
        test_size: float = 0.2, verbose: bool = True) -> dict:
    """Compute the fidelity row for one dataset (see `_run`).

    With `save`, the full printed report (per-class precision/recall, confusion matrix,
    baseline, training setup) is written next to the CSV as `<save stem>_report.txt` —
    the CSV only holds accuracy and macro-F1. The report is captured whether or not
    `verbose` echoes it to the terminal.
    """
    import contextlib
    import io
    import sys
    from datetime import datetime

    class _Tee(io.TextIOBase):
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                st.write(s)
            return len(s)

        def flush(self):
            for st in self.streams:
                st.flush()

    buf = io.StringIO()
    sink = _Tee(sys.stdout, buf) if verbose else buf
    with contextlib.redirect_stdout(sink):
        out = _run(dataset, weights, results_dir, step, save, pair, train_pairs,
                   test_size, verbose=True)
    if save:
        rep = Path(save).with_name(Path(save).stem + "_report.txt")
        rep.parent.mkdir(parents=True, exist_ok=True)
        from tslearning.config import REPO_ROOT
        text = (f"# written by tslearning.eval.fidelity.run, "
                f"{datetime.now():%Y-%m-%d %H:%M}\n"
                f"# dataset={dataset} pair={pair} step={step} "
                f"results={results_dir} csv={Path(save).name}\n\n" + buf.getvalue())
        # paths relative to the repository: no personal folders in a shared file
        text = text.replace(str(REPO_ROOT) + "/", "").replace(str(REPO_ROOT), ".")
        rep.write_text(text)
        if verbose:
            print(f"wrote -> {rep}")
    return out


def valid_set(pair: str = "sp500_gold", step: int = 5, seq_len: int = 50) -> dict:
    """VALIDATION set: real windows from the TRAINING years (config split.train),
    labelled by the exact Python labeller (same concordance and deadband as the reasoner;
    agreement 141/141 on gold, same%/diff% identical on every transfer set).

    The classifier never sees real data in training, so these windows are unseen by it,
    and they are not the 2022-2024 test period: choices made here (recipe, seeds, data)
    leave the test numbers honest. Window t covers days [t, t+seq_len), every `step`.
    """
    from tslearning.datasets.pairs import describe, load_series
    from tslearning.distill.classifier import build_timeline_input, regime_labels
    from tslearning.labeling.py_labeler import _aligned, compute_concordance, default_deadband

    d = describe(pair)
    lo, hi = CFG["split"]["train"]
    ser = load_series([d["a"], d["b"]])
    idx = _aligned({d["a"]: ser[d["a"]], d["b"]: ser[d["b"]]}, lo, hi)
    a = ser[d["a"]].loc[idx].to_numpy(float)
    b = ser[d["b"]].loc[idx].to_numpy(float)
    starts = list(range(0, len(a) - seq_len + 1, step))
    wa = [a[t:t + seq_len] for t in starts]
    wb = [b[t:t + seq_len] for t in starts]
    cols = regime_labels()
    db = default_deadband()
    y = np.zeros((len(starts), len(cols)), np.float32)
    for i, (x1, x2) in enumerate(zip(wa, wb)):
        s, df = compute_concordance(x1, x2, denom=seq_len)
        sc = s - df
        r = ("isPositiveCorrelation" if sc > db else
             "isNegativeCorrelation" if sc < -db else "isNoCorrelation")
        y[i, cols.index(r)] = 1.0
    return {"X": build_timeline_input(wa, wb), "y_soft": y, "soft_cols": cols,
            "n": len(starts), "period": [str(idx[0].date()), str(idx[-1].date())],
            "slug": d["slug"]}


def _neg_recall(m, soft_cols):
    i = soft_cols.index("isNegativeCorrelation")
    cm = np.asarray(m["confusion"])
    sup = int(cm[i].sum())
    return (float(cm[i, i] / sup) if sup else None), sup


def _run(dataset: str = "real", weights=None, results_dir=None, step: int = 10,
         save=None, pair: "str | None" = None, train_pairs=None,
         test_size: float = 0.2, verbose: bool = True) -> dict:
    """Compute the fidelity row for one dataset.

    dataset="valid"     : real windows of the TRAINING years, exact Python labels
                          (see valid_set) — for choosing, never the test period.

    dataset="real"      : reasoner labels vs the student on the same real windows.
                          `pair=None` uses the original sp500-bitcoin pickle route;
                          any other pair goes through build_real_pair_test_set.
    dataset="synthetic" : reasoner labels in batched_results/ vs the student on the
                          engineered transfer set.
    """
    from tslearning.distill.classifier import build_input, load_classifier
    from tslearning.labeling.predicate_targets import build_targets

    data_dir = PATHS["data_processed"]
    from tslearning.distill.classifier import default_weights
    weights = Path(weights) if weights else default_weights()
    model = load_classifier(weights)

    if dataset == "valid":
        v = valid_set(pair or "sp500_gold", step=step,
                      seq_len=CFG["windows"]["financial"]["length"])
        title = f"VALIDATION real {v['period'][0]}..{v['period'][1]} · {v['slug']}"
        m = evaluate_regime(model, v["X"], v["y_soft"], v["soft_cols"], title=title,
                            verbose=verbose)
        b = majority_baseline(v["y_soft"], v["soft_cols"])
        nr, ns = _neg_recall(m, v["soft_cols"])
        if verbose:
            print(f"majority-class baseline (always '{b['majority_class']}'): "
                  f"accuracy {b['baseline_accuracy']:.3f} | macro-F1 "
                  f"{b['baseline_macro_f1']:.3f}")
        cm = np.asarray(m["confusion"])
        per = {}
        for i, c in enumerate(v["soft_cols"]):
            short = {"isPositiveCorrelation": "pos", "isNegativeCorrelation": "neg",
                     "isNoCorrelation": "none"}.get(c, c)
            sup, called = int(cm[i].sum()), int(cm[:, i].sum())
            per[f"{short}_recall"] = float(cm[i, i] / sup) if sup else None
            per[f"{short}_precision"] = float(cm[i, i] / called) if called else None
            per[f"{short}_support"] = sup
            per[f"{short}_predicted"] = called
        row = {"dataset": "valid", "pair": v["slug"], "period": "..".join(v["period"]),
               "step": step, "n_windows": m["n"], "n_expected": v["n"],
               "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
               "neg_recall": nr, "neg_support": ns, **per,
               "confusion": cm.tolist(), **b, "weights": Path(weights).name}
        if save:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([row]).to_csv(save, index=False)
            if verbose:
                print(f"wrote -> {save}")
        return {"row": row, "metrics": m, "pair_names": []}
    side_a, side_b = "sp500", "btc"

    if dataset == "real" and pair and pair not in ("sp500_bitcoin",):
        # Any pair: windows named {side}_real_{t}, one results folder per pair.
        ts = build_real_pair_test_set(pair, step=step, verbose=verbose)
        slug, side_a, side_b = ts["pair"], ts["base"], ts["other"]
        results_dir = Path(results_dir or (data_dir / f"real_test_{slug}"))
        a, b, tps = ts["windows_a"], ts["windows_b"], ts["timepoints"]
        lookup = {f"{side_a}_real_{t}": a[i] for i, t in enumerate(tps)}
        lookup.update({f"{side_b}_real_{t}": b[i] for i, t in enumerate(tps)})
        title = (f"REAL test {ts['lo']}..{ts['hi']} · {slug}")
        n_expected = len(tps)
    elif dataset == "real":
        results_dir = Path(results_dir or (data_dir / "real_test_results"))
        ts = build_real_test_set(step=step, verbose=verbose)
        a, b = ts["windows_a"], ts["windows_b"]
        lookup = {f"sp500_synth_{i}": a[i] for i in range(len(a))}
        lookup.update({f"btc_synth_{i}": b[i] for i in range(len(b))})
        title = f"REAL test {CFG['split']['test'][0]}..{CFG['split']['test'][1]}"
        n_expected = min(len(a), len(b))
    elif dataset == "synthetic":
        # the held-out part of the classifier's own training split — not v7's old flat
        # sp500-bitcoin set, which is partly another distribution for a multi-pair model
        from tslearning.distill import ensemble
        if ensemble.is_ensemble(weights):
            raise ValueError(
                "synthetic fidelity is not defined for an ensemble: each member held out a "
                "different part of the transfer set, so no window is unseen by all of them. "
                "Run --dataset synthetic on each member and report mean ± sd.")
        setup = _training_setup(weights)
        from_setup = results_dir is None and train_pairs is None
        if from_setup:
            # no override: reproduce the classifier's own training split
            train_pairs = setup.get("pairs")
            if not train_pairs and setup.get("pair") and not setup.get("results_dir"):
                train_pairs = [setup["pair"]]   # single-pair model on transfer_<pair>*/
            results_dir = None if train_pairs else setup.get("results_dir")
            test_size = setup.get("test_size", test_size)
            if verbose:
                print(f"training setup ({setup['from']}): "
                      f"{train_pairs or results_dir}, test_size {test_size}")
            if setup["from"] != "sidecar":
                print("note: these weights have no training record (<weights>.json); the "
                      "held-out split is reproduced from an assumption and may not match "
                      "the one used in training exactly")
        h = _synthetic_heldout(results_dir, train_pairs, test_size,
                               transfer_dirs=(setup.get("transfer_dirs")
                                              if from_setup else None),
                               seed=setup.get("seed") if from_setup else None)
        title = f"held-out synthetic · {h['what']}"
        m = evaluate_regime(model, h["X"], h["y_soft"], h["soft_cols"], title=title,
                            verbose=verbose)
        b = majority_baseline(h["y_soft"], h["soft_cols"])
        row = {"dataset": dataset, "n_windows": m["n"], "n_expected": h["n"],
               "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], **b,
               "weights": Path(weights).name, "heldout_of": h["what"]}
        if save:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([row]).to_csv(save, index=False)
            if verbose:
                print(f"wrote -> {save}")
        return {"row": row, "metrics": m, "pair_names": []}
    else:
        raise ValueError(f"dataset must be 'real' or 'synthetic', got {dataset!r}")

    # side_a/side_b tell build_targets which names count as a genuine cross-series
    # pair; they differ per pair ("sp500"/"gold"), so they must be threaded through.
    tgt = build_targets(results_dir, side_a=side_a, side_b=side_b)
    if not tgt["pair_names"]:
        raise FileNotFoundError(
            f"no reasoner output in {results_dir}. For the real row, run "
            "`eval.py --metric fidelity --dataset real --label-first` to build and "
            "label the subsampled real test set"
            + (f" (pair {pair})." if pair else "."))

    keep = [i for i, (x, y) in enumerate(tgt["pair_names"])
            if x in lookup and y in lookup]
    pair_names = [tgt["pair_names"][i] for i in keep]
    X = build_input(pair_names, lookup, standardize=True)
    y_soft = tgt["y_soft"][keep]

    _warn_if_degenerate_targets(y_soft, tgt["soft_cols"], results_dir, dataset)
    # Windows without reasoner labels are dropped by the matching above. That is not an
    # error (a partial set can be a deliberate quick look) but it must not look like a
    # full-set score: warn, and record the expected count in the row.
    if len(pair_names) < n_expected:
        print(f"WARNING: only {len(pair_names)} of {n_expected} test windows have reasoner "
              f"labels in {results_dir} — the score covers only those. Finish labelling "
              f"(REPRODUCIBILITY.md §2) for the full set.")

    m = evaluate_regime(model, X, y_soft, tgt["soft_cols"], title=title, verbose=verbose)
    b = majority_baseline(y_soft, tgt["soft_cols"])
    if verbose:
        print(f"majority-class baseline (always '{b['majority_class']}'): "
              f"accuracy {b['baseline_accuracy']:.3f} | macro-F1 {b['baseline_macro_f1']:.3f}")

    row = {"dataset": dataset, "n_windows": m["n"], "n_expected": n_expected,
           "accuracy": m["accuracy"],
           "macro_f1": m["macro_f1"], **b, "weights": Path(weights).name}
    if save:
        out = Path(save)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(out, index=False)
        if verbose:
            print(f"wrote -> {out}")
    return {"row": row, "metrics": m, "pair_names": pair_names}
