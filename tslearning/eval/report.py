"""
The distillation tables and the consistency checks run before any paper table is built.

A FORMATTER, like rules/tables.py: it reads result files that other steps computed and
lays them out. Nothing here recomputes a number.

  distillation.{csv,md,tex}   fidelity per pair vs the majority-class baseline
                              (from results/fidelity_real_<other side>.csv, `eval.py --metric fidelity`)
  speedup.{csv,md,tex}        reasoner vs classifier seconds per window-pair
                              (from results/reasoner_timing.csv, timing_table.py)

`checks()` returns the problems that would make a paper table misleading: partial
fidelity scores, mixed classifier weights, verification without the engine, classifier
folders out of step with the Python labels, LLM runs or rule tables older than the labels.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from tslearning.config import CFG, PATHS


def _pairs() -> list:
    return list((CFG.get("paper") or {}).get("pairs") or [])


def _other(pair: str) -> str:
    from tslearning.datasets.pairs import resolve_pair
    return resolve_pair(pair)[1].lower()


def fidelity_file(pair: str, results=None) -> Path:
    return Path(results or PATHS["results"]) / f"fidelity_real_{_other(pair)}.csv"


def _read(path: Path):
    return pd.read_csv(path) if path.exists() else None


# ══════════════════════════════════════════════════════════════════════════════
#  Checks
# ══════════════════════════════════════════════════════════════════════════════

def checks(results=None) -> "list[tuple[str, str]]":
    """[(level, message)] with level 'ERROR' (the table would be wrong) or 'WARNING'."""
    from tslearning.distill.classifier import default_weights
    from tslearning.distill.inference import stale_copies

    res = Path(results or PATHS["results"])
    out = []
    fid_pairs = list(((CFG.get("paper") or {}).get("fidelity") or {}))
    weights_seen = {}

    # fidelity: complete, and on known weights
    for pair in fid_pairs:
        df = _read(fidelity_file(pair, res))
        if df is None:
            out.append(("WARNING", f"no fidelity file for {pair} ({fidelity_file(pair, res).name})"))
            continue
        r = df.iloc[0]
        if "n_expected" not in df.columns or "weights" not in df.columns:
            out.append(("WARNING", f"{fidelity_file(pair, res).name} predates the n_expected / "
                                   f"weights columns — rerun step 13"))
        elif int(r["n_windows"]) != int(r["n_expected"]):
            out.append(("ERROR", f"{fidelity_file(pair, res).name}: scored {int(r['n_windows'])} "
                                 f"of {int(r['n_expected'])} test windows — finish step 12"))
        if "weights" in df.columns:
            weights_seen[f"fidelity {pair}"] = str(r["weights"])

    # timing: on the same weights
    ct = _read(res / "classifier_timing.csv")
    if ct is not None and "weights" in ct.columns:
        weights_seen["classifier timing"] = str(ct["weights"].iloc[0])
    elif ct is not None:
        out.append(("WARNING", "classifier_timing.csv predates the weights column — rerun it"))
    expected = default_weights().name
    if weights_seen and set(weights_seen.values()) != {expected}:
        seen = ", ".join(f"{k}: {v}" for k, v in weights_seen.items())
        out.append(("ERROR", f"classifier weights differ from the config's ({expected}): {seen}"))

    # verification: engine numbers present
    for name in ("verification.csv", "verification_classifier.csv"):
        v = _read(res / name)
        if v is None:
            out.append(("WARNING", f"{name} missing — rules not verified"))
        elif "vadalog_confidence" not in v.columns or v["vadalog_confidence"].isna().all():
            out.append(("WARNING", f"{name}: not verified by the engine (no vadalog numbers) — "
                                   f"|Δ| in the tables compares Python with Python"))

    # labels vs everything derived from them
    for pair in _pairs():
        stale = stale_copies(pair)
        if stale:
            out.append(("WARNING", f"classifier folder for {pair} out of date ({len(stale)} "
                                   f"files) — rerun infer.py -p {pair}"))
        d = PATHS["data_processed"] / "real_pairs_regime_py" / pair
        csvs = list(d.glob("*.csv")) if d.exists() else []
        if not csvs:
            continue
        newest = max(c.stat().st_mtime for c in csvs)
        for name in ("rules.csv", "rules_classifier.csv"):
            f = res / name
            if f.exists() and f.stat().st_mtime < newest:
                out.append(("WARNING", f"{name} is older than the {pair} labels — rerun step 8"))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Tables
# ══════════════════════════════════════════════════════════════════════════════

def distillation_table(results=None) -> "tuple[pd.DataFrame, str]":
    """One row per fidelity pair: classifier vs majority-class baseline."""
    from tslearning.rules.tables import pair_label, _cell

    res = Path(results or PATHS["results"])
    rows, weights = [], set()
    for pair in ((CFG.get("paper") or {}).get("fidelity") or {}):
        df = _read(fidelity_file(pair, res))
        if df is None:
            continue
        r = df.iloc[0]
        n = int(r["n_windows"])
        n_exp = int(r["n_expected"]) if "n_expected" in df.columns else n
        rows.append({
            "Pair": pair_label(pair),
            "n": str(n) if n == n_exp else f"{n}/{n_exp}",
            "Accuracy": _cell(r["accuracy"]), "Baseline acc.": _cell(r.get("baseline_accuracy")),
            "Macro-F1": _cell(r["macro_f1"]), "Baseline F1": _cell(r.get("baseline_macro_f1")),
            "Majority class": {"isPositiveCorrelation": "positive",
                               "isNegativeCorrelation": "negative",
                               "isNoCorrelation": "none"}.get(
                str(r.get("majority_class")), "—"),
        })
        if "weights" in df.columns:
            weights.add(str(r["weights"]))
    w = ", ".join(sorted(weights)) or "unrecorded weights"
    # the subsampling of each pair's test windows (paper.fidelity steps), in words
    from collections import defaultdict
    by_step = defaultdict(list)
    for pair, spec in ((CFG.get("paper") or {}).get("fidelity") or {}).items():
        if pair_label(pair) in {r["Pair"] for r in rows}:
            by_step[int(spec["step"])].append(pair_label(pair).split("– ")[-1])
    ordinal = lambda k: f"{k}th"
    steps = sorted(by_step, key=lambda k: -len(by_step[k]))
    every = (f"every {ordinal(steps[0])} window"
             + "".join(f"; {ordinal(k)} for {', '.join(by_step[k])}" for k in steps[1:])
             if steps else "test windows")
    caption = ("Classifier vs reasoner regimes on the real test windows (<TEST>). "
               "\\emph{n}: windows (" + every + "). \\emph{Macro-F1} weighs the three "
               "regimes equally. Baseline: majority class. Classifier: \\texttt{"
               + w.replace("_", "\\_") + "}.")
    return pd.DataFrame(rows), caption


def speedup_table(results=None) -> "tuple[pd.DataFrame, str]":
    """Reasoner vs classifier per window-pair, from timing_table.py's output."""
    from tslearning.rules.tables import _cell

    res = Path(results or PATHS["results"])
    rt = _read(res / "reasoner_timing.csv")
    if rt is None:
        return pd.DataFrame(), ""
    rows = []
    for _, r in rt.iterrows():
        clf = r.get("classifier_s_per_window_pair")
        rows.append({
            "Run": "all runs" if r["folder"] == "ALL" else str(r["folder"]).lstrip("_"),
            "Window-pairs": str(int(r["window_pairs"])) if "window_pairs" in rt.columns else "—",
            "Reasoner (s)": _cell(r["s_per_window_pair"], 1),
            # milliseconds: 0.064 ms reads better than 6.4e-05 s
            "Classifier (ms)": "—" if pd.isna(clf) else f"{1000 * float(clf):.3f}",
            "Speedup": "—" if pd.isna(r.get("speedup")) else f"{int(r['speedup']):,}×",
        })
    w = str(rt["classifier_weights"].iloc[0]) if "classifier_weights" in rt.columns else "?"
    caption = ("Labelling time per window-pair, reasoner vs classifier. \\emph{timing}: "
               "dedicated run, warm-up discarded; other rows: the reasoner's labeling "
               "runs, timed as they ran. Classifier: \\texttt{"
               + w.replace("_", "\\_") + "}.")
    return pd.DataFrame(rows), caption


def unscaled(tex: str) -> str:
    r"""Drop to_latex's \resizebox{\textwidth}: on a small table it would ENLARGE the
    text to the full page width."""
    return tex.replace("\\resizebox{\\textwidth}{!}{%\n", "", 1) \
              .replace("\\end{tabular}\n}", "\\end{tabular}", 1)


def speedup_pub_table(results=None) -> "tuple[pd.DataFrame, str]":
    """speedup with two rows only: the clean timing run and the all-runs average."""
    df, _ = speedup_table(results)
    if df.empty:
        return df, ""
    df = df[df["Run"].isin(["timing", "all runs"])].reset_index(drop=True)
    res = Path(results or PATHS["results"])
    rt = _read(res / "reasoner_timing.csv")
    w = str(rt["classifier_weights"].iloc[0]) if "classifier_weights" in rt.columns else "?"
    caption = ("Labelling time per window-pair, reasoner vs classifier. \\emph{timing}: "
               "dedicated run, warm-up discarded; \\emph{all runs}: average over all "
               "labeling runs. Classifier: \\texttt{" + w.replace("_", "\\_") + "}.")
    return df, caption


def write_tables(results=None, fmt: str = "both", verbose: bool = True) -> list:
    """Write distillation.*, speedup.* and speedup_pub.*; returns the paths written."""
    from tslearning.rules.tables import to_latex, to_markdown

    res = Path(results or PATHS["results"])
    written = []
    for name, (df, cap) in (("distillation", distillation_table(res)),
                            ("speedup", speedup_table(res)),
                            ("speedup_pub", speedup_pub_table(res))):
        if df.empty:
            if verbose:
                print(f"  {name}: no input files — skipped")
            continue
        df.to_csv(res / f"{name}.csv", index=False)
        written.append(res / f"{name}.csv")
        if fmt in ("md", "both"):
            (res / f"{name}.md").write_text(to_markdown(df) + "\n")
            written.append(res / f"{name}.md")
        if fmt in ("tex", "both"):
            tex = to_latex(df, cap, f"fig:{name}")
            if name == "speedup_pub":
                tex = unscaled(tex)
            (res / f"{name}.tex").write_text(tex + "\n")
            written.append(res / f"{name}.tex")
        if verbose:
            print(f"\n### {name}\n\n{to_markdown(df)}")
    return written


def _tabular(tex: str) -> str:
    """Just the tabular of a to_latex() figure."""
    a = tex.index("\\begin{tabular}")
    b = tex.index("\\end{tabular}") + len("\\end{tabular}")
    return tex[a:b]


def classifier_side_by_side(dist: pd.DataFrame, speed: pd.DataFrame, weights: str = "?",
                            label: str = "fig:classifier-pub") -> str:
    """A NEW figure: classifier fidelity (a) and speed (b) side by side.

    Built from the same data as distillation.tex / speedup_pub.tex, which are left as
    they are. (a) keeps the fidelity table's column headers; only the asset is named (every
    pair is the S&P 500 against it) and the Majority-class column is left out. (b) is the
    speed table transposed (one column per run), which is narrower. Each panel keeps its
    natural size and is shrunk only if it would not fit — never stretched.
    """
    from tslearning.rules.tables import periods, to_latex

    short = {"US Treasury 20y+ (TLT)": "TLT", "US 10y zero-coupon (GSW)": "10y ZC",
             "US 10y Treasury": "UST 10y"}
    d = dist.drop(columns=[c for c in ["Majority class"] if c in dist.columns]).copy()
    d["Pair"] = d["Pair"].astype(str).str.split("– ").str[-1].map(lambda a: short.get(a, a))
    d = d.rename(columns={"Pair": "S&P 500 vs"})
    sp = speed.set_index("Run").T.reset_index().rename(columns={"index": " "})
    t1, t2 = _tabular(to_latex(d)), _tabular(to_latex(sp))

    def panel(tab, width, sub):
        return "\n".join([
            f"\\begin{{minipage}}[t]{{{width}\\textwidth}}\\centering\\small"
            "\\setlength{\\tabcolsep}{4pt}",
            "{\\setbox0=\\hbox{%", tab + "}%",
            "\\ifdim\\wd0>\\linewidth\\resizebox{\\linewidth}{!}{\\box0}"
            "\\else\\leavevmode\\box0\\fi\\par}",
            f"\\vspace{{2pt}}{{\\footnotesize {sub}}}",
            "\\end{minipage}"])

    w = weights.replace("_", "\\_")
    cap = ("Classifier \\texttt{" + w + "}. (a) Agreement with the reasoner on the real "
           "test windows (<TEST>); \\emph{Macro-F1} weighs the regimes equally; "
           "baselines: majority class. (b) Time per window-pair; \\emph{timing}: "
           "dedicated run; \\emph{all runs}: average of all labeling runs.")
    return "\n".join([
        "\\begin{figure}[t]", "\\centering",
        panel(t1, "0.64", "(a) Fidelity"), "\\hfill",
        panel(t2, "0.34", "(b) Speed"),
        f"\\caption{{\\small{{{periods(cap)}}}}}", f"\\label{{{label}}}",
        "\\end{figure}"])
