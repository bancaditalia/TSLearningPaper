"""
The rule pipeline on the simulated series (`data_simulation_econ tests/simulation/`).

Each dataset file holds `reps` simulated pairs, columns `rep, date, X, Y, split,
inflationHigh` (export_datasets.py). The file name says which phenomenon was planted
and how strongly: `T2_theta0.4_rho0.3_n2520.csv.gz` = contagion, theta 0.4, background
correlation 0.3, 2,520 days. theta = 0 is the null (nothing planted).

Every replication goes through the SAME code as a real pair labelled in Python
(prep_real_pairs.py --engine python, then generate_rules.py --score --ci):

  1. tau / qLow per series: fitted on the rep's train split (70% of days), or rolling
     (labels.rolling_days before each window, the train value when the history is
     shorter than labels.rolling_min_days);
  2. py_labeler.label_pair, window and cut-offs from the config (lags only when a T4
     template is asked for);
  3. T7: the file's inflationHigh column becomes inflationHigh("us") on the days it is 1,
     and market(simx, "us") ties it to the pair, as config `markets` does for sp500;
  4. generate.load_pair_facts, the template fitted on the train split, constants frozen,
     scored on the test split, bootstrap CI on the test lift (scoring.bootstrap_lift);
  5. tables._verdict, the paper's verdict. A rep DETECTS the phenomenon when the
     verdict is "significant" (lift >= 0.05 and the 95% CI excludes zero).

Not detected includes "not fitted" and "ineligible" reps, as in mc_econ.py; the share
of reps that could be scored is reported separately (testable_share).

The series are named simx (X, the S&P 500 role) and simy (Y, the second asset).
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from tslearning.config import CFG, PATHS, REPO_ROOT

SIM_DIR = REPO_ROOT / "data_simulation_econ tests" / "simulation"
DATASETS = SIM_DIR / "datasets"
X, Y, ECON = "simx", "simy", "us"
PAIR = f"{X}_{Y}"

# the rule templates that look for each planted phenomenon (forward and backward)
FAMILY = {
    "T1b": ["T1_safeHaven", "T1b_safeHaven_bw"],
    "T2": ["T2_contagion", "T2b_contagion_bw"],
    "T3b": ["T3_decoupled", "T3b_decoupled_bw"],
    "T4": ["T4_leadsUnderStress", "T4b_leadsUnderStress_bw"],
    "T7": ["T7_macroConditioned", "T7b_macroConditioned_bw"],
}
LEAD_PREDS = {"leadsPositively", "leadsNegatively", "noLead"}
_NAME = re.compile(r"^(?P<tpl>T\w+?)_theta(?P<theta>[0-9.]+)_rho(?P<rho0>[0-9.]+)"
                   r"_n(?P<n>\d+)\.csv(?:\.gz)?$")


def parse_name(path) -> dict:
    m = _NAME.match(Path(path).name)
    if not m:
        raise ValueError(f"{Path(path).name}: expected <T>_theta<t>_rho<r>_n<n>.csv.gz")
    return {"sim_template": m["tpl"], "theta": float(m["theta"]),
            "rho0": float(m["rho0"]), "n": int(m["n"])}


def find_files(names=None, folder=None) -> list:
    """Dataset files under `folder` and its subfolders (default: datasets/, i.e. the
    theta = 0 files and core/); `names` = file names or stems. A name found twice is
    read once."""
    folder = Path(folder or DATASETS)
    seen, files = set(), []
    for f in sorted(folder.rglob("T*_theta*_rho*_n*.csv*")):
        if f.name not in seen:
            seen.add(f.name); files.append(f)
    files.sort(key=lambda f: f.name)
    if names:
        want = {str(n).removesuffix(".gz").removesuffix(".csv") for n in names}
        files = [f for f in files if f.name.removesuffix(".gz").removesuffix(".csv") in want]
        found = {f.name.removesuffix(".gz").removesuffix(".csv") for f in files}
        if want - found:
            raise FileNotFoundError(f"not in {folder}: {sorted(want - found)}")
    if not files:
        raise FileNotFoundError(f"no simulated datasets in {folder}")
    return files


def templates_for(sim_template: str, which="matching") -> list:
    from tslearning.rules.templates import TEMPLATES
    if which in (None, "matching"):
        return FAMILY[sim_template]
    if which == "all":
        return sorted(TEMPLATES)
    unknown = [t for t in which if t not in TEMPLATES]
    if unknown:
        raise ValueError(f"unknown template(s) {unknown}. Known: {sorted(TEMPLATES)}")
    return list(which)


# ── thresholds ────────────────────────────────────────────────────────────────

def window_thresholds(r: np.ndarray, n_train: int, starts, mode: str):
    """(tau, qlow) per window start for one series. Same percentiles as
    transfer_set.fit_thresholds (train) and thresholds.rolling_values (rolling)."""
    vol_p = float(CFG["predicates"]["vol_percentile"])
    crash_p = float(CFG["predicates"]["crash_percentile"])
    tr = r[:n_train]
    tau_tr, q_tr = float(np.percentile(np.abs(tr), vol_p)), float(np.percentile(tr, crash_p))
    if mode == "train":
        return np.full(len(starts), tau_tr), np.full(len(starts), q_tr)
    lab = CFG.get("labels") or {}
    days = int(lab.get("rolling_days", 500))
    min_days = int(lab.get("rolling_min_days", 250))
    tau, q = [], []
    for p in starts:                              # returns before the window's first day
        past = r[max(0, p - days):p]
        if len(past) < min_days:
            tau.append(tau_tr); q.append(q_tr)
        else:
            tau.append(float(np.percentile(np.abs(past), vol_p)))
            q.append(float(np.percentile(past, crash_p)))
    return np.asarray(tau), np.asarray(q)


# ── one replication ───────────────────────────────────────────────────────────

def _day_runs(flags) -> list:
    """[start, end) day-index runs where flags is 1."""
    out, start = [], None
    for d, v in enumerate(flags):
        if v == 1 and start is None:
            start = d
        elif v != 1 and start is not None:
            out.append((start, d)); start = None
    if start is not None:
        out.append((start, len(flags)))
    return out


def label_rep(d: pd.DataFrame, out_root: Path, lags, thresholds: str) -> dict:
    """Label one replication into out_root/<PAIR>/, exactly as a real pair."""
    from tslearning.labeling.py_labeler import label_pair

    d = d.sort_values("date")
    idx = pd.DatetimeIndex(pd.to_datetime(d["date"]))
    x = pd.Series(d["X"].to_numpy(float), index=idx)
    y = pd.Series(d["Y"].to_numpy(float), index=idx)
    seq_len = int(CFG["windows"]["financial"]["length"])
    n_train = int((d["split"] == "train").sum())
    starts = list(range(0, len(idx) - seq_len + 1))
    ta, qa = window_thresholds(x.to_numpy(), n_train, starts, thresholds)
    tb, qb = window_thresholds(y.to_numpy(), n_train, starts, thresholds)
    s = label_pair(PAIR, x, y, Y, {}, {}, out_dir=out_root, lo=idx[0], hi=idx[-1],
                   stride=1, seq_len=seq_len, base_name=X, lags=lags or None,
                   window_thresholds=(ta, qa, tb, qb))

    infl = pd.to_numeric(d["inflationHigh"], errors="coerce")
    if infl.notna().any():
        runs = _day_runs(infl.fillna(0).astype(int).to_numpy())
        pd.DataFrame([{"X": f"{ECON}_real_{a}", "Start": a, "End": b} for a, b in runs],
                     columns=["X", "Start", "End"]).to_csv(
            out_root / PAIR / f"inflationHigh_{PAIR}.csv", index=False)
    return s


def score_rep(d: pd.DataFrame, out_root: Path, templates) -> list:
    """Fit each template on the train split, score on test, bootstrap, verdict."""
    from tslearning.rules import generate as G
    from tslearning.rules import scoring
    from tslearning.rules.fitting import W_GRID
    from tslearning.rules.rule import Rule
    from tslearning.rules.tables import _verdict
    from tslearning.rules.templates import TEMPLATES

    G.SOURCE_DIR_OVERRIDE["py"] = out_root
    facts = G.load_pair_facts(PAIR, "py")
    facts.intervals["market"] = [((X, ECON), facts.T0, facts.T1)]
    dates = pd.to_datetime(d["date"])
    is_tr = (d["split"] == "train").to_numpy()
    train = (dates[is_tr].min(), dates[is_tr].max())
    test = (dates[~is_tr].min(), dates[~is_tr].max())
    periods = scoring.split_facts(facts, train, test)

    rows = []
    for name in templates:
        spec = TEMPLATES[name]
        missing = [p for p in spec["needs"] if not facts.intervals.get(p)]
        if missing:
            rule = Rule.rejected(name, facts, f"missing predicates: {missing}")
        else:
            rule = (spec["fn"](periods["train"], w_grid=W_GRID)
                    or Rule.rejected(name, facts, "template returned nothing"))
        ci = {}
        if rule.accepted:
            rule.scores["train"] = scoring.score_rule(rule, periods["train"])
            rule.scores["test"] = scoring.score_rule(rule, periods["test"])
            if (rule.parts or {}).get("kind") == "conditional":
                ci = scoring.bootstrap_lift(rule, periods["test"])
        row = rule.to_row()
        row.update({"ci_lo": ci.get("ci_lo"), "ci_hi": ci.get("ci_hi"),
                    "ci_spans_zero": ci.get("spans_zero"),
                    "ci_degenerate": ci.get("degenerate"),
                    "ci_n_episodes": ci.get("n_episodes"), "ci_method": ci.get("method"),
                    "ci_reason": ci.get("reason")})
        row = {k: (np.nan if v is None else v) for k, v in row.items()}
        row["verdict"] = _verdict(row)
        rows.append(row)
    return rows


def _run_rep(args) -> list:
    """Worker: label + score one replication in its own temporary folder."""
    d, templates, lags, thresholds = args
    with tempfile.TemporaryDirectory(prefix="tsl_sim_") as tmp:
        label_rep(d, Path(tmp), lags, thresholds)
        rows = score_rep(d, Path(tmp), templates)
    rep = int(d["rep"].iloc[0])
    return [{"rep": rep, **r} for r in rows]


# ── one file ──────────────────────────────────────────────────────────────────

KEEP = ["rep", "template", "accepted", "reason", "const_w", "const_k",
        "train_confidence", "test_confidence", "test_baseline", "test_lift",
        "test_condition_days", "test_eligible", "ci_lo", "ci_hi", "ci_spans_zero",
        "ci_degenerate", "ci_n_episodes", "ci_method", "verdict"]
NOT_TESTABLE = ("not fitted", "ineligible", "not scored")
# DECIDED = the method gave an answer. "not fitted" (no effect on train, or the
# conclusion never occurs) is an answer: no. Only too few condition days on test
# ("ineligible") is an abstention, like the econometric arm's "too few trigger days".
UNDECIDED = ("ineligible", "not scored")


def load_reps(folder=None) -> pd.DataFrame:
    """Every reps_<file>.csv in the output folder, for rebuilding summary.csv."""
    files = sorted(Path(folder or out_dir()).glob("reps_*.csv"))
    if not files:
        raise FileNotFoundError(f"no reps_*.csv in {folder or out_dir()}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def run_file(path, templates="matching", reps=None, jobs: int = 1,
             thresholds: str = "train", verbose: bool = True) -> pd.DataFrame:
    """Every replication of one dataset file. Returns one row per (rep, template)."""
    meta = parse_name(path)
    names = templates_for(meta["sim_template"], templates)
    from tslearning.rules.templates import TEMPLATES
    needs = {p for t in names for p in TEMPLATES[t]["needs"]}
    lags = ((CFG.get("real_pairs") or {}).get("lags") or CFG["predicates"]["lags"]
            if needs & LEAD_PREDS else None)

    df = pd.read_csv(path)
    rep_ids = sorted(df["rep"].unique())
    if reps is not None:
        rep_ids = rep_ids[:int(reps)]
    tasks = [(df[df["rep"] == r].reset_index(drop=True), names, lags, thresholds)
             for r in rep_ids]

    rows = []
    if jobs and jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for i, out in enumerate(ex.map(_run_rep, tasks), start=1):
                rows += out
                if verbose and (i % 10 == 0 or i == len(tasks)):
                    print(f"    {Path(path).name}: {i}/{len(tasks)} reps", flush=True)
    else:
        for i, t in enumerate(tasks, start=1):
            rows += _run_rep(t)
            if verbose and (i % 10 == 0 or i == len(tasks)):
                print(f"    {Path(path).name}: {i}/{len(tasks)} reps", flush=True)

    out = pd.DataFrame(rows)
    out = out[[c for c in KEEP if c in out.columns]]
    for k, v in meta.items():
        out.insert(0, k, v)
    out.insert(0, "file", Path(path).name)
    out["thresholds"] = thresholds
    out["detected"] = out["verdict"].eq("significant")
    return out


def summarise(reps: pd.DataFrame) -> pd.DataFrame:
    """One row per (file, template): shares over the reps, next to the econometric
    test's rate on the same configuration (results_econ.csv), when there is one."""
    keys = ["file", "sim_template", "theta", "rho0", "n", "template", "thresholds"]
    v = reps["verdict"].astype(str)
    decided = ~v.str.startswith(UNDECIDED)
    reps = reps.assign(_fitted=reps["accepted"].astype(bool),
                       _testable=~v.str.startswith(NOT_TESTABLE),
                       _decided=decided,
                       _det_decided=reps["detected"].astype(float).where(decided),
                       detected=reps["detected"].astype(float))
    g = reps.groupby(keys, sort=False)
    s = g.agg(reps=("rep", "nunique"), fitted_share=("_fitted", "mean"),
              testable_share=("_testable", "mean"), decided_share=("_decided", "mean"),
              detection_rate=("detected", "mean"),
              detection_rate_decided=("_det_decided", "mean"),
              mean_test_lift=("test_lift", "mean")).reset_index()

    econ_f = SIM_DIR / "results_econ.csv"
    if econ_f.exists():
        e = pd.read_csv(econ_f).rename(columns={
            "template": "sim_template", "detection_rate": "econ_detection_rate",
            "testable_share": "econ_testable_share", "reps": "econ_reps"})
        e = e[["sim_template", "theta", "rho0", "n", "econ_reps",
               "econ_testable_share", "econ_detection_rate"]]
        s = s.merge(e, on=["sim_template", "theta", "rho0", "n"], how="left")
    return s


def out_dir() -> Path:
    return PATHS["results"] / "simulation"


def save(reps: pd.DataFrame, verbose: bool = True) -> Path:
    """reps_<file stem>.csv per file, and summary.csv (rows for these files replaced,
    the others kept)."""
    od = out_dir()
    od.mkdir(parents=True, exist_ok=True)
    for f, part in reps.groupby("file", sort=False):
        stem = f.removesuffix(".gz").removesuffix(".csv")
        part.to_csv(od / f"reps_{stem}.csv", index=False)
    sf = write_summary(reps)
    (od / "settings.json").write_text(json.dumps({
        "run": CFG["run"]["name"],
        "window": CFG["windows"]["financial"]["length"],
        "labels": CFG.get("labels"), "rules": CFG.get("rules"),
        "vol_percentile": CFG["predicates"]["vol_percentile"],
        "crash_percentile": CFG["predicates"]["crash_percentile"],
    }, indent=2, default=str))
    return sf


def write_summary(reps: pd.DataFrame) -> Path:
    """summary.csv: rows for the files in `reps` replaced, the others kept."""
    od = out_dir()
    od.mkdir(parents=True, exist_ok=True)
    new = summarise(reps)
    sf = od / "summary.csv"
    if sf.exists():
        old = pd.read_csv(sf)
        old = old[~old["file"].isin(new["file"])]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(["sim_template", "rho0", "n", "theta", "template"], kind="stable")
    new.to_csv(sf, index=False)
    return sf
