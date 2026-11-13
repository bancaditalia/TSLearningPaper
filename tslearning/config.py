"""
tslearning/config.py
====================
Loads configs/config.yaml and hands back the settings, with all paths
resolved to absolute Path objects (relative to the repo root, so it works
from a notebook in any folder).

Usage
-----
    from tslearning.config import CFG, PATHS, RUN

    win = CFG["windows"][CFG["series_freq"]["Bitcoin"]]   # {'length': 50, ...}
    df  = pd.read_csv(PATHS["data_raw"] / "SP500_raw.csv") # absolute path

Runs (robustness checks: other periods, window lengths, strides)
----------------------------------------------------------------
The paper's settings are the BASE run: configs/config.yaml alone, outputs in
data/processed/ and results/ exactly as before.

Any other run is a small YAML file in configs/runs/<name>.yaml that overrides only what
changes (split, real_pairs dates/stride, windows.financial, labels, rules, ...). It is
selected with the environment variable TSL_RUN=<name>, or `--run <name>` on any script
(scripts/_cli.py turns the flag into the variable before tslearning is imported).

A named run gets its own folders, so nothing of the base run is touched:
    data/processed/runs/<name>/     every intermediate (DDPM, transfer sets, labels, ...)
    results/runs/<name>/            every table and CSV
and both carry a run.json recording the parameters (tslearning/runs.py), so a folder
always says which periods / window / stride produced it.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

# Repo root = one level above this file's package folder.
REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_FILE = REPO_ROOT / "configs" / "config.yaml"
RUNS_DIR = REPO_ROOT / "configs" / "runs"
BASE_RUN = "base"

# Paths that hold a run's OUTPUTS: a named run gets its own subfolder under each.
_RUN_SCOPED_PATHS = ("data_processed", "results", "figures")


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge `over` onto a copy of `base` (dicts merge, everything else
    is replaced). Lists are replaced, not appended. A key set to null in `over` is
    REMOVED (e.g. a run without Bitcoin drops paper.fidelity.sp500_bitcoin)."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if v is None and isinstance(out.get(k), dict):
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def run_file(name: str) -> Path:
    return RUNS_DIR / f"{name}.yaml"


def load_config(path: "str | Path" = _CONFIG_FILE, run: "str | None" = None) -> dict:
    """Read the YAML config (+ the run overlay, if any) and turn every entry under
    `paths:` into an absolute Path anchored at the repo root.

    `run=None` reads TSL_RUN from the environment; "" or "base" means the base run.
    """
    cfg = yaml.safe_load(Path(path).read_text())
    run = os.environ.get("TSL_RUN", "") if run is None else run
    run = (run or "").strip() or BASE_RUN

    overlay = {}
    if run != BASE_RUN:
        f = run_file(run)
        if not f.exists():
            known = sorted(p.stem for p in RUNS_DIR.glob("*.yaml")) if RUNS_DIR.exists() else []
            raise FileNotFoundError(
                f"run '{run}' not found: expected {f}. Known runs: "
                f"{', '.join(known) or '(none)'}. Create one with "
                f"`python scripts/runs.py new <name> ...`.")
        overlay = yaml.safe_load(f.read_text()) or {}
        if "paths" in overlay:
            raise ValueError(f"{f}: a run may not override `paths:` — its folders are "
                             f"derived from the run name")
        cfg = _deep_merge(cfg, overlay)

    paths = {k: REPO_ROOT / v for k, v in cfg.get("paths", {}).items()}
    if run != BASE_RUN:
        for k in _RUN_SCOPED_PATHS:
            if k in paths:
                paths[k] = paths[k] / "runs" / run
    # the base run's processed folder, for inputs every run may share read-only
    paths["base_processed"] = REPO_ROOT / yaml.safe_load(
        Path(path).read_text())["paths"]["data_processed"]
    cfg["paths"] = paths
    f = run_file(run) if run != BASE_RUN else Path(path)
    cfg["run"] = {"name": run, "overlay": overlay,
                  "file": str(f.relative_to(REPO_ROOT)) if f.is_relative_to(REPO_ROOT)
                  else str(f)}
    return cfg


# Load once at import; reuse everywhere.
CFG = load_config()
PATHS = CFG["paths"]
RUN = CFG["run"]["name"]


def resolve_path(p: "str | Path") -> Path:
    """A path written in the config (classifier.weights, paper.fidelity.*.results, ensemble
    members): absolute stays as is; "data/processed/..." points into the CURRENT run's
    processed folder (so a named run never silently reads the base run's files);
    "base:data/processed/..." is the base run's file on purpose (e.g. a stride-only run
    reusing the paper's classifier); anything else is relative to the repo root."""
    if str(p).startswith("base:"):           # explicitly the BASE run's file
        return REPO_ROOT / str(p)[len("base:"):]
    p = Path(p)
    if p.is_absolute():
        return p
    base_rel = Path(yaml.safe_load(_CONFIG_FILE.read_text())["paths"]["data_processed"])
    try:
        return PATHS["data_processed"] / p.relative_to(base_rel)
    except ValueError:
        return REPO_ROOT / p


def run_summary(cfg: "dict | None" = None) -> str:
    """One line saying which run this is: name, periods, window, stride."""
    c = cfg or CFG
    w = c["windows"]["financial"]
    tr, te = c["split"]["train"], c["split"]["test"]
    rp = c.get("real_pairs") or {}
    return (f"run '{c['run']['name']}': train {tr[0]}..{tr[1]}, test {te[0]}..{te[1]}, "
            f"window {w['length']}, real-pair stride {rp.get('stride', 1)}")
