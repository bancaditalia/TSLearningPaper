#!/usr/bin/env python3
"""
Rebuild the PUBLIC repository from scratch: copy the selected files of this repo into a
separate folder, make ONE commit there, and (with --push) replace what is online.

    python scripts/make_public.py                   # build ../TSLearning-public, no push
    python scripts/make_public.py --push            # also force-push to the public remote
    python scripts/make_public.py --list            # only print what would be copied

The public folder is deleted and rebuilt every time, so the public history is always a
single commit. Never work or commit in it by hand: change this repo, then rerun.

What is published is decided by INCLUDE / EXCLUDE below (paths relative to the repo root,
glob patterns; a directory pattern takes everything under it).
"""
import argparse
import fnmatch
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INCLUDE = [
    # documentation and environment
    "README.md", "SCRIPTS.md", "requirements.txt", "requirements-frozen.txt",
    "environment.txt",
    # code and configuration
    "configs/", "scripts/", "tslearning/",
    # the paper's tables
    "paper_tables/",
    # input data: only the Fed GSW snapshot (public). The market and FRED series are NOT
    # redistributed (provider terms); scripts/data_prep.py downloads them.
    "data/raw/feds200628.csv.gz", "data/raw_checksums.json",
    "docs/gold_old_vs_new.png",
    # intermediates of the paper's run (main) and of the run whose classifier it uses
    # (main_minw0): ONLY the synthetic and trained artefacts. Everything holding real
    # returns (windows, batches, real-pair labels, .pkl) is recomputed, see
    # REPRODUCIBILITY.md.
    "data/processed/runs/main*/run.json", "data/processed/runs/main*/run_config.yaml",
    "data/processed/runs/main*/thresholds/",
    "data/processed/runs/main*/ddpm_unet1d_*.pt", "data/processed/runs/main*/synthetic_ddpm_*.npy",
    "data/processed/runs/main*/transfer_*/", "data/processed/runs/main*/classifier_*",
    "REPRODUCIBILITY.md", "results-index.md",
    # the reasoner's labels of the real test windows (input of fidelity and label
    # agreement): ONLY the label files those steps read: window names and day intervals,
    # plus the share of big days for highVol/lowVol. No sums of returns (sumFlow, trend,
    # deadZone), no batch inputs, no cached engine responses.
    *[f"data/processed/runs/main/{d}/{p}_*.csv"
      for d in ("real_test_sp500_*", "real_test_results_dominance")
      for p in ("isPositiveCorrelation", "isNegativeCorrelation", "isNoCorrelation",
                "highVol", "lowVol", "tailDependent")],
    # bitcoin: the two concordance shares the dominance relabelling reads (aggregates)
    *[f"data/processed/runs/main/real_test_results/{p}_*.csv"
      for p in ("magnitudeSameSignConsistencyPercentage", "magnitudeDiffSignConsistencyPercentage")],
    # the reasoner timing (seconds per batch), input of the speed table
    "data/processed/runs/main/_timing/timing.json", "data/processed/runs/main/_warmup/timing.json",
    # results: every paper run (main, sensitivity, robustness) and the comparisons
    "results/README.md", "results/compare/", "results/runs/",
    # econometric tests and simulation
    "data_simulation_econ tests/",
]

EXCLUDE = [
    "**/__pycache__/**", "**/.ipynb_checkpoints/**", "**/.DS_Store",
    "**/*.vadanb",                       # editor notebooks: saved outputs with local paths
    "data/raw/EU_*",                     # Eurostat files: large, used by no result
    "data/raw/feds200628.csv",           # the plain CSV; the .gz snapshot is published
    "data_simulation_econ tests/market_data/raw/*",   # provider downloads, not redistributed
    "data_simulation_econ tests/market_data/market_daily_levels.csv",       # derived from them
    "data_simulation_econ tests/market_data/market_daily_logreturns.csv",
    # the LLM prompts contain the training-period returns given to the model; they are
    # rebuilt by `llm_rules.py --dry-run` (the prompt hash in each run's JSON identifies them)
    "results/runs/*/llm_prompt_*",
    "data/raw/_backup/*", "data/raw/_extend/*",   # download backups, not inputs
    # simulated datasets: regenerated deterministically by simulation/export_datasets.py
    "data_simulation_econ tests/simulation/datasets/**",
]

# git refuses files above this; GitHub warns above 50 MB
MAX_MB = 95


def _match(rel: str, patterns) -> bool:
    """`dir/` = everything under dir; `**` crosses folders; `*` stays within one folder."""
    for p in patterns:
        if p.endswith("/"):
            if rel.startswith(p) or ("*" in p and fnmatch.fnmatch(
                    "/".join(rel.split("/")[:p.count("/")]) + "/", p)):
                return True
        elif "**" in p:
            if fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(rel, p.replace("**/", "")):
                return True
        elif rel.count("/") == p.count("/") and fnmatch.fnmatch(rel, p):
            return True
    return False


def engine_artifacts(run: str = "main") -> list:
    """Folder patterns of the reasoner's verification artefacts that the run's results
    actually use: for each verified rule, the DatalogMTL program the engine ran, the facts
    it was given (predicate intervals, no prices) and the engine's output. Read from the
    `artifacts` column of results/runs/<run>/verification*.csv, so only the current
    verification is published, not older attempts."""
    import csv
    pats = set()
    for f in (ROOT / "results" / "runs" / run).glob("verification*.csv"):
        if f.stem.endswith("_filtered"):
            continue
        with open(f) as fh:
            for row in csv.DictReader(fh):
                a = (row.get("artifacts") or "").strip()
                if a:
                    pats.add(f"data/processed/runs/{run}/{a.rstrip('/')}/")
    return sorted(pats)


def selected() -> list:
    out = []
    include = INCLUDE + engine_artifacts("main")
    for f in sorted(ROOT.rglob("*")):
        if not f.is_file() or f.is_symlink():
            continue
        rel = f.relative_to(ROOT).as_posix()
        if rel.startswith((".git/", "_archive/", "my-docs/", "ts311/", ".venv")):
            continue
        if _match(rel, include) and not _match(rel, EXCLUDE):
            out.append(rel)
    return out


def _summary(files, total) -> None:
    """One line per top-level folder, and one per run: files and MB."""
    from collections import defaultdict
    groups = defaultdict(lambda: [0, 0])
    for r in files:
        parts = r.split("/")
        # top-level folders as one line; the runs one line each
        depth = 4 if parts[:3] == ["data", "processed", "runs"] and len(parts) > 4 else \
                3 if parts[:2] == ["results", "runs"] and len(parts) > 3 else \
                2 if parts[0] == "data" and len(parts) > 2 else 1
        key = "/".join(parts[:depth]) + ("/" if depth < len(parts) else "")
        groups[key][0] += 1
        groups[key][1] += (ROOT / r).stat().st_size
    w = max(len(k) for k in groups)
    for k in sorted(groups):
        n, b = groups[k]
        print(f"  {k:<{w}}  {n:>5} file{'s' if n != 1 else ' '}  {b / 1e6:7.1f} MB")
    print(f"\n  total: {len(files)} files, {total:.0f} MB")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=str(ROOT.parent / "TSLearning-public"),
                    help="the public folder (deleted and rebuilt; default ../TSLearning-public)")
    ap.add_argument("--remote", default=None,
                    help="URL of the public repository (needed with --push)")
    ap.add_argument("--push", action="store_true", help="force-push the single commit")
    ap.add_argument("--list", action="store_true", help="print every file, copy nothing")
    ap.add_argument("--summary", action="store_true",
                    help="print a short overview (folders, files, size), copy nothing")
    a = ap.parse_args(argv)

    files = selected()
    big = [(r, (ROOT / r).stat().st_size / 1e6) for r in files
           if (ROOT / r).stat().st_size / 1e6 > MAX_MB]
    total = sum((ROOT / r).stat().st_size for r in files) / 1e6
    if a.summary:
        _summary(files, total)
        return
    if a.list:
        print("\n".join(files))
        print(f"\n{len(files)} files, {total:.0f} MB")
        return
    if big:
        sys.exit("files too large for git: " + ", ".join(f"{r} ({mb:.0f} MB)" for r, mb in big))
    # no personal folders in what is published (e.g. /Users/<name>/... in a log)
    import re
    home = re.compile(r"/Users/[^/\s]+/|/home/[^/\s]+/|" + re.escape(str(Path.home())))
    leaks = [r for r in files if r.endswith((".txt", ".md", ".csv", ".json", ".yaml", ".py",
                                             ".tex", ".vada"))
             and r != "scripts/make_public.py"          # this check's own pattern
             and home.search((ROOT / r).read_text(errors="ignore"))]
    if leaks:
        sys.exit("these files contain a personal folder path; fix them first:\n  "
                 + "\n  ".join(leaks))

    out = Path(a.out).resolve()
    if out == ROOT or ROOT in out.parents:
        sys.exit(f"refusing: {out} is inside this repository")
    if out.exists():
        shutil.rmtree(out)
    for r in files:
        dst = out / r
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / r, dst)
    (out / ".gitignore").write_text("__pycache__/\n.DS_Store\n")

    def git(*args):
        subprocess.run(["git", *args], cwd=out, check=True)
    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("commit", "-q", "-m", "Code, data and results for the paper")
    print(f"public repo rebuilt: {out} ({len(files)} files, {total:.0f} MB, one commit)")
    if a.push:
        if not a.remote:
            sys.exit("--push needs --remote <URL of the public repository>")
        git("remote", "add", "origin", a.remote)
        git("push", "-f", "origin", "main")
        print(f"pushed (replacing the previous version) to {a.remote}")


if __name__ == "__main__":
    main()
