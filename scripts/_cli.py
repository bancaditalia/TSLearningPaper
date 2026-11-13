"""
Shared plumbing for the scripts/ entry points.

Every script here is a thin argparse wrapper: it parses flags, calls one `run()` in
the tslearning package, and prints a one-line result. No pipeline logic lives in
scripts/ — that way the same code path serves the CLI, the notebooks (which can
import the library directly) and the tests.

Two conventions:
  * `bootstrap()` puts the repo root on sys.path, so `python scripts/foo.py` works
    from any working directory without installing the package.
  * heavy imports (torch, sklearn, yfinance) happen INSIDE run(), never at module
    import, so `--help` works everywhere and a missing optional dependency only
    breaks the stage that needs it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _take_run_flag(argv: list) -> "str | None":
    """Remove `--run NAME` / `--run=NAME` from argv and return NAME (or None)."""
    for i, a in enumerate(list(argv)):
        if a == "--run" and (i + 1 >= len(argv) or argv[i + 1].startswith("-")
                             or not argv[i + 1].strip()):
            sys.exit("--run needs a run name, and none was given. If the command uses "
                     "$R, R is not set in this terminal: set it first, e.g. "
                     "R=<run name>  (python scripts/runs.py lists them)")
        if a == "--run" and i + 1 < len(argv):
            name = argv[i + 1]
            del argv[i:i + 2]
            return name
        if a.startswith("--run="):
            del argv[i]
            return a.split("=", 1)[1]
    return None


def _rebase_paths(argv: list) -> None:
    """Under a named run, point relative `data/processed/...` and `results/...` arguments
    at the RUN's folders, so every README command works unchanged with `--run NAME`
    (e.g. `--save results/fidelity_real_gold.csv` -> results/runs/NAME/...). A path that
    should stay in the base run is written `base:data/processed/...`."""
    from tslearning.config import CFG, PATHS, REPO_ROOT
    import yaml

    base = yaml.safe_load((REPO_ROOT / "configs" / "config.yaml").read_text())["paths"]
    roots = {base["data_processed"].rstrip("/") + "/": PATHS["data_processed"],
             base["results"].rstrip("/") + "/": PATHS["results"]}
    changed = []
    for i, a in enumerate(argv[1:], start=1):
        flag, eq, val = a.partition("=") if a.startswith("--") and "=" in a else ("", "", a)
        if val.startswith("base:"):
            argv[i] = f"{flag}{eq}{REPO_ROOT / val[len('base:'):]}"
            continue
        for rel, new_root in roots.items():
            if val.startswith(rel) and "/runs/" not in val:
                argv[i] = f"{flag}{eq}{new_root / val[len(rel):]}"
                changed.append(f"{val} -> {Path(argv[i].partition('=')[2] or argv[i]).relative_to(REPO_ROOT)}")
                break
    if changed:
        print(f"[run '{CFG['run']['name']}': paths moved to the run's folders: "
              + "; ".join(changed) + "]", file=sys.stderr)


def bootstrap() -> Path:
    """Make `import tslearning` work when running this file directly, and select the run.

    `--run NAME` (any script) selects configs/runs/NAME.yaml: it is taken out of sys.argv
    here and turned into TSL_RUN, BEFORE tslearning is imported, so every module sees
    the run's config and folders. Without it, TSL_RUN from the environment applies, and
    without that the base run (configs/config.yaml). A named run is announced on stderr
    and its folders get a run.json recording the parameters (tslearning/runs.py).
    """
    import os

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    name = _take_run_flag(sys.argv)
    if name is not None:
        os.environ["TSL_RUN"] = name
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        return REPO_ROOT                 # --help never creates folders
    from tslearning.config import BASE_RUN, CFG, run_summary
    if CFG["run"]["name"] != BASE_RUN and not os.environ.get("_TSL_RUN_ANNOUNCED"):
        print(f"[{run_summary()}]", file=sys.stderr)
        os.environ["_TSL_RUN_ANNOUNCED"] = "1"   # once, even when scripts call scripts
    if CFG["run"]["name"] != BASE_RUN:
        _rebase_paths(sys.argv)
    from tslearning.runs import ensure_manifest
    try:
        ensure_manifest()
    except RuntimeError as e:            # parameters changed under a named run: stop cleanly
        sys.exit(f"stopped: {e}")
    return REPO_ROOT


def parser(description: str, epilog: "str | None" = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=description, epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-q", "--quiet", action="store_true",
                   help="only print the final result line")
    return p


def csv_list(value: str) -> "list[str]":
    """--series SP500,Bitcoin -> ['SP500', 'Bitcoin']"""
    return [v.strip() for v in str(value).split(",") if v.strip()]


def int_list(value: str) -> "list[int]":
    """--lags 0,1,2 -> [0, 1, 2]"""
    return [int(v) for v in csv_list(value)]


class stage:
    """Context manager that frames a stage's output and times it.

        with stage("data prep", quiet=args.quiet):
            result = ingest_financial.run(...)
    """

    def __init__(self, name: str, quiet: bool = False):
        self.name = name
        self.quiet = quiet

    def __enter__(self):
        self.t0 = time.perf_counter()
        if not self.quiet:
            print(f"\n=== {self.name} " + "=" * max(0, 60 - len(self.name)))
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        if exc_type is None:
            print(f"[ok]   {self.name} ({dt:.1f}s)")
        else:
            print(f"[FAIL] {self.name} ({dt:.1f}s): {exc_type.__name__}: {exc}")
        return False
