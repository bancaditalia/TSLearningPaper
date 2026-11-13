"""
Vadalog reasoner client — batch runner for the symbolic pre-labeling stage.

Ported from the identical, thrice-duplicated cells in nb_10_classifiers-clean
(cell 36), nb_10_part_2 (cell A.2b) and nb_prep_real_pairs (cell B2). The
notebooks are NOT modified.

The engine is reached over HTTP (`reasoner_endpoint` in configs/config.yaml) and
takes a comma-separated params string:

    X = batch CSV (flow facts, disjoint-block layout)
    T = tau.csv        (90th-percentile |return|, one row PER WINDOW of the batch)
    Q = qlow.csv       (5th-percentile return, one row PER WINDOW of the batch)
                       Both written next to the batch by thresholds.write_for_batch; see
                       tslearning/labeling/thresholds.py for why per window.
    I = run tag        -> the .vada names its outputs {predicate}_{I}.csv
    R = result folder  -> where those CSVs are written

Runs are CACHED and RESUMABLE: the JSON response for tag I is stored as
{R}/{I}.json, and a batch whose JSON already contains "resultSet" is skipped. A
FAILED call is deliberately never cached — otherwise a transient error would
poison the cache and the batch would be silently skipped forever.

This matters because a full labeling run is measured in days, so being able to stop and resume is not a nicety.
"""
from __future__ import annotations

import datetime
import glob
import json
import os
import time
from pathlib import Path

from tslearning.config import CFG, PATHS
from tslearning.config import REPO_ROOT


def repository_dir() -> Path:
    """The engine's program repository, as an absolute path.

    `programName` is resolved by the engine relative to THIS folder, so every ruleset
    reference in the codebase is derived from it (see `reasoner_repository` in
    configs/config.yaml). Change the engine's repository setting and the config together.
    """
    return (REPO_ROOT / CFG.get("reasoner_repository", ".")).resolve()


def program_name(path) -> str:
    """Convert a .vada path on disk into the `programName` the engine expects.

    Raises if the program lies outside the repository — that means the engine cannot see
    it, and the fix is a config change, not a silent fallback.
    """
    p = Path(path).resolve()
    repo = repository_dir()
    try:
        return str(p.relative_to(repo))
    except ValueError:
        raise ValueError(
            f"{p} is outside the reasoner repository ({repo}), so the engine cannot load "
            f"it. Either set reasoner_repository in configs/config.yaml to a folder that "
            f"contains it (\".\" = the repo root covers both tslearning/symbolic and "
            f"data/), and point the engine at the same folder — or write the program "
            f"inside the current repository.")


# The values written as literals in tslearning/symbolic/preLabeling.vada (the paper's
# settings). A run whose config differs gets a generated copy of the program.
_VADA_PAPER = {"window": 50, "trend": 0.02, "high_vol_pct": 30.0, "low_vol_pct": 5.0,
               "tail_min": 2}


def _fmt(x: float) -> str:
    """A float literal the way the .vada writes them (30.0, 0.02, -0.02)."""
    return repr(float(x))


def ruleset_path() -> Path:
    """The pre-labeling program FILE for the current run.

    The base program, tslearning/symbolic/preLabeling.vada, hard-codes the window
    (`/\\50`) and the cut-offs (trend 0.02, highVol 30, lowVol 5, tailDependent 2).
    When the run's config (windows.financial.length, labels.*) differs, a copy with
    those values substituted is written to <data_processed>/vada/preLabeling.vada and
    returned — so the reasoner always labels with the same constants as the Python
    labeller. Every substitution is checked to hit exactly the expected line, so an
    edit to the base program fails loudly instead of being half-applied.
    """
    import re
    from tslearning.labeling.exact_flags import label_constants

    base = PATHS["symbolic"] / "preLabeling.vada"
    want = {"window": int(CFG["windows"]["financial"]["length"]), **label_constants()}
    if all(float(want[k]) == float(v) for k, v in _VADA_PAPER.items()):
        return base

    text = base.read_text()
    n_box = len(re.findall(r"/\\50(?=\s)", text))
    text = re.sub(r"/\\50(?=\s)", lambda m: "/\\" + str(want["window"]), text)
    subs = [  # (old, new) — each must occur exactly once
        ("bigMovePct(X,Pct), Pct >= 30.0.", f"bigMovePct(X,Pct), Pct >= {_fmt(want['high_vol_pct'])}."),
        ("bigMovePct(X,Pct), Pct <= 5.0.", f"bigMovePct(X,Pct), Pct <= {_fmt(want['low_vol_pct'])}."),
        ("sumFlow(X,Sum), Sum >= 0.02.", f"sumFlow(X,Sum), Sum >= {_fmt(want['trend'])}."),
        ("sumFlow(X,Sum), Sum <= -0.02.", f"sumFlow(X,Sum), Sum <= {_fmt(-want['trend'])}."),
        ("sumFlow(X,Sum), Sum < 0.02, Sum > -0.02.",
         f"sumFlow(X,Sum), Sum < {_fmt(want['trend'])}, Sum > {_fmt(-want['trend'])}."),
        ("coCrashMax(X,Y,Max), Max >= 2.", f"coCrashMax(X,Y,Max), Max >= {int(want['tail_min'])}."),
    ]
    for old, new in subs:
        if text.count(old) != 1:
            raise RuntimeError(f"preLabeling.vada changed: expected exactly one "
                               f"'{old}', found {text.count(old)} — update ruleset_path()")
        text = text.replace(old, new)
    header = (f"% GENERATED from tslearning/symbolic/preLabeling.vada for run "
              f"'{CFG['run']['name']}' — do not edit.\n% window {want['window']} "
              f"({n_box} aggregation boxes), trend {want['trend']}, highVol "
              f"{want['high_vol_pct']}, lowVol {want['low_vol_pct']}, tailDependent "
              f">= {want['tail_min']}\n")
    out = PATHS["data_processed"] / "vada" / "preLabeling.vada"
    # the engine reads `out`; the copy in the results folder (tracked by git) keeps the
    # exact program next to the numbers it produced
    for f in (out, PATHS["results"] / "preLabeling.vada"):
        f.parent.mkdir(parents=True, exist_ok=True)
        if not f.exists() or f.read_text() != header + text:
            f.write_text(header + text)
    return out


def default_ruleset() -> str:
    """The pre-labeling program's name for the current run, relative to the configured
    repository (see ruleset_path)."""
    return program_name(ruleset_path())


class ReasonerError(RuntimeError):
    """The engine answered, but with an error instead of a resultSet."""


def _abs(p) -> str:
    return os.path.abspath(str(p))


def result_is_valid(json_path) -> bool:
    """True if a cached response exists and holds a genuine resultSet."""
    p = Path(json_path)
    if not p.exists():
        return False
    try:
        with open(p) as f:
            return "resultSet" in json.load(f)
    except Exception:
        return False


def deadband() -> float:
    """The regime cut, from configs/config.yaml. One source of truth — see the note
    there; three independent defaults is what produced the label mismatch."""
    return float(CFG.get("labels", {}).get("deadband", 10.0))


def evaluate(batch_csv, tag: str, result_dir, thresholds_dir=None,
             ruleset: "str | None" = None, endpoint: "str | None" = None,
             timeout: "float | None" = None, band: "float | None" = None) -> dict:
    """POST one batch to the engine and return the parsed JSON response.

    Raises ReasonerError if the response has no resultSet, so the caller can decide
    whether to stop or continue. Nothing is cached here — see `run_batches`.

    `band` is passed to the program as ${Deadband}, alongside X/T/Q/I/R, so the
    reasoner decides the regime with the SAME constant the Python paths use.
    Defaults to configs/config.yaml `labels.deadband`.
    """
    import requests

    from tslearning.labeling import thresholds as thr

    endpoint = endpoint or CFG["reasoner_endpoint"]
    # T and Q: one row per WINDOW of this batch, keyed by the window name the batch uses.
    # (A per-series file never joined with the windows: the engine applied no threshold.)
    # Files a batch writer already wrote (e.g. rolling thresholds) are kept.
    # `thresholds_dir` is no longer used: the values come from the run's thresholds.
    # Rule verification ("facts" mode) passes a FOLDER of fact CSVs, not a batch: its
    # program reads no thresholds, so there is nothing to write (writing them crashed
    # every verification from 2026-10-21 to 2026-10-27 with "Is a directory").
    if Path(batch_csv).is_dir():
        tau_csv = qlow_csv = Path(batch_csv)
    else:
        tau_csv, qlow_csv = thr.write_for_batch(batch_csv)
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    band = deadband() if band is None else float(band)
    ruleset = ruleset or default_ruleset()

    params = (f"X={_abs(batch_csv)},"
              f"T={_abs(tau_csv)},"
              f"Q={_abs(qlow_csv)},"
              f"I={tag},"
              f"R={_abs(result_dir)},"
              f"Deadband={band:g}")

    resp = requests.post(endpoint, data={"programName": ruleset, "params": params},
                         timeout=timeout)
    try:
        payload = resp.json()
    except Exception:
        raise ReasonerError(f"non-JSON response (HTTP {resp.status_code}) for tag {tag}")
    if "resultSet" not in payload:
        raise ReasonerError(payload.get("errorMessage", f"no resultSet for tag {tag}"))
    return payload


# Predicates emitted for EVERY window pair, unconditionally. Completeness has to be
# judged on one of these: highVol or tailDependent legitimately having zero rows means
# "nothing was volatile", whereas percentagePositive having zero rows can only mean the
# batch did not run.
ANCHOR_PREDICATES = ("percentagePositive",
                     "magnitudeSameSignConsistencyPercentage",
                     "magnitudeDiffSignConsistencyPercentage")


def _has_rows(result_dir, tag) -> bool:
    """True if an ANCHOR predicate's {predicate}_{tag}.csv holds at least one data row.

    Presence of a file is not evidence the batch ran: the reasoner writes the CSVs with
    their headers up front, so a run that dies mid-way leaves a complete-looking set of
    empty files. Testing existence alone marked such a batch complete FOREVER — observed
    on the real test set, where batch 0 had 21 header-only CSVs, no .json, and was
    reported "cached, skip" on every later run while silently dropping 25 windows.

    Testing "any file has a row" is also too weak: that same failed batch had two stray
    rows across two conditional predicates. Hence the anchors.
    """
    for pred in ANCHOR_PREDICATES:
        p = Path(result_dir) / f"{pred}_{tag}.csv"
        try:
            with open(p) as f:
                next(f, None)              # header
                if next(f, None):          # at least one data row
                    return True
        except OSError:
            continue
    return False


def _save_timing(result_dir, batch_times, t_start, done, skipped, failed, band):
    """Write timing.json, merging with any existing entries from prior runs."""
    timing_path = Path(result_dir) / "timing.json"
    prev_batches = {}
    if timing_path.exists():
        try:
            with open(timing_path) as f:
                prev_batches = json.load(f).get("batches", {})
        except Exception:
            pass
    prev_batches.update(batch_times)
    timing = {
        "batches": prev_batches,
        "total_seconds": round(time.perf_counter() - t_start, 1),
        "deadband": band,
        "n_done": len(done),
        "n_skipped": len(skipped),
        "n_failed": len(failed),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)


def run_batches(batch_paths, result_dir, tags=None, thresholds_dir=None,
                ruleset: "str | None" = None, endpoint: "str | None" = None,
                stop_on_error: bool = True, verbose: bool = True,
                band: "float | None" = None) -> dict:
    """Evaluate a list of batch CSVs, caching and skipping completed ones.

    batch_paths : ordered paths to the batch CSVs.
    tags        : run tags, one per batch (defaults to the zero-padded index).
                  The tag ends up in every output filename, so it must be unique
                  per batch AND per pair — use "{pair}_{NNN}" for multi-pair runs.
    band        : the regime deadband passed to the program as ${Deadband}
                  (default: configs/config.yaml `labels.deadband`).

    Returns {"done": [...], "skipped": [...], "failed": [(tag, msg), ...],
             "seconds": float, "deadband": float}.
    """
    batch_paths = [Path(p) for p in batch_paths]
    tags = list(tags) if tags else [f"{i:03d}" for i in range(len(batch_paths))]
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    band = deadband() if band is None else float(band)

    done, skipped, failed = [], [], []
    batch_times = {}  # tag -> seconds
    t_start = time.perf_counter()
    if verbose:
        print(f"deadband {band:g} (config labels.deadband) -> ${{Deadband}}")

    for csv_path, tag in zip(batch_paths, tags):
        json_path = result_dir / f"{tag}.json"

        # Two ways a batch can already be complete: our cached JSON, or the .vada's
        # own {predicate}_{tag}.csv files from a previous run that predates caching.
        #
        # The CSV check requires at least one file with an actual ROW in it. A batch
        # that failed part-way leaves header-only CSVs behind, and testing mere
        # existence marked those as complete FOREVER — an interrupted run then looked
        # resumable while silently skipping 25 windows. Observed on the real test set:
        # batch 0 had 21 empty CSVs, no .json, and was reported "cached, skip" on
        # every subsequent run.
        #
        # A cached result also has to have been computed WITH this batch's per-window
        # threshold files (thresholds.py): results from before 2026-10-21 were computed
        # with no threshold at all (no highVol, no crash ever) and are redone.
        from tslearning.labeling import thresholds as thr
        tau_p, _ = thr._thr_paths(csv_path)
        # ... and with the CURRENT program: a result older than preLabeling.vada (e.g.
        # before the 2026-10-27 fix of the duplicated totalCount rule) is redone.
        cached = result_is_valid(json_path) or _has_rows(result_dir, tag)
        made = json_path if json_path.exists() else None
        prog = ruleset_path()
        fresh = (tau_p.exists() and made is not None
                 and made.stat().st_mtime >= tau_p.stat().st_mtime
                 and made.stat().st_mtime >= prog.stat().st_mtime)
        if cached and fresh:
            skipped.append(tag)
            if verbose:
                print(f"{tag} — cached, skip")
            continue
        if cached and verbose:
            print(f"{tag} — cached result predates its per-window thresholds "
                  f"(_thr/{tau_p.name}) or the current {prog.name}; recomputing")

        if not csv_path.exists():
            failed.append((tag, f"batch CSV missing: {csv_path}"))
            if verbose:
                print(f"{tag} — MISSING {csv_path.name}")
            if stop_on_error:
                break
            continue

        t0 = time.perf_counter()
        try:
            payload = evaluate(csv_path, tag, result_dir, thresholds_dir,
                               ruleset=ruleset, endpoint=endpoint, band=band)
        except Exception as e:
            failed.append((tag, str(e)))
            if verbose:
                print(f"{tag} — ERROR: {e} (not cached)")
            if stop_on_error:
                if verbose:
                    print("  stopping — fix the error, then re-run (done batches are skipped)")
                break
            continue

        with open(json_path, "w") as f:
            json.dump(payload, f)
        batch_sec = time.perf_counter() - t0
        batch_times[tag] = round(batch_sec, 1)
        done.append(tag)
        if verbose:
            print(f"{tag} — saved ({batch_sec:.1f}s)")

        # Write timing after every batch so nothing is lost on crash/sleep
        _save_timing(result_dir, batch_times, t_start, done, skipped, failed, band)

    elapsed = time.perf_counter() - t_start
    if verbose:
        print(f"reasoner: {len(done)} run, {len(skipped)} cached, {len(failed)} failed "
              f"in {elapsed:.1f}s -> {result_dir}")

    # Final write with accurate totals — only if this run timed something: a run where
    # every batch was cached would otherwise overwrite the earlier run's totals
    if done or not (result_dir / "timing.json").exists():
        _save_timing(result_dir, batch_times, t_start, done, skipped, failed, band)

    # Cached batches are skipped, so they are not timed. Say so, loudly: a "timing run"
    # into a folder that is already labelled measures nothing, and timing.json then
    # still holds the earlier times.
    if skipped and not done:
        print(f"note: all {len(skipped)} batch(es) were already labelled in {result_dir} "
              f"— nothing was run, and timing.json keeps the earlier times. If this was "
              f"meant as a TIMING run, it measured nothing: use an empty --results folder.")
    elif skipped:
        print(f"note: timing.json now mixes {len(done)} batch(es) timed in this run with "
              f"{len(skipped)} timed in an earlier one (cached here).")

    return {"done": done, "skipped": skipped, "failed": failed, "seconds": elapsed,
            "deadband": band}


def run_pair_batches(pair: str, batch_dir, result_dir, n_batches: "int | None" = None,
                     thresholds_dir=None, ruleset: "str | None" = None,
                     endpoint: "str | None" = None, stop_on_error: bool = True,
                     verbose: bool = True, band: "float | None" = None) -> dict:
    """Convenience wrapper for the real-pair layout: {pair}_batch_{NNN}.csv in
    `batch_dir`, tagged "{pair}_{NNN}" so several pairs can share one result folder.
    """
    batch_dir = Path(batch_dir)
    paths = sorted(batch_dir.glob(f"{pair}_batch_*.csv"))
    if n_batches is not None:
        paths = paths[:n_batches]
    tags = [f"{pair}_{p.stem.split('_')[-1]}" for p in paths]
    if verbose:
        print(f"{pair}: {len(paths)} batches -> {result_dir}")
    return run_batches(paths, result_dir, tags=tags, thresholds_dir=thresholds_dir,
                       ruleset=ruleset, endpoint=endpoint,
                       stop_on_error=stop_on_error, verbose=verbose, band=band)
