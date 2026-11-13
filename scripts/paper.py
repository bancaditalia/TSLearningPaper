"""
Every paper table in one command — formatted from disk, or recomputed first.

    python scripts/paper.py tables               # format only, from the files on disk (seconds)
    python scripts/paper.py results              # recompute, incl. engine verification, then tables
    python scripts/paper.py results --no-engine  # recompute the non-engine parts only, then tables

`tables`   runs the consistency checks, then writes the rule tables (exactly what
           rule_table.py writes, for the py and the classifier source) and the
           distillation + speedup tables. Recomputes nothing.
`results`  runs the pipeline steps that produce the paper's numbers, in order, by calling
           the same scripts with the README's arguments, then `tables`:
             8   generate_rules.py, py + classifier source; multiple_testing.py
             9   verify_rules.py --engine both, py + classifier   [skipped with --no-engine]
             13  relabel.py (bitcoin targets) + eval.py fidelity per pair in paper.fidelity
             14  classifier_timing.py + timing_table.py (the reasoner timing on disk is kept)
           With --no-engine the verification files from the last engine run are left
           untouched; the checks say so if they hold no engine numbers.

NOT run here, on purpose (separate experiments, start them yourself first):
  * LLM runs (llm_rules.py)              * a new reasoner timing measurement (REPRODUCIBILITY.md §2)

Pairs: `paper.pairs`; fidelity sets: `paper.fidelity`; classifier: `classifier.weights` —
all in configs/config.yaml. Stops at the first step that fails.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _cli import bootstrap  # noqa: E402

bootstrap()


def _step(title: str, fn, *a, **kw):
    print(f"\n{'═' * 78}\n▶ {title}\n{'═' * 78}")
    try:
        return fn(*a, **kw)
    except SystemExit as e:          # argparse errors / p.error in a called script
        if e.code not in (0, None):
            sys.exit(f"\nstopped: '{title}' failed (exit {e.code}). Fix it and rerun.")
    except Exception as e:           # anything else: stop here, don't feed the next step
        sys.exit(f"\nstopped: '{title}' failed — {type(e).__name__}: {e}")


def results(no_engine: bool) -> None:
    import generate_rules, verify_rules, eval as eval_script, relabel
    import classifier_timing, timing_table, multiple_testing
    from tslearning.config import CFG, REPO_ROOT, resolve_path

    P = ",".join(CFG["paper"]["pairs"])
    _step("step 8 · rules · py", generate_rules.main,
          ["--source", "py", "-p", P, "--score", "--ci", "-q"])
    _step("step 8 · rules · classifier", generate_rules.main,
          ["--source", "classifier", "-p", P, "--score", "--ci", "-q"])
    _step("step 8 · multiple testing · py", multiple_testing.main, ["-q"])
    if no_engine:
        print("\n(step 9 skipped: --no-engine — verification files left as they are)")
    else:
        _step("step 9 · verification · py [ENGINE]", verify_rules.main,
              ["--source", "py", "-p", P, "--engine", "both", "-q"])
        _step("step 9 · verification · classifier [ENGINE]", verify_rules.main,
              ["--source", "classifier", "-p", P, "--engine", "both", "-q"])

    from tslearning.config import PATHS
    from tslearning.datasets.pairs import describe

    def _labelled(folder: Path) -> bool:
        return folder.is_dir() and any(folder.glob("isPositiveCorrelation_*.csv"))

    for pair, spec in (CFG["paper"].get("fidelity") or {}).items():
        args = ["--metric", "fidelity", "--dataset", "real", "--step", str(spec["step"]), "-q"]
        # the fidelity TARGETS are reasoner labels of the real test windows (README
        # step 12, [ENGINE]); without them there is nothing to score against
        if spec.get("results"):
            raw = resolve_path(Path(spec["results"]))
            raw = raw.with_name(raw.name[:-len("_dominance")]) \
                if raw.name.endswith("_dominance") else raw
        else:
            raw = PATHS["data_processed"] / f"real_test_{describe(pair)['slug']}"
        if not _labelled(raw):
            print(f"\n(step 13 skipped for {pair}: no reasoner labels in "
                  f"{raw.relative_to(REPO_ROOT)} — run the reasoner on the test windows first: REPRODUCIBILITY.md §2 [ENGINE])")
            continue
        if spec.get("results"):
            res = Path(spec["results"])
            res = resolve_path(res)      # data/processed/... -> this run's folder
            if res.name.endswith("_dominance"):
                _step(f"step 13 · relabel targets · {pair}", relabel.main,
                      ["--results", str(res.with_name(res.name[:-len("_dominance")])), "-q"])
            args += ["--results", str(res)]
        if pair != "sp500_bitcoin":          # bitcoin's test set is the aligned-pickle route
            args += ["--pair", pair]
        _step(f"step 13 · fidelity · {pair}", eval_script.main, args)

    # the classifier is timed on the window-pairs of the reasoner timing runs
    if any(PATHS["data_processed"].glob("*/timing.json")):
        _step("step 14 · classifier timing", classifier_timing.main, [])
        _step("step 14 · timing table", timing_table.main, [])
    else:
        print("\n(step 14 skipped: no reasoner timing.json in this run's folders — "
              "REPRODUCIBILITY.md §2 [ENGINE])")


def tables() -> int:
    from tslearning.config import CFG
    from tslearning.eval import report
    from tslearning.rules import tables as T

    print(f"\n{'═' * 78}\n▶ checks\n{'═' * 78}")
    problems = report.checks()
    for level, msg in problems:
        print(f"  {level}: {msg}")
    if not problems:
        print("  all inputs consistent")

    pairs = CFG["paper"]["pairs"]
    _step("tables · rules · py", T.run, pairs=pairs, verbose=False)
    _step("tables · rules · classifier", T.run, pairs=pairs, tag="classifier", verbose=False)
    _step("tables · distillation + speedup", report.write_tables, verbose=False)

    errors = [m for lv, m in problems if lv == "ERROR"]
    print(f"\ntables written to results/ — {len(errors)} error(s), "
          f"{len(problems) - len(errors)} warning(s) above")
    return 1 if errors else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["tables", "results"])
    ap.add_argument("--no-engine", action="store_true",
                    help="results mode: skip engine verification (step 9) and keep the "
                         "verification files from the last engine run")
    args = ap.parse_args(argv)
    if args.mode == "results":
        results(args.no_engine)
    elif args.no_engine:
        print("note: --no-engine only matters with `results`")
    sys.exit(tables())


if __name__ == "__main__":
    main()
