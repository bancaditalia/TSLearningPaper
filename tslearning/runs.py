"""
Runs: which periods / window / stride produced a folder of results.

Every run-scoped output folder (PATHS["data_processed"], PATHS["results"]) carries a
run.json with the run's name and the parameters that change its numbers. It is written
the first time a script runs under that run, and checked every time after:

  * named run (configs/runs/<name>.yaml): if the parameters on disk differ from the
    YAML now, the script STOPS. Results made with two different settings would otherwise
    sit in one folder with nothing to tell them apart. Make a new run instead (or delete
    the folder on purpose).
  * base run (configs/config.yaml alone): a change is reported and recorded in the
    file's history, but does not stop anything — the base config is edited in normal
    development, and the paper's folders predate this mechanism.

`python scripts/runs.py` lists every run with its parameters and folders.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import warnings
from pathlib import Path

from tslearning.config import BASE_RUN, CFG, PATHS, RUNS_DIR, load_config

MANIFEST = "run.json"
FULL_CONFIG = "run_config.yaml"   # every setting the run used (config.yaml + its overlay)


def write_full_config(d, cfg: "dict | None" = None) -> Path:
    """Write <folder>/run_config.yaml: the run's COMPLETE configuration, i.e.
    configs/config.yaml merged with configs/runs/<name>.yaml (paths left out: they are
    derived from the run name). Rewritten only when it changes."""
    import yaml
    c = cfg or CFG
    full = {k: v for k, v in c.items() if k not in ("paths", "run")}
    text = (f"# run '{c['run']['name']}': complete configuration "
            f"(configs/config.yaml + {c['run']['file']}).\n"
            f"# Written by tslearning/runs.py; do not edit — change the run's YAML instead.\n"
            + yaml.safe_dump(json.loads(json.dumps(full, default=str)), sort_keys=False,
                             allow_unicode=True))
    f = Path(d) / FULL_CONFIG
    if not f.exists() or f.read_text() != text:
        f.write_text(text)
    return f


def parameters(cfg: "dict | None" = None) -> dict:
    """The settings that change a run's numbers — and nothing else (paths, endpoints,
    plotting options are left out on purpose)."""
    c = cfg or CFG
    rp = c.get("real_pairs") or {}
    return {
        "split": c["split"],
        "window": c["windows"]["financial"],
        "real_pairs": {k: rp.get(k) for k in ("date_from", "date_to", "stride", "lags",
                                              "macro_states")},
        "labels": {k: v for k, v in (c.get("labels") or {}).items()
                   if k not in ("pos_threshold", "neg_threshold")},
        "rules": c.get("rules") or {},
        "synth": c.get("synth") or {},
        "data": c.get("data") or {},          # download period: feeds the DDPM training
        "seed": c.get("seed"),
    }


def info(cfg: "dict | None" = None) -> dict:
    """Recorded in run.json and kept up to date, but NOT part of the fingerprint: they
    are set during a run's life (the classifier is trained inside the run, then named in
    its YAML; pairs can be added) without changing what the other outputs mean."""
    c = cfg or CFG
    return {"classifier_weights": (c.get("classifier") or {}).get("weights"),
            "pairs": (c.get("paper") or {}).get("pairs")}


def fingerprint(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=str)
                          .encode()).hexdigest()[:12]


def _diff(old: dict, new: dict, prefix: str = "") -> list:
    out = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k), new.get(k)
        if isinstance(a, dict) and isinstance(b, dict):
            out += _diff(a, b, f"{prefix}{k}.")
        elif a != b:
            out.append(f"{prefix}{k}: {a!r} -> {b!r}")
    return out


def ensure_manifest(dirs=None, cfg: "dict | None" = None) -> None:
    """Write or check run.json in each run-scoped folder (see module docstring)."""
    c = cfg or CFG
    name = c["run"]["name"]
    params = parameters(c)
    fp = fingerprint(params)
    now = _dt.datetime.now().isoformat(timespec="seconds")
    for d in dirs or (PATHS["data_processed"], PATHS["results"]):
        d = Path(d)
        if name != BASE_RUN:
            d.mkdir(parents=True, exist_ok=True)
            write_full_config(d, c)
        f = d / MANIFEST
        if not f.exists():
            d.mkdir(parents=True, exist_ok=True)
            rec = {"run": name, "fingerprint": fp, "created": now,
                   "config": c["run"]["file"], "parameters": params, "info": info(c),
                   "history": []}
            if name == BASE_RUN and any(p.name != MANIFEST for p in d.iterdir()):
                rec["note"] = ("written into a folder that already held outputs: those "
                               "predate this record and are not covered by it")
            f.write_text(json.dumps(rec, indent=2, default=str))
            continue
        rec = json.loads(f.read_text())
        if rec.get("fingerprint") == fp:
            if rec.get("info") != info(c):          # e.g. the run's classifier was set
                rec["info"] = info(c)
                f.write_text(json.dumps(rec, indent=2, default=str))
            continue
        changes = _diff(rec.get("parameters", {}),
                        json.loads(json.dumps(params, default=str)))
        if name != BASE_RUN:
            raise RuntimeError(
                f"run '{name}': {f} was written with different parameters:\n  "
                + "\n  ".join(changes)
                + f"\nResults made with both settings would mix in one folder. Create a "
                  f"new run for the new settings (scripts/runs.py new ...), or delete "
                  f"{d} if the old results are no longer wanted.")
        warnings.warn(f"base run: parameters changed since {f} was written "
                      f"({'; '.join(changes)}). Outputs made before and after the change "
                      f"now share this folder — recorded in its history.")
        rec.setdefault("history", []).append({"until": now, "fingerprint":
                                              rec.get("fingerprint"),
                                              "parameters": rec.get("parameters")})
        rec.update(fingerprint=fp, parameters=params, updated=now)
        f.write_text(json.dumps(rec, indent=2, default=str))


# Parameters that act only at the RULE stage: a change to them leaves the run's labels,
# classifier and engine labels valid, and every rule output is rebuilt by rerunning the
# rule steps. Only these can be accepted into an existing run (accept_rules_change).
RULE_STAGE = ("rules.",)


def accept_rules_change(name: str) -> list:
    """Let an existing named run take a new RULE-STAGE setting (e.g. rules.min_w) without
    a new run. Refuses if anything else changed. The old parameters go to the history;
    the rule outputs in the folder are stale until the rule steps are rerun.
    Returns the list of changes accepted."""
    c = load_config(run=name)
    params = parameters(c)
    fp = fingerprint(params)
    now = _dt.datetime.now().isoformat(timespec="seconds")
    accepted = []
    for d in (c["paths"]["data_processed"], c["paths"]["results"]):
        f = Path(d) / MANIFEST
        if not f.exists():
            continue
        rec = json.loads(f.read_text())
        if rec.get("fingerprint") == fp:
            continue
        changes = _diff(rec.get("parameters", {}), json.loads(json.dumps(params, default=str)))
        other = [ch for ch in changes if not ch.startswith(RULE_STAGE)]
        if other:
            raise RuntimeError(f"run '{name}': not only rule-stage settings changed "
                               f"({'; '.join(other)}) — make a new run instead")
        rec.setdefault("history", []).append({"until": now, "fingerprint":
                                              rec.get("fingerprint"),
                                              "parameters": rec.get("parameters"),
                                              "accepted_change": changes})
        rec.update(fingerprint=fp, parameters=params, updated=now)
        f.write_text(json.dumps(rec, indent=2, default=str))
        accepted += [ch for ch in changes if ch not in accepted]
    return accepted


def list_runs() -> list:
    """[{name, file, parameters, data, results, has_outputs}] for the base run and every
    configs/runs/*.yaml."""
    names = [BASE_RUN] + sorted(p.stem for p in RUNS_DIR.glob("*.yaml")) \
        if RUNS_DIR.exists() else [BASE_RUN]
    out = []
    for n in names:
        try:
            c = load_config(run=n)
        except Exception as e:            # a broken overlay should not hide the others
            out.append({"name": n, "error": str(e)})
            continue
        dp, rs = c["paths"]["data_processed"], c["paths"]["results"]
        out.append({"name": n, "file": c["run"]["file"], "parameters": parameters(c),
                    "fingerprint": fingerprint(parameters(c)),
                    "data": dp, "results": rs,
                    "has_outputs": (dp / MANIFEST).exists() or (rs / MANIFEST).exists()})
    return out
