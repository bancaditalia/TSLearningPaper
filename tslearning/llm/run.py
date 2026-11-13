"""
Orchestration for the LLM arm: build the prompt, call a model, parse, score, save.

Both modes live here.

  propose   the model sees the TRAIN period and proposes rules; they are frozen and
            scored on test, exactly as the template fitter's are.
  explain   the model sees the FINAL scored rules and explains them: the LLM used only as
            an explainer at the end, never as a miner.
"""
from __future__ import annotations

from pathlib import Path

import hashlib
from datetime import datetime

from tslearning.config import CFG, PATHS


def _prompt_hash(text: str) -> str:
    """Short SHA-256 prefix of the prompt text. Two runs with the same hash used the
    same prompt; different hashes mean the prompt changed between runs."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _timestamp() -> str:
    """YYYYMMDD_HHMMSS for the current local time."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _next_run_path(out_dir: Path, tag: str) -> Path:
    """Return out_dir / {tag}_{YYYYMMDD_HHMMSS}.json."""
    return out_dir / f"{tag}_{_timestamp()}.json"


def _train_window(facts, train=None):
    """(day_lo, day_hi, lo_date, hi_date) for the training split."""
    from tslearning.rules import scoring
    lo, hi = train or CFG["split"]["train"]
    d0, d1 = scoring.day_range_for_dates(facts, lo, hi)
    return d0, d1, lo, hi


def series_block(pair: str, facts, d0: int, d1: int, max_days: int = 1800) -> str:
    """The RAW daily returns for the training days, on the pair's own day numbering.

    This is the primary input mode. The pipeline starts from raw series, so a comparison
    that hands the model pre-derived predicate facts would be giving it the output of the
    most expensive stage for free — it would compare only the last step. Here the model
    starts where the pipeline starts.

    Two things this must get right, or the experiment is void:

    1. **Day numbers must match the facts.** The rules the model proposes are scored
       against intervals indexed by the labelled pair's aligned calendar. Re-aligning the
       series independently could shift the index and silently mis-score everything, so
       the alignment is taken from `facts.dates` when it exists.
    2. **Training days only.** Sliced to [d0, d1) before serialising.
    """
    import numpy as np
    import pandas as pd

    from tslearning.datasets.pairs import describe, load_series

    d = describe(pair)
    series = load_series([d["a"], d["b"]])
    sa, sb = series[d["a"]].dropna().sort_index(), series[d["b"]].dropna().sort_index()

    if getattr(facts, "dates", None):
        # authoritative: the calendar the predicate intervals are indexed by
        day_of = {int(k): pd.Timestamp(v) for k, v in facts.dates.items()}
        days = sorted(k for k in day_of if d0 <= k < d1)
        idx = pd.DatetimeIndex([day_of[k] for k in days])
    else:
        idx = sa.index.intersection(sb.index).sort_values()[d0:d1]
        days = list(range(d0, d0 + len(idx)))

    a = sa.reindex(idx).to_numpy(float)
    b = sb.reindex(idx).to_numpy(float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    days = [k for k, ok in zip(days, keep) if ok]
    idx = idx[keep]

    from tslearning.llm import prompt as P
    return P.serialise_series(a, b, facts.x, facts.y, days=days, dates=idx,
                              max_days=max_days)


def propose(pair: str, provider: str = "ollama", model: "str | None" = None,
            source: str = "py", mode: str = "facts", temperature: float = 0.0,
            max_days: int = 1800, num_ctx: int = 32768, json_mode: bool = True,
            save_prompt=None,
            response_file=None, out_dir=None, dry_run: bool = False,
            verbose: bool = True) -> dict:
    """Ask a model for rules on the training period, then score them on test.

    `dry_run=True` builds and saves the prompt and stops — no provider is contacted, so
    the prompt can be inspected, version-controlled, or pasted into a chat window by hand.
    """
    import numpy as np

    from tslearning.llm import client, prompt as P, propose as prop
    from tslearning.rules import scoring
    from tslearning.rules.generate import load_pair_facts, normalize_pair

    pair = normalize_pair(pair, source)
    facts = load_pair_facts(pair, source, verbose=False)
    d0, d1, lo, hi = _train_window(facts)
    train_facts = scoring.clip_facts(facts, d0, d1)

    if mode == "facts":
        block = P.serialise_facts(train_facts.intervals)
    elif mode == "series":
        block = series_block(pair, facts, d0, d1, max_days=max_days)
    else:
        raise ValueError("mode must be 'facts' or 'series'")

    text = P.propose_prompt(facts.x, facts.y, block, mode, lo, hi)
    if dry_run and not save_prompt:
        # a dry run exists to produce the prompt: never let it vanish
        save_prompt = Path(out_dir or PATHS["results"]) / \
            f"llm_prompt_{pair}_{mode}_{_timestamp()}_dryrun.txt"
    if save_prompt:
        Path(save_prompt).write_text(text)
        if verbose:
            print(f"prompt -> {save_prompt} ({len(text)} chars)")

    if dry_run:
        if verbose:
            print(f"\ndry run — nothing sent.\n"
                  f"  pair        {pair}  ({facts.x} vs {facts.y})\n"
                  f"  train       {lo} .. {hi}  (days {d0}..{d1})\n"
                  f"  input       {mode}, {len(block.splitlines())} lines\n"
                  f"  prompt      {len(text)} chars\n"
                  f"  predicates  {len(train_facts.intervals)} present in the train slice")
        return {"prompt": text, "rules": [], "rejected": [], "path": save_prompt,
                "completion": None, "dry_run": True}

    got = client.complete(text, provider=provider, model=model,
                          temperature=temperature, num_ctx=num_ctx,
                          json_mode=json_mode, path=response_file, verbose=verbose)
    payload = client.extract_json(got["text"])

    known = set(CFG["predicates"]["kind"])
    rules, bad = prop.to_rules(payload, pair, facts.x, facts.y, known)
    if verbose:
        print(f"\n{len(rules)} rule(s) parsed, {len(bad)} rejected before scoring")
        for b in bad:
            print(f"  REJECTED {b['name']}: {b['reason']}")
        print()
    rules = prop.score_llm_rules(rules, facts, verbose=verbose)

    out_dir = Path(out_dir or PATHS["results"])
    tag = f"llm_{pair}_{got['provider']}_{(got['model'] or 'model').replace('/','-')}"
    ts = _timestamp()

    # Always save the prompt next to the results so it's never lost.
    # The timestamp matches the result file so they're always paired.
    prompt_path = out_dir / f"llm_prompt_{pair}_{mode}_{ts}.txt"
    prompt_path.write_text(text)
    if verbose:
        print(f"prompt -> {prompt_path}")

    result_path = out_dir / f"{tag}_{ts}.json"
    path = prop.save(rules, bad, {
        "pair": pair, "source": source, "input_mode": mode,
        "provider": got["provider"], "model": got["model"],
        "temperature": got["temperature"], "seconds": got["seconds"],
        "json_mode": json_mode, "num_ctx": num_ctx,
        "train": [lo, hi], "test": CFG["split"]["test"],
        "n_proposed": len(rules) + len(bad), "n_parsed": len(rules),
        "n_rejected": len(bad),
        "prompt_hash": _prompt_hash(text),
        "prompt_chars": len(text),
    }, result_path)
    if verbose:
        print(f"\nwrote -> {path}")
    return {"rules": rules, "rejected": bad, "path": path, "completion": got}


def rescore(paths, verbose: bool = True) -> list:
    """Re-parse and re-score SAVED proposals with the current parser and scorer.

    LLM rules are scored once, when the model answers; a later parser/scorer fix would
    otherwise never reach them. The model is NOT called: the same rules (body, head,
    asserted constants) go through `to_rules` + `score_llm_rules` again. The original
    file is kept as <name>.json.bak; `meta.rescored` records when. A rule the current
    parser rejects moves to `rejected`, with its reason.
    """
    import json
    import re
    import shutil

    from tslearning.llm import propose as prop
    from tslearning.rules.generate import load_pair_facts

    known = set(CFG["predicates"]["kind"])
    done = []
    for p in map(Path, paths):
        rec = json.loads(p.read_text())
        meta = rec["meta"]
        facts = load_pair_facts(meta["pair"], meta.get("source", "py"), verbose=False)
        payload = {"rules": [
            {"name": a["template"].removeprefix("LLM_"),
             "head": re.sub(r"[{}]", "", a.get("head") or ""),
             "body": re.sub(r"[{}]", "", a.get("body") or ""),
             "constants": (a.get("diagnostics") or {}).get("declared_constants")
                          or a.get("constants"),
             "phenomenon": (a.get("diagnostics") or {}).get("phenomenon"),
             "rationale": (a.get("diagnostics") or {}).get("rationale")}
            for a in rec.get("accepted", [])]}
        rules, bad = prop.to_rules(payload, meta["pair"], facts.x, facts.y, known)
        rules = prop.score_llm_rules(rules, facts, verbose=False)
        bak = p.with_suffix(".json.bak")
        if not bak.exists():
            shutil.copy2(p, bak)
        meta = dict(meta, n_parsed=len(rules),
                    n_rejected=len(rec.get("rejected", [])) + len(bad),
                    rescored=_timestamp())
        prop.save(rules, list(rec.get("rejected", [])) + bad, meta, p)
        if verbose:
            print(f"rescored {p.name}: {len(rules)} rule(s)"
                  + (f", {len(bad)} now rejected" if bad else ""))
        done.append(p)
    return done


def explain(pairs=None, provider: str = "ollama", model: "str | None" = None,
            rules_csv=None, out_path=None, verbose: bool = True) -> dict:
    """Hand the model the FINAL scored rules and ask for plain-language explanations."""
    import pandas as pd

    from tslearning.llm import client, prompt as P

    rules_csv = Path(rules_csv or (Path(PATHS["results"]) / "rule_results.csv"))
    if not rules_csv.exists():
        raise FileNotFoundError(f"{rules_csv} not found — run scripts/rule_table.py first")
    df = pd.read_csv(rules_csv)
    if pairs:
        # the table holds display names ("S&P 500 – Gold"); accept slugs too
        from tslearning.rules.tables import pair_label
        wanted = {pair_label(p) for p in pairs}
        df = df[df.Pair.isin(wanted)]
        if df.empty:
            raise ValueError(f"no rows for pairs {sorted(wanted)} in {rules_csv.name}")
    scored = df[df["Test"] != "—"]
    if scored.empty:
        raise ValueError("no scored rules to explain")

    block = "\n".join(
        f"- {r.Rule} on {r.Pair}: {r.Constants} | test confidence {r.Test}, "
        f"baseline {r.Base}, lift {r.Lift}, n={r.n}"
        for r in scored.itertuples(index=False))
    text = P.explain_prompt(block)
    got = client.complete(text, provider=provider, model=model, verbose=verbose)

    # timestamped like the propose runs, so a new explanation never replaces an old one
    out_path = Path(out_path or (Path(PATHS["results"]) /
                                 f"llm_explanations_{got['provider']}_{_timestamp()}.md"))
    out_path.write_text(
        f"# LLM explanations of the mined rules\n\n"
        f"Model: {got['provider']}/{got['model']} · "
        f"temperature {got['temperature']} · {len(scored)} rules\n\n"
        f"---\n\n{got['text']}\n")
    if verbose:
        print(f"wrote -> {out_path}")
    return {"path": out_path, "completion": got, "n_rules": len(scored)}
