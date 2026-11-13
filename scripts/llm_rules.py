#!/usr/bin/env python3
"""
The LLM comparison arm — an LLM as a rule PROPOSER, scored by the existing machinery.

The point of the experiment: the scoring layer does not care where a rule came from, so a
language model can be judged on exactly the same terms as the template fitter —
fit on train, freeze the constants, score on test, report lift.

Two modes:

  --mode propose   the model sees the TRAINING period only and proposes rules. They are
                   frozen and scored on the unseen test period. This is the comparison.
  --mode explain   the model sees the FINAL scored rules and explains them in plain
                   language. The LLM as explainer, never as miner.

Providers — hook up whichever you have:

  ollama     local, free, nothing leaves the machine, still reproducible in two years.
             `ollama serve` then `ollama pull llama3.1`. Recommended for the paper.
  openai     needs OPENAI_API_KEY.     anthropic  needs ANTHROPIC_API_KEY.
  file       replay a saved completion — develop the parser with no network at all.

Check your institution's policy on outbound API calls before using a hosted provider.

Examples
--------
    # local model, raw daily returns (the default input, the main experiment)
    python scripts/llm_rules.py --mode propose -p sp500_gold --provider ollama

    # like-for-like input: the same predicate facts the template fitter sees (ablation)
    python scripts/llm_rules.py --mode propose -p sp500_gold --provider ollama --input facts

    # save the prompt without calling anything, to inspect or paste elsewhere
    python scripts/llm_rules.py --mode propose -p sp500_gold --dry-run

    # replay a completion you saved by hand
    python scripts/llm_rules.py --mode propose -p sp500_gold \
           --provider file --response-file /tmp/reply.json

    # hosted model
    python scripts/llm_rules.py --mode propose -p sp500_gold \
           --provider openai --model gpt-4o

    # explain the final rules
    python scripts/llm_rules.py --mode explain --provider ollama

    # re-score saved runs after a parser/scorer fix (no model call)
    python scripts/llm_rules.py --rescore results/llm_sp500_gold_*.json

Record the model string AND the date in the paper: hosted model names get reassigned, so
the run is not reproducible without them.
"""
from _cli import bootstrap, csv_list, parser, stage

bootstrap()


def main(argv=None):
    p = parser(__doc__.strip().split("\n")[0], epilog=__doc__)
    p.add_argument("--mode", default="propose", choices=["propose", "explain"],
                   help="propose rules from the train period, or explain the final rules")
    p.add_argument("-p", "--pair", default="sp500_gold",
                   help="pair to propose rules for (default: sp500_gold)")
    p.add_argument("--provider", default="ollama",
                   choices=["ollama", "openai", "anthropic", "file"])
    p.add_argument("--model", default=None,
                   help="model name (default: per provider — llama3.1 / gpt-4o / "
                        "claude-sonnet-4-5)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 by default: a rule set that changes between runs cannot be "
                        "compared with a deterministic fitter (default: 0.0)")
    p.add_argument("--source", default="py", choices=["py", "reasoner", "classifier"],
                   help="which label source provides the facts (default: py)")
    p.add_argument("--input", dest="input_mode", default="series",
                   choices=["series", "facts"],
                   help="series = RAW daily returns, where the pipeline itself starts — "
                        "this is the primary experiment (default). facts = the derived "
                        "predicate intervals, which hands the model the output of the "
                        "labelling stage for free; useful as an ablation to isolate "
                        "whether the model's weakness is labelling or rule proposal")
    p.add_argument("--num-ctx", type=int, default=32768,
                   help="ollama context window. The SERVER default is 4096, which "
                        "silently truncates a prompt this size — the model would answer "
                        "from a fragment and nothing in the reply would say so. Larger "
                        "costs memory for the KV cache (default: 32768)")
    p.add_argument("--no-json-mode", action="store_true",
                   help="do NOT constrain ollama to valid JSON. Off by default: a small\n"
                        "model will otherwise answer a different question entirely "
                        "(llama3.1:8b first replied with Python implementations of the "
                        "predicates). Turn it on to measure whether the model can follow "
                        "the output contract unaided — that failure is itself reportable")
    p.add_argument("--max-days", type=int, default=1800,
                   help="cap on training days sent in series mode, taken from the START "
                        "so the test boundary is never straddled (default: 1800)")
    p.add_argument("--response-file", default=None,
                   help="with --provider file: a saved completion to replay")
    p.add_argument("--save-prompt", default=None,
                   help="write the prompt here as well as sending it")
    p.add_argument("--dry-run", action="store_true",
                   help="build and save the prompt, call nothing")
    p.add_argument("--rules-csv", default=None,
                   help="explain mode: which results table to explain "
                        "(default: results/rule_results.csv)")
    p.add_argument("--pairs", type=csv_list, default=None,
                   help="explain mode: restrict to these pairs")
    p.add_argument("--rescore", nargs="+", default=None, metavar="JSON",
                   help="re-parse and re-score SAVED runs with the current parser and "
                        "scorer, without calling the model (originals kept as .json.bak)")
    p.add_argument("-o", "--out-dir", default=None)
    args = p.parse_args(argv)

    from tslearning.llm import run as llm

    if args.rescore:
        return llm.rescore(args.rescore, verbose=not args.quiet)

    if args.mode == "explain":
        with stage(f"llm explain · {args.provider}", args.quiet):
            r = llm.explain(pairs=args.pairs, provider=args.provider, model=args.model,
                            rules_csv=args.rules_csv, verbose=not args.quiet)
        print(f"explained {r['n_rules']} rules -> {r['path']}")
        return r

    if args.dry_run:
        # default: results/llm_prompt_<pair>_<input>_<timestamp>_dryrun.txt (run.py)
        dest = args.save_prompt
        with stage(f"llm prompt · {args.pair}", args.quiet):
            r = llm.propose(args.pair, source=args.source, mode=args.input_mode,
                            max_days=args.max_days, save_prompt=dest, dry_run=True,
                            verbose=not args.quiet)
        print(f"prompt written to {r['path']} — nothing was sent")
        return r

    with stage(f"llm propose · {args.pair} · {args.provider}", args.quiet):
        r = llm.propose(args.pair, provider=args.provider, model=args.model,
                        source=args.source, mode=args.input_mode,
                        temperature=args.temperature, max_days=args.max_days,
                        num_ctx=args.num_ctx, json_mode=not args.no_json_mode,
                        save_prompt=args.save_prompt,
                        response_file=args.response_file,
                        out_dir=args.out_dir, verbose=not args.quiet)

    scored = [x for x in r["rules"] if x.scores]
    print(f"{len(r['rules'])} parsed ({len(r['rejected'])} rejected), "
          f"{len(scored)} scored -> {r['path']}")
    return r


if __name__ == "__main__":
    main()
