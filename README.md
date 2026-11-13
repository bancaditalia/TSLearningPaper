# TSLearning

Co-movement phenomena between financial time series (contagion, safe haven, decoupling,
lead–lag, macro-conditioned correlation) expressed as temporal rules in DatalogMTL.

- A **DatalogMTL reasoner** (Temporal Vadalog) labels windows of the series with a
  vocabulary of predicates: correlation regime, volatility, trend, co-exceedance.
- A **neural classifier**, distilled from the reasoner on synthetic data, makes the
  correlation labelling fast.
- **Rule templates** drawn from the finance literature are fitted on a training period and
  scored on a held-out test period (confidence, lift, bootstrap confidence interval).
- Every reported rule is **re-evaluated by the reasoner** as an independent check.
- Two further studies: rules proposed by **LLMs**, scored the same way, and a
  **simulation** in which each phenomenon is planted and the rules must find it.

To reproduce the paper's results from a fresh clone, follow
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Repository layout

| folder | content |
|---|---|
| `tslearning/` | the library: data, synthesis, labelling, distillation, rules, evaluation |
| `tslearning/symbolic/` | the DatalogMTL programs run by the reasoner |
| `scripts/` | one command-line script per pipeline step |
| `configs/config.yaml` | all settings; `configs/runs/<run>.yaml` overrides them per run |
| `data/raw/` | downloaded series (filled by `data_prep.py`) |
| `data/processed/runs/<run>/` | intermediate files of each run |
| `results/runs/<run>/` | results of each run; see [`results/README.md`](results/README.md) |
| `results/compare/` | tables comparing runs |
| `paper_tables/` | the tables as they appear in the paper |
| `data_simulation_econ tests/` | econometric tests and the simulated datasets' generator |

## Runs

A **run** is one complete setting (periods, window length, pairs, constants). The paper's
results come from the run `main`; the others are sensitivity and robustness checks.

| run | train / test | window | role |
|---|---|---|---|
| `main` | 2015–2021 / 2022–2024 | 50 | **the paper's results** |
| `main_minw0` | 2015–2021 / 2022–2024 | 50 | as `main` with min_w 0; trains the classifier `main` uses |
| `main_mind20` | 2015–2021 / 2022–2024 | 50 | sensitivity: 20 condition days instead of 30 |
| `window25`, `window25_minw0` | 2015–2021 / 2022–2024 | 25 | sensitivity: shorter window |
| `window100` | 2015–2021 / 2022–2024 | 100 | sensitivity: longer window |
| `robust_2016-2024` | 2005–2015 / 2016–2024 | 50 | robustness: second test period |

Every script takes `--run <run>`. Each run folder holds a `run_config.yaml` with its
complete configuration. A new run is created with
`python scripts/runs.py new --name <name> …` (see `configs/runs/README.md`).

## Setup

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements-frozen.txt
export FRED_API_KEY='...'          # for downloading the macro series
export ANTHROPIC_API_KEY='...'     # only for the LLM study with Claude
```

**Data.** `python scripts/data_prep.py -d all --run <run>` downloads the market and macro
series (Yahoo Finance, FRED) into `data/raw/`; they are not redistributed here. The Fed
yield-curve file is included as `data/raw/feds200628.csv.gz`.

**The reasoner.** Steps marked [ENGINE] need the Temporal Vadalog reasoner, started with
its repository set to this folder. `configs/config.yaml` must point to it:

```yaml
reasoner_endpoint: "http://localhost:8080/evaluateFromRepoWithParams"
reasoner_repository: "."
```

## Pipeline

| step | script | what it does |
|---|---|---|
| 1 | `data_prep.py` | download the series, log returns, windows |
| 2 | `synthesize.py` | train a DDPM per series and sample synthetic windows |
| 3 | `build_realistic_transfer.py` | build labelled synthetic pairs (the transfer set) |
| 4 | `distill.py` | train the classifier on the transfer set |
| 5 | `prep_real_pairs.py` | exact labels of the real pairs, and the macro state |
| 6 | `infer.py` | classifier labels of the real pairs |
| 7 | `generate_rules.py` | fit the templates on the training period, score on the test period |
| 8 | `verify_rules.py` | [ENGINE] re-evaluate every rule with the reasoner |
| 9 | `eval.py` | classifier fidelity and label agreement ([ENGINE] for the reasoner labels) |
| 10 | `llm_rules.py` | rules proposed by an LLM, scored as the templates |
| 11 | `simulate_rules.py` | the simulation study |
| 12 | `multiple_testing.py` | Bonferroni check of the significant rules |

**Every option of every script:** [`SCRIPTS.md`](SCRIPTS.md) (each script's `--help`;
regenerate with `python scripts/make_cli_reference.py`). The options that matter most:
`--run <run>` (every script), `-p/--pairs` (which pairs), `--no-engine` (`paper.py`: skip
the reasoner).

`python scripts/paper.py results --run <run>` runs the rule steps (7, 8, 12), the fidelity
scores and the timing tables in order (add `--no-engine` to skip the reasoner);
`python scripts/paper.py tables --run <run>` writes the run's tables;
`python scripts/paper_final.py` builds `paper_tables/`.

## License

This work is licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
You may share and adapt it for non-commercial purposes, with attribution.

Third-party data (market and macro series, the Fed yield curve) remain under their
providers' terms and are not covered by this license.
