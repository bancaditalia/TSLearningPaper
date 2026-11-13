# Results

**The paper's numbers are in [`runs/main/`](runs/main/).** The tables as they appear in the paper
are in [`../paper_tables/`](../paper_tables/) (rebuilt by `python scripts/paper_final.py`).

One folder per run in `runs/` (and in `data/processed/runs/` for the run's intermediate
files). Every script takes the run's name: `--run main`, `--run window25`, ...

| run | what it is | train / test | window | min_w | pairs (S&P 500 with) |
|---|---|---|---|---|---|
| **`main`** | **the paper's results** | 2015–21 / 2022–24 | 50 | 1 | gold, bitcoin, 10y futures, TLT, 10y zero-coupon |
| `main_minw0` | same with min_w 0; its classifier is the one `main` uses | 2015–21 / 2022–24 | 50 | 0 | same |
| `main_mind20` | sensitivity: rules need 20 condition days instead of 30 | 2015–21 / 2022–24 | 50 | 1 | same |
| `window25` | sensitivity: shorter window | 2015–21 / 2022–24 | 25 | 1 | same |
| `window25_minw0` | sensitivity: shorter window, min_w 0 | 2015–21 / 2022–24 | 25 | 0 | same |
| `window100` | sensitivity: longer window | 2015–21 / 2022–24 | 100 | 1 | same |
| `robust_2016-2024` | robustness: a second out-of-sample period, rolling thresholds | 2005–15 / 2016–24 | 50 | 0 | gold, 10y futures, TLT, 10y zero-coupon |

**`compare/`**: tables comparing runs (`main_vs_robustness/`, `windows/`, `min_days/`,
`robustness*.md`), built by `scripts/compare_runs.py`; used for the robustness paragraph.

Every run folder has a `run_config.yaml` with the run's complete configuration and a
`run.json` with its key parameters (and its former long name, e.g.
`tr2015-2021_te2022-2024_w50_s1_minw1_…`).

## Inside a run folder

| file | content |
|---|---|
| `rule_results.md` / `.csv` | rule table, exact (Python) labels |
| `rule_results_classifier.md` / `.csv` | rule table, classifier labels |
| `verification.csv`, `verification_classifier.csv` | the reasoner re-evaluating every rule |
| `agreement_<pair>.csv` | Python vs reasoner regime labels on the test windows |
| `fidelity_real_<pair>.csv` (+ `_report.txt`) | classifier vs reasoner on the test windows |
| `multiple_testing.md` | Bonferroni check |
| `*_filtered.*` | the same, restricted to the pairs the paper reports (`paper_final.pairs`); the files without the suffix hold every pair of the run |
| `simulation/` | simulation study (`summary.csv`, one `reps_*.csv` per setting) |
| `paper/` | the tables in paper format, before they are copied to `paper_tables/` |
| `run_config.yaml` | the run's complete configuration |
| `run.json` | the run's key parameters, fingerprint and history |

## Not here any more

The earlier draft's results (gold, bitcoin, NASDAQ, before the October 2026 fixes) and the
exploration runs not used in the paper are in `_archive/` at the repository root; every
moved path is listed in `_archive/MOVED_2026-11-09.md`.
