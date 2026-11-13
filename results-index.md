# Where to find the results

## Runs (`results/runs/<run>/`)

| run | differs from `main` in |
|---|---|
| **`main`** | **the paper's results** (train 2015–2021, test 2022–2024, window 50, min_w 1) |
| `main_minw0` | min_w 0; trains the classifier `main` uses |
| `main_mind20` | rules need 20 condition days instead of 30 |
| `window25`, `window25_minw0` | window 25 (and min_w 0) |
| `window100` | window 100 |
| `robust_2016-2024` | train 2005–2015, test 2016–2024, rolling thresholds; no Bitcoin |

Full configuration of each run: `run_config.yaml` (key parameters in `run.json`), in the
run's folder.

## Inside a run folder

| what | file |
|---|---|
| rule scoring, exact labels | `rule_results.md` |
| rule scoring, classifier labels | `rule_results_classifier.md` |
| rule templates | `rule_library.md` |
| LLM rules vs templates (train and test) | `rule_comparison.md` |
| all LLM proposals, train lift | `llm_rules_<timestamps>.txt`; raw runs in `llm_sp500_gold_*.json` |
| verification by the reasoner | `verification_summary_filtered.md` (`verification*.csv` per rule) |
| label agreement, Python vs reasoner | `agreement_summary_filtered.md` (`agreement_<pair>.csv`) |
| classifier fidelity | `fidelity_real_<pair>_report.txt` |
| distillation | `distillation.md` |
| speed | `speedup.md` |
| multiple testing (Bonferroni) | `multiple_testing.md` |
| simulation study | `simulation/simulation_pub.md` (`simulation/summary.csv`) |
| tables in paper format | `paper/`, copied to `paper_tables/` |

`*_filtered.*`: only the pairs the paper reports. `*_pub.*`: the paper's version of a table.

## Comparisons (`results/compare/`)

Each folder has `robustness.md` (exact labels) and `robustness_classifier.md`;
`compare.json` lists the runs compared.

| folder | runs compared |
|---|---|
| `main_vs_robustness/` | `main` vs `robust_2016-2024` |
| `windows/` | `window25`, `main`, `window100` |
| `min_days/` | `main` vs `main_mind20` |
| `robustness*.md` (top level) | `main_minw0` vs `robust_2016-2024` |
