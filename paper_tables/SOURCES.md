# Paper tables — copied 2026-11-13 15:14

Run `main`, pairs: sp500_gold, sp500_bitcoin, sp500_tlt, sp500_ust10z.
Rebuild with:

    python scripts/paper_final.py --run main -p sp500_gold,sp500_bitcoin,sp500_tlt,sp500_ust10z

| file | what | copied from |
|---|---|---|
| rule_results_combined_pub.tex | rule results, exact and classifier labels side by side | `results/runs/main/paper/rule_results_combined_pub.tex` |
| rule_results_pub.tex | rule results, exact labels | `results/runs/main/paper/rule_results_pub.tex` |
| rule_results_classifier_pub.tex | rule results, classifier labels | `results/runs/main/paper/rule_results_classifier_pub.tex` |
| rule_library_box_pub.tex | rule library (templates) | `results/runs/main/paper/rule_library_box_pub.tex` |
| rule_comparison_pub.tex | template rules vs LLM-proposed rules (S&P 500 – gold) | `results/runs/main/paper/rule_comparison_pub.tex` |
| llm_rules_box_pub.tex | the LLM rules shown in the comparison, per model | `results/runs/main/paper/llm_rules_box_pub.tex` |
| distillation.tex | classifier fidelity vs the reasoner | `results/runs/main/paper/distillation.tex` |
| speedup.tex | reasoner vs classifier time | `results/runs/main/paper/speedup.tex` |
| speedup_pub.tex | reasoner vs classifier time: clean timing run + all-runs average | `results/runs/main/paper/speedup_pub.tex` |
| classifier_pub.tex | classifier fidelity (a) and speed (b), side by side | `results/runs/main/paper/classifier_pub.tex` |
| multiple_testing.md | Bonferroni check, exact labels (numbers for the text) | `results/runs/main/paper/multiple_testing.md` |
| multiple_testing_classifier.md | Bonferroni check, classifier labels | `results/runs/main/paper/multiple_testing_classifier.md` |
| verification_summary_filtered.md | rule verification, Python vs Vadalog, reported pairs only | `results/runs/main/paper/verification_summary_filtered.md` |
| agreement_summary_filtered.md | label agreement, Python vs reasoner, reported pairs only | `results/runs/main/paper/agreement_summary_filtered.md` |
| simulation_pub.tex | simulation study (false alarms, power) | `results/runs/main/simulation/simulation_pub.tex` |
