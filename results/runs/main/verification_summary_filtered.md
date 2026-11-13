# Rule verification: Python scorer vs Vadalog

FILTERED to the pairs reported in the paper: sp500_gold, sp500_bitcoin, sp500_tlt, sp500_ust10z. Rows from verification_filtered.csv and verification_classifier_filtered.csv (the unfiltered files, all pairs of the run, are next to them).

| labels | verified | scored by both | identical | differ | max |Δ| |
|---|---|---|---|---|---|
| exact labels | 37 | 34 | 25 | 9 | 0.0081 |
| classifier labels | 34 | 31 | 26 | 5 | 0.0062 |

Rules that differ:

| labels | template | pair | Python | Vadalog | |Δ| |
|---|---|---|---|---|---|
| exact labels | T3_decoupled | sp500_gold | 0.7797 | 0.7753 | 0.0044 |
| exact labels | T7_macroConditioned | sp500_gold | 0.3171 | 0.3117 | 0.0054 |
| exact labels | T7b_macroConditioned_bw | sp500_gold | 0.7205 | 0.7143 | 0.0062 |
| exact labels | T3_decoupled | sp500_bitcoin | 0.5788 | 0.5761 | 0.0027 |
| exact labels | T7b_macroConditioned_bw | sp500_bitcoin | 0.4837 | 0.4776 | 0.0061 |
| exact labels | T7_macroConditioned | sp500_tlt | 0.1816 | 0.1734 | 0.0081 |
| exact labels | T7b_macroConditioned_bw | sp500_tlt | 0.2876 | 0.2832 | 0.0044 |
| exact labels | T7_macroConditioned | sp500_ust10z | 0.0954 | 0.0926 | 0.0027 |
| exact labels | T7b_macroConditioned_bw | sp500_ust10z | 0.2593 | 0.2519 | 0.0074 |
| classifier labels | T1_safeHaven | sp500_gold | 0.2188 | 0.2143 | 0.0045 |
| classifier labels | T7_macroConditioned | sp500_gold | 0.3171 | 0.3117 | 0.0054 |
| classifier labels | T7b_macroConditioned_bw | sp500_gold | 0.7205 | 0.7143 | 0.0062 |
| classifier labels | T3_decoupled | sp500_bitcoin | 0.5788 | 0.5761 | 0.0027 |
| classifier labels | T7b_macroConditioned_bw | sp500_bitcoin | 0.4837 | 0.4776 | 0.0061 |
