# Multiple testing

27 rules scored on the test period (classifier labels). At 95% about 1.35 interval(s) would exclude zero by chance alone. 3 exclude zero at 95%; 3 still do at the Bonferroni level 99.81% (alpha 0.05/27).

| Rule | Pair | Lift | 95% CI | Bonferroni 99.81% CI | excl. 0 (95%) | excl. 0 (Bonf.) | note |
|---|---|---|---|---|---|---|---|
| T1 | sp500_gold | +0.083 | [-0.158, +0.472] | [-0.158, +0.658] | no | no | 1 episode(s), block bootstrap |
| T2 | sp500_gold | +0.614 | [+0.589, +0.689] | [+0.537, +0.689] | yes | yes | 1 episode(s), block bootstrap |
| T4 | sp500_gold | +0.115 | [-0.285, +0.506] | [-0.285, +0.715] | no | no | 1 episode(s), block bootstrap |
| T7 | sp500_gold | -0.040 | [-0.309, +0.456] | [-0.311, +0.620] | no | no | 6 episode(s), episode bootstrap |
| T1b | sp500_gold | +0.154 | [-0.299, +0.611] | [-0.299, +0.701] | no | no | 2 episode(s), block bootstrap |
| T2b | sp500_gold | +0.357 | [-0.075, +0.820] | [-0.180, +0.820] | no | no | 3 episode(s), block bootstrap |
| T4b | sp500_gold | +0.130 | [-0.318, +0.557] | [-0.318, +0.682] | no | no | 3 episode(s), block bootstrap |
| T7b | sp500_gold | -0.060 | [-0.307, +0.313] | [-0.440, +0.435] | no | no | 3 episode(s), block bootstrap |
| T1 | sp500_bitcoin | +0.000 | [+0.000, +0.000] | [+0.000, +0.000] | no | no | zero-width CI |
| T2 | sp500_bitcoin | +0.318 | [+0.318, +0.318] | [+0.318, +0.318] | no | no | zero-width CI |
| T3 | sp500_bitcoin | +0.204 | [-0.120, +0.525] | [-0.258, +0.620] | no | no | 3 episode(s), block bootstrap |
| T4 | sp500_bitcoin | -0.310 | [-0.310, -0.310] | [-0.310, -0.310] | no | no | zero-width CI |
| T7 | sp500_bitcoin | +0.165 | [-0.270, +0.318] | [-0.576, +0.318] | no | no | 5 episode(s), episode bootstrap |
| T2b | sp500_bitcoin | +0.139 | [-0.138, +0.428] | [-0.294, +0.539] | no | no | 3 episode(s), block bootstrap |
| T3b | sp500_bitcoin | +0.312 | [+0.202, +0.507] | [+0.035, +0.507] | yes | yes | 2 episode(s), block bootstrap |
| T4b | sp500_bitcoin | -0.318 | [-0.318, -0.318] | [-0.318, -0.318] | no | no | zero-width CI |
| T7b | sp500_bitcoin | +0.092 | [-0.168, +0.332] | [-0.288, +0.452] | no | no | 3 episode(s), block bootstrap |
| T1 | sp500_tlt | -0.004 | [-0.066, +0.084] | [-0.066, +0.134] | no | no | 1 episode(s), block bootstrap |
| T4 | sp500_tlt | +0.186 | [-0.256, +0.434] | [-0.433, +0.434] | no | no | 1 episode(s), block bootstrap |
| T1b | sp500_tlt | -0.028 | — | — | no | no | no bootstrap CI |
| T3b | sp500_tlt | +0.014 | [-0.029, +0.083] | [-0.029, +0.115] | no | no | 7 episode(s), episode bootstrap |
| T4b | sp500_tlt | +0.105 | [-0.193, +0.432] | [-0.318, +0.571] | no | no | 4 episode(s), block bootstrap |
| T1 | sp500_ust10z | -0.089 | [-0.111, -0.086] | [-0.111, -0.066] | yes | yes | 1 episode(s), block bootstrap |
| T4 | sp500_ust10z | +0.177 | [-0.289, +0.461] | [-0.482, +0.491] | no | no | 1 episode(s), block bootstrap |
| T1b | sp500_ust10z | -0.240 | — | — | no | no | no bootstrap CI |
| T3b | sp500_ust10z | +0.000 | [+0.000, +0.000] | [+0.000, +0.000] | no | no | zero-width CI |
| T4b | sp500_ust10z | +0.114 | [-0.321, +0.536] | [-0.321, +0.679] | no | no | 5 episode(s), episode bootstrap |
