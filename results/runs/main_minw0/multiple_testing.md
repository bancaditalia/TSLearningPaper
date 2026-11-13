# Multiple testing (E.1)

36 rules scored on the test period (py labels). At 95% about 1.80 interval(s) would exclude zero by chance alone. 3 exclude zero at 95%; 2 still do at the Bonferroni level 99.86% (alpha 0.05/36).

| Rule | Pair | Lift | 95% CI | Bonferroni 99.86% CI | excl. 0 (95%) | excl. 0 (Bonf.) | note |
|---|---|---|---|---|---|---|---|
| T2 | sp500_gold | +0.443 | [+0.216, +0.786] | [+0.090, +0.786] | yes | yes | 1 episode(s), block bootstrap |
| T3 | sp500_gold | +0.053 | [-0.362, +0.278] | [-0.542, +0.278] | no | no | 4 episode(s), block bootstrap |
| T4 | sp500_gold | +0.117 | [-0.282, +0.508] | [-0.282, +0.718] | no | no | 1 episode(s), block bootstrap |
| T7 | sp500_gold | +0.098 | [-0.167, +0.579] | [-0.214, +0.693] | no | no | 6 episode(s), episode bootstrap |
| T2b | sp500_gold | +0.368 | [-0.105, +0.822] | [-0.178, +0.822] | no | no | 3 episode(s), block bootstrap |
| T3b | sp500_gold | +0.022 | [-0.176, +0.264] | [-0.279, +0.400] | no | no | 4 episode(s), block bootstrap |
| T4b | sp500_gold | +0.132 | [-0.316, +0.559] | [-0.316, +0.684] | no | no | 3 episode(s), block bootstrap |
| T7b | sp500_gold | +0.224 | [-0.118, +0.456] | [-0.211, +0.509] | no | no | 3 episode(s), block bootstrap |
| T2 | sp500_bitcoin | +0.346 | [+0.346, +0.346] | [+0.346, +0.346] | no | no | zero-width CI |
| T3 | sp500_bitcoin | +0.230 | [-0.083, +0.540] | [-0.209, +0.654] | no | no | 3 episode(s), block bootstrap |
| T4 | sp500_bitcoin | -0.307 | [-0.307, -0.307] | [-0.307, -0.307] | no | no | zero-width CI |
| T7 | sp500_bitcoin | +0.162 | [-0.242, +0.346] | [-0.548, +0.346] | no | no | 5 episode(s), episode bootstrap |
| T2b | sp500_bitcoin | +0.154 | [-0.180, +0.464] | [-0.290, +0.634] | no | no | 3 episode(s), block bootstrap |
| T3b | sp500_bitcoin | +0.326 | [+0.231, +0.511] | [+0.084, +0.511] | yes | yes | 2 episode(s), block bootstrap |
| T4b | sp500_bitcoin | -0.316 | [-0.316, -0.316] | [-0.316, -0.316] | no | no | zero-width CI |
| T7b | sp500_bitcoin | +0.095 | [-0.163, +0.333] | [-0.266, +0.476] | no | no | 3 episode(s), block bootstrap |
| T1 | sp500_ust10y | -0.142 | [-0.142, -0.142] | [-0.142, -0.142] | no | no | zero-width CI |
| T4 | sp500_ust10y | -0.051 | [-0.467, +0.283] | [-0.467, +0.497] | no | no | 1 episode(s), block bootstrap |
| T7 | sp500_ust10y | +0.062 | [-0.248, +0.492] | [-0.323, +0.606] | no | no | 6 episode(s), episode bootstrap |
| T1b | sp500_ust10y | -0.298 | [-0.298, -0.298] | [-0.298, -0.298] | no | no | zero-width CI |
| T3b | sp500_ust10y | +0.000 | [+0.000, +0.000] | [+0.000, +0.000] | no | no | zero-width CI |
| T4b | sp500_ust10y | +0.101 | [-0.211, +0.538] | [-0.316, +0.684] | no | no | 5 episode(s), episode bootstrap |
| T7b | sp500_ust10y | +0.094 | [-0.396, +0.404] | [-0.491, +0.509] | no | no | 3 episode(s), block bootstrap |
| T1 | sp500_tlt | +0.056 | [-0.024, +0.081] | [-0.024, +0.189] | no | no | 1 episode(s), block bootstrap |
| T4 | sp500_tlt | +0.185 | [-0.258, +0.438] | [-0.460, +0.438] | no | no | 1 episode(s), block bootstrap |
| T7 | sp500_tlt | -0.127 | [-0.276, +0.109] | [-0.301, +0.247] | no | no | 6 episode(s), episode bootstrap |
| T3b | sp500_tlt | +0.013 | [-0.028, +0.202] | [-0.028, +0.283] | no | no | 5 episode(s), episode bootstrap |
| T4b | sp500_tlt | +0.104 | [-0.192, +0.434] | [-0.316, +0.571] | no | no | 4 episode(s), block bootstrap |
| T7b | sp500_tlt | -0.208 | [-0.450, +0.279] | [-0.487, +0.509] | no | no | 5 episode(s), episode bootstrap |
| T1 | sp500_ust10z | -0.060 | [-0.060, -0.060] | [-0.060, -0.060] | no | no | zero-width CI |
| T4 | sp500_ust10z | +0.177 | [-0.289, +0.461] | [-0.491, +0.491] | no | no | 1 episode(s), block bootstrap |
| T7 | sp500_ust10z | -0.088 | [-0.181, +0.153] | [-0.181, +0.268] | no | no | 6 episode(s), episode bootstrap |
| T1b | sp500_ust10z | -0.300 | — | — | no | no |  |
| T3b | sp500_ust10z | +0.000 | [+0.000, +0.000] | [+0.000, +0.000] | no | no | zero-width CI |
| T4b | sp500_ust10z | +0.116 | [-0.320, +0.537] | [-0.320, +0.680] | no | no | 5 episode(s), episode bootstrap |
| T7b | sp500_ust10z | -0.239 | [-0.491, -0.001] | [-0.491, +0.155] | yes | no | 1 episode(s), block bootstrap |
