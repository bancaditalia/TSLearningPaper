# Multiple testing (E.1)

40 rules scored on the test period (py labels). At 95% about 2.00 interval(s) would exclude zero by chance alone. 3 exclude zero at 95%; 3 still do at the Bonferroni level 99.88% (alpha 0.05/40).

| Rule | Pair | Lift | 95% CI | Bonferroni 99.88% CI | excl. 0 (95%) | excl. 0 (Bonf.) | note |
|---|---|---|---|---|---|---|---|
| T1 | sp500_gold | +0.129 | [-0.104, +0.483] | [-0.104, +0.745] | no | no | 3 episode(s), block bootstrap |
| T3 | sp500_gold | -0.019 | [-0.173, +0.211] | [-0.286, +0.315] | no | no | 7 episode(s), episode bootstrap |
| T4 | sp500_gold | -0.073 | [-0.392, +0.214] | [-0.569, +0.311] | no | no | 2 episode(s), block bootstrap |
| T7 | sp500_gold | +0.111 | [-0.136, +0.562] | [-0.231, +0.640] | no | no | 6 episode(s), episode bootstrap |
| T1b | sp500_gold | +0.302 | [-0.233, +0.719] | [-0.233, +0.761] | no | no | 5 episode(s), episode bootstrap |
| T3b | sp500_gold | -0.014 | [-0.188, +0.168] | [-0.274, +0.377] | no | no | 9 episode(s), episode bootstrap |
| T4b | sp500_gold | -0.025 | [-0.249, +0.282] | [-0.249, +0.469] | no | no | 9 episode(s), episode bootstrap |
| T7b | sp500_gold | +0.162 | [-0.146, +0.350] | [-0.409, +0.458] | no | no | 7 episode(s), episode bootstrap |
| T1 | sp500_bitcoin | -0.028 | [-0.028, -0.028] | [-0.028, -0.028] | no | no | zero-width CI |
| T3 | sp500_bitcoin | +0.186 | [-0.089, +0.372] | [-0.185, +0.441] | no | no | 6 episode(s), episode bootstrap |
| T4 | sp500_bitcoin | -0.092 | [-0.243, +0.180] | [-0.243, +0.384] | no | no | 2 episode(s), block bootstrap |
| T7 | sp500_bitcoin | +0.130 | [-0.266, +0.318] | [-0.465, +0.318] | no | no | 5 episode(s), episode bootstrap |
| T1b | sp500_bitcoin | -0.233 | — | — | no | no |  |
| T3b | sp500_bitcoin | +0.357 | [+0.271, +0.412] | [+0.224, +0.415] | yes | yes | 7 episode(s), episode bootstrap |
| T4b | sp500_bitcoin | -0.249 | [-0.249, -0.249] | [-0.249, -0.249] | no | no | zero-width CI |
| T7b | sp500_bitcoin | +0.076 | [-0.368, +0.290] | [-0.390, +0.305] | no | no | 7 episode(s), episode bootstrap |
| T1 | sp500_ust10y | +0.088 | [-0.065, +0.362] | [-0.178, +0.454] | no | no | 3 episode(s), block bootstrap |
| T3 | sp500_ust10y | +0.338 | — | — | no | no |  |
| T4 | sp500_ust10y | +0.030 | [-0.298, +0.239] | [-0.491, +0.353] | no | no | 2 episode(s), block bootstrap |
| T7 | sp500_ust10y | -0.006 | [-0.146, +0.194] | [-0.238, +0.259] | no | no | 6 episode(s), episode bootstrap |
| T1b | sp500_ust10y | +0.083 | [-0.163, +0.528] | [-0.233, +0.767] | no | no | 6 episode(s), episode bootstrap |
| T3b | sp500_ust10y | +0.061 | [-0.062, +0.245] | [-0.062, +0.369] | no | no | 14 episode(s), episode bootstrap |
| T4b | sp500_ust10y | +0.012 | [-0.203, +0.267] | [-0.249, +0.393] | no | no | 10 episode(s), episode bootstrap |
| T7b | sp500_ust10y | -0.014 | [-0.271, +0.299] | [-0.419, +0.423] | no | no | 11 episode(s), episode bootstrap |
| T1 | sp500_tlt | -0.006 | [-0.110, +0.083] | [-0.110, +0.221] | no | no | 3 episode(s), block bootstrap |
| T3 | sp500_tlt | -0.162 | [-0.400, +0.160] | [-0.470, +0.355] | no | no | 4 episode(s), block bootstrap |
| T4 | sp500_tlt | +0.263 | [+0.125, +0.365] | [+0.011, +0.376] | yes | yes | 2 episode(s), block bootstrap |
| T7 | sp500_tlt | -0.027 | [-0.180, +0.142] | [-0.379, +0.339] | no | no | 6 episode(s), episode bootstrap |
| T1b | sp500_tlt | -0.005 | [-0.233, +0.367] | [-0.233, +0.510] | no | no | 5 episode(s), episode bootstrap |
| T3b | sp500_tlt | -0.047 | [-0.164, +0.149] | [-0.164, +0.258] | no | no | 11 episode(s), episode bootstrap |
| T4b | sp500_tlt | +0.106 | [-0.175, +0.383] | [-0.249, +0.507] | no | no | 10 episode(s), episode bootstrap |
| T7b | sp500_tlt | -0.044 | [-0.339, +0.371] | [-0.447, +0.503] | no | no | 7 episode(s), episode bootstrap |
| T1 | sp500_ust10z | +0.018 | [-0.166, +0.301] | [-0.199, +0.401] | no | no | 3 episode(s), block bootstrap |
| T3 | sp500_ust10z | +0.564 | — | — | no | no |  |
| T4 | sp500_ust10z | +0.270 | [+0.211, +0.400] | [+0.127, +0.411] | yes | yes | 2 episode(s), block bootstrap |
| T7 | sp500_ust10z | -0.024 | [-0.241, +0.248] | [-0.320, +0.320] | no | no | 6 episode(s), episode bootstrap |
| T1b | sp500_ust10z | +0.013 | [-0.219, +0.470] | [-0.232, +0.768] | no | no | 6 episode(s), episode bootstrap |
| T3b | sp500_ust10z | +0.049 | [-0.035, +0.166] | [-0.035, +0.247] | no | no | 16 episode(s), episode bootstrap |
| T4b | sp500_ust10z | +0.111 | [-0.162, +0.421] | [-0.244, +0.611] | no | no | 11 episode(s), episode bootstrap |
| T7b | sp500_ust10z | -0.034 | [-0.200, +0.353] | [-0.361, +0.496] | no | no | 10 episode(s), episode bootstrap |
