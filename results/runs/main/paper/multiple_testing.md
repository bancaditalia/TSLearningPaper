# Multiple testing

29 rules scored on the test period (py labels). At 95% about 1.45 interval(s) would exclude zero by chance alone. 2 exclude zero at 95%; 2 still do at the Bonferroni level 99.83% (alpha 0.05/29).

| Rule | Pair | Lift | 95% CI | Bonferroni 99.83% CI | excl. 0 (95%) | excl. 0 (Bonf.) | note |
|---|---|---|---|---|---|---|---|
| T2 | sp500_gold | +0.446 | [+0.212, +0.782] | [+0.089, +0.782] | yes | yes | 1 episode(s), block bootstrap |
| T3 | sp500_gold | +0.054 | [-0.356, +0.274] | [-0.534, +0.274] | no | no | 4 episode(s), block bootstrap |
| T4 | sp500_gold | +0.115 | [-0.285, +0.506] | [-0.285, +0.715] | no | no | 1 episode(s), block bootstrap |
| T7 | sp500_gold | +0.099 | [-0.169, +0.589] | [-0.218, +0.686] | no | no | 6 episode(s), episode bootstrap |
| T2b | sp500_gold | +0.367 | [-0.106, +0.820] | [-0.180, +0.820] | no | no | 3 episode(s), block bootstrap |
| T3b | sp500_gold | +0.024 | [-0.177, +0.267] | [-0.277, +0.395] | no | no | 4 episode(s), block bootstrap |
| T4b | sp500_gold | +0.130 | [-0.318, +0.557] | [-0.318, +0.682] | no | no | 3 episode(s), block bootstrap |
| T7b | sp500_gold | +0.223 | [-0.117, +0.456] | [-0.197, +0.503] | no | no | 3 episode(s), block bootstrap |
| T2 | sp500_bitcoin | +0.343 | [+0.343, +0.343] | [+0.343, +0.343] | no | no | zero-width CI |
| T3 | sp500_bitcoin | +0.230 | [-0.083, +0.537] | [-0.208, +0.652] | no | no | 3 episode(s), block bootstrap |
| T4 | sp500_bitcoin | -0.310 | [-0.310, -0.310] | [-0.310, -0.310] | no | no | zero-width CI |
| T7 | sp500_bitcoin | +0.159 | [-0.245, +0.343] | [-0.551, +0.343] | no | no | 5 episode(s), episode bootstrap |
| T2b | sp500_bitcoin | +0.155 | [-0.178, +0.466] | [-0.291, +0.618] | no | no | 3 episode(s), block bootstrap |
| T3b | sp500_bitcoin | +0.326 | [+0.231, +0.507] | [+0.088, +0.507] | yes | yes | 2 episode(s), block bootstrap |
| T4b | sp500_bitcoin | -0.318 | [-0.318, -0.318] | [-0.318, -0.318] | no | no | zero-width CI |
| T7b | sp500_bitcoin | +0.094 | [-0.165, +0.330] | [-0.269, +0.469] | no | no | 3 episode(s), block bootstrap |
| T1 | sp500_tlt | +0.056 | [-0.024, +0.081] | [-0.024, +0.187] | no | no | 1 episode(s), block bootstrap |
| T4 | sp500_tlt | +0.186 | [-0.256, +0.434] | [-0.439, +0.434] | no | no | 1 episode(s), block bootstrap |
| T7 | sp500_tlt | -0.126 | [-0.278, +0.116] | [-0.307, +0.259] | no | no | 6 episode(s), episode bootstrap |
| T3b | sp500_tlt | +0.014 | [-0.029, +0.211] | [-0.029, +0.293] | no | no | 5 episode(s), episode bootstrap |
| T4b | sp500_tlt | +0.105 | [-0.193, +0.432] | [-0.318, +0.572] | no | no | 4 episode(s), block bootstrap |
| T7b | sp500_tlt | -0.210 | [-0.450, +0.273] | [-0.490, +0.503] | no | no | 5 episode(s), episode bootstrap |
| T1 | sp500_ust10z | -0.062 | [-0.062, -0.062] | [-0.062, -0.062] | no | no | zero-width CI |
| T4 | sp500_ust10z | +0.177 | [-0.289, +0.461] | [-0.484, +0.491] | no | no | 1 episode(s), block bootstrap |
| T7 | sp500_ust10z | -0.087 | [-0.182, +0.161] | [-0.182, +0.279] | no | no | 6 episode(s), episode bootstrap |
| T1b | sp500_ust10z | -0.301 | — | — | no | no | no bootstrap CI |
| T3b | sp500_ust10z | +0.000 | [+0.000, +0.000] | [+0.000, +0.000] | no | no | zero-width CI |
| T4b | sp500_ust10z | +0.114 | [-0.321, +0.536] | [-0.321, +0.679] | no | no | 5 episode(s), episode bootstrap |
| T7b | sp500_ust10z | -0.239 | [-0.498, +0.012] | [-0.498, +0.165] | no | no | 1 episode(s), block bootstrap |
