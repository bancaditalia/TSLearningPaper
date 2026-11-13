# Multiple testing (E.1)

40 rules scored on the test period (py labels). At 95% about 2.00 interval(s) would exclude zero by chance alone. 3 exclude zero at 95%; 2 still do at the Bonferroni level 99.88% (alpha 0.05/40).

| Rule | Pair | Lift | 95% CI | Bonferroni 99.88% CI | excl. 0 (95%) | excl. 0 (Bonf.) | note |
|---|---|---|---|---|---|---|---|
| T1 | sp500_gold | +0.130 | [-0.097, +0.490] | [-0.097, +0.746] | no | no | 3 episode(s), block bootstrap |
| T3 | sp500_gold | -0.028 | [-0.183, +0.198] | [-0.289, +0.293] | no | no | 7 episode(s), episode bootstrap |
| T4 | sp500_gold | -0.073 | [-0.398, +0.225] | [-0.570, +0.322] | no | no | 2 episode(s), block bootstrap |
| T7 | sp500_gold | +0.115 | [-0.134, +0.572] | [-0.229, +0.650] | no | no | 6 episode(s), episode bootstrap |
| T1b | sp500_gold | +0.306 | [-0.229, +0.722] | [-0.229, +0.765] | no | no | 5 episode(s), episode bootstrap |
| T3b | sp500_gold | -0.017 | [-0.186, +0.156] | [-0.270, +0.359] | no | no | 9 episode(s), episode bootstrap |
| T4b | sp500_gold | -0.026 | [-0.246, +0.284] | [-0.246, +0.470] | no | no | 9 episode(s), episode bootstrap |
| T7b | sp500_gold | +0.164 | [-0.141, +0.352] | [-0.402, +0.465] | no | no | 7 episode(s), episode bootstrap |
| T1 | sp500_bitcoin | -0.027 | [-0.027, -0.027] | [-0.027, -0.027] | no | no | zero-width CI |
| T3 | sp500_bitcoin | +0.184 | [-0.091, +0.367] | [-0.187, +0.437] | no | no | 6 episode(s), episode bootstrap |
| T4 | sp500_bitcoin | -0.092 | [-0.243, +0.180] | [-0.243, +0.384] | no | no | 2 episode(s), block bootstrap |
| T7 | sp500_bitcoin | +0.135 | [-0.269, +0.326] | [-0.461, +0.326] | no | no | 5 episode(s), episode bootstrap |
| T1b | sp500_bitcoin | -0.229 | — | — | no | no |  |
| T3b | sp500_bitcoin | +0.355 | [+0.264, +0.412] | [+0.215, +0.422] | yes | yes | 7 episode(s), episode bootstrap |
| T4b | sp500_bitcoin | -0.246 | [-0.246, -0.246] | [-0.246, -0.246] | no | no | zero-width CI |
| T7b | sp500_bitcoin | +0.077 | [-0.361, +0.287] | [-0.383, +0.302] | no | no | 7 episode(s), episode bootstrap |
| T1 | sp500_ust10y | +0.079 | [-0.070, +0.343] | [-0.172, +0.430] | no | no | 3 episode(s), block bootstrap |
| T3 | sp500_ust10y | +0.311 | — | — | no | no |  |
| T4 | sp500_ust10y | +0.026 | [-0.292, +0.239] | [-0.485, +0.358] | no | no | 2 episode(s), block bootstrap |
| T7 | sp500_ust10y | -0.010 | [-0.152, +0.191] | [-0.245, +0.242] | no | no | 6 episode(s), episode bootstrap |
| T1b | sp500_ust10y | +0.081 | [-0.163, +0.523] | [-0.229, +0.771] | no | no | 6 episode(s), episode bootstrap |
| T3b | sp500_ust10y | +0.060 | [-0.060, +0.237] | [-0.060, +0.357] | no | no | 14 episode(s), episode bootstrap |
| T4b | sp500_ust10y | +0.011 | [-0.202, +0.264] | [-0.246, +0.392] | no | no | 10 episode(s), episode bootstrap |
| T7b | sp500_ust10y | -0.011 | [-0.265, +0.303] | [-0.412, +0.429] | no | no | 11 episode(s), episode bootstrap |
| T1 | sp500_tlt | -0.000 | [-0.105, +0.088] | [-0.105, +0.227] | no | no | 3 episode(s), block bootstrap |
| T3 | sp500_tlt | -0.147 | [-0.385, +0.175] | [-0.455, +0.370] | no | no | 4 episode(s), block bootstrap |
| T4 | sp500_tlt | +0.259 | [+0.108, +0.371] | [-0.006, +0.388] | yes | no | 2 episode(s), block bootstrap |
| T7 | sp500_tlt | -0.028 | [-0.177, +0.134] | [-0.370, +0.326] | no | no | 6 episode(s), episode bootstrap |
| T1b | sp500_tlt | -0.001 | [-0.229, +0.371] | [-0.229, +0.514] | no | no | 5 episode(s), episode bootstrap |
| T3b | sp500_tlt | -0.044 | [-0.158, +0.147] | [-0.158, +0.252] | no | no | 11 episode(s), episode bootstrap |
| T4b | sp500_tlt | +0.104 | [-0.173, +0.377] | [-0.246, +0.506] | no | no | 10 episode(s), episode bootstrap |
| T7b | sp500_tlt | -0.037 | [-0.333, +0.378] | [-0.440, +0.509] | no | no | 7 episode(s), episode bootstrap |
| T1 | sp500_ust10z | +0.009 | [-0.165, +0.275] | [-0.191, +0.374] | no | no | 3 episode(s), block bootstrap |
| T3 | sp500_ust10z | +0.584 | — | — | no | no |  |
| T4 | sp500_ust10z | +0.256 | [+0.196, +0.402] | [+0.106, +0.424] | yes | yes | 2 episode(s), block bootstrap |
| T7 | sp500_ust10z | -0.024 | [-0.237, +0.244] | [-0.312, +0.316] | no | no | 6 episode(s), episode bootstrap |
| T1b | sp500_ust10z | +0.010 | [-0.215, +0.453] | [-0.228, +0.772] | no | no | 6 episode(s), episode bootstrap |
| T3b | sp500_ust10z | +0.047 | [-0.033, +0.159] | [-0.033, +0.238] | no | no | 16 episode(s), episode bootstrap |
| T4b | sp500_ust10z | +0.109 | [-0.163, +0.423] | [-0.244, +0.612] | no | no | 11 episode(s), episode bootstrap |
| T7b | sp500_ust10z | -0.031 | [-0.197, +0.360] | [-0.355, +0.502] | no | no | 10 episode(s), episode bootstrap |
