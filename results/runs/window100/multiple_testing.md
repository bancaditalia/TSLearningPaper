# Multiple testing (E.1)

25 rules scored on the test period (py labels). At 95% about 1.25 interval(s) would exclude zero by chance alone. 4 exclude zero at 95%; 4 still do at the Bonferroni level 99.80% (alpha 0.05/25).

| Rule | Pair | Lift | 95% CI | Bonferroni 99.80% CI | excl. 0 (95%) | excl. 0 (Bonf.) | note |
|---|---|---|---|---|---|---|---|
| T4 | sp500_gold | +0.349 | [-0.084, +0.751] | [-0.134, +0.841] | no | no | 1 episode(s), block bootstrap |
| T7 | sp500_gold | -0.003 | [-0.125, +0.347] | [-0.125, +0.691] | no | no | 6 episode(s), episode bootstrap |
| T4b | sp500_gold | +0.006 | [-0.279, +0.371] | [-0.279, +0.583] | no | no | 3 episode(s), block bootstrap |
| T7b | sp500_gold | +0.213 | — | — | no | no |  |
| T2 | sp500_bitcoin | +0.419 | [+0.419, +0.419] | [+0.419, +0.419] | no | no | zero-width CI |
| T3 | sp500_bitcoin | +0.445 | [+0.365, +0.578] | [+0.255, +0.578] | yes | yes | 1 episode(s), block bootstrap |
| T4 | sp500_bitcoin | -0.162 | [-0.162, -0.162] | [-0.162, -0.162] | no | no | zero-width CI |
| T7 | sp500_bitcoin | +0.127 | [-0.484, +0.416] | [-0.581, +0.419] | no | no | 5 episode(s), episode bootstrap |
| T2b | sp500_bitcoin | +0.204 | [-0.242, +0.663] | [-0.282, +0.718] | no | no | 2 episode(s), block bootstrap |
| T3b | sp500_bitcoin | +0.496 | [+0.497, +0.531] | [+0.487, +0.531] | yes | yes | 1 episode(s), block bootstrap |
| T4b | sp500_bitcoin | -0.279 | — | — | no | no |  |
| T7b | sp500_bitcoin | +0.083 | [-0.252, +0.423] | [-0.345, +0.555] | no | no | 2 episode(s), block bootstrap |
| T1 | sp500_ust10y | -0.065 | [-0.065, -0.065] | [-0.065, -0.065] | no | no | zero-width CI |
| T4 | sp500_ust10y | -0.152 | [-0.511, +0.119] | [-0.511, +0.204] | no | no | 1 episode(s), block bootstrap |
| T7 | sp500_ust10y | -0.126 | [-0.270, +0.182] | [-0.270, +0.333] | no | no | 6 episode(s), episode bootstrap |
| T3b | sp500_ust10y | +0.000 | [+0.000, +0.000] | [+0.000, +0.000] | no | no | zero-width CI |
| T4b | sp500_ust10y | -0.083 | [-0.279, +0.094] | [-0.279, +0.294] | no | no | 2 episode(s), block bootstrap |
| T7b | sp500_ust10y | -0.235 | [-0.497, -0.152] | [-0.497, -0.067] | yes | yes | 1 episode(s), block bootstrap |
| T1 | sp500_tlt | +0.069 | [-0.027, +0.073] | [-0.027, +0.138] | no | no | 1 episode(s), block bootstrap |
| T4 | sp500_tlt | +0.423 | [+0.266, +0.426] | [+0.266, +0.481] | yes | yes | 1 episode(s), block bootstrap |
| T3b | sp500_tlt | +0.000 | [+0.000, +0.000] | [+0.000, +0.000] | no | no | zero-width CI |
| T4b | sp500_tlt | +0.272 | [-0.199, +0.721] | [-0.279, +0.721] | no | no | 2 episode(s), block bootstrap |
| T1 | sp500_ust10z | +0.000 | [+0.000, +0.000] | [+0.000, +0.000] | no | no | zero-width CI |
| T4 | sp500_ust10z | +0.213 | [-0.138, +0.637] | [-0.218, +0.637] | no | no | 1 episode(s), block bootstrap |
| T4b | sp500_ust10z | +0.166 | [-0.282, +0.693] | [-0.282, +0.718] | no | no | 2 episode(s), block bootstrap |
