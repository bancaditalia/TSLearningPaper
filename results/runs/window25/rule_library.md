| # | Phenomenon | DatalogMTL body (schema) | Fitted | Source |
|---|---|---|---|---|
| T1 | Safe haven | highVol(X), trendDown(X), <+>[0,w] isNegativeCorrelation(X,Y) | w | Baur & Lucey 2010; Baur & McDermott 2010 |
| T1b | Safe haven (backward) | stressedDown(X) :- highVol(X), trendDown(X); isNegativeCorrelation(X,Y), <->[0,w] stressedDown(X) | w | Baur & Lucey 2010; Baur & McDermott 2010 |
| T2 | Contagion / co-exceedance | tailDependent(X,Y), <+>[0,w] isPositiveCorrelation(X,Y) | w | Longin & Solnik 2001; Bae, Karolyi & Stulz 2003 |
| T2b | Contagion / co-exceedance (backward) | isPositiveCorrelation(X,Y), <->[0,w] tailDependent(X,Y) | w | Longin & Solnik 2001; Bae, Karolyi & Stulz 2003 |
| T3 | Decoupling in calm markets | lowVol(X), lowVol(Y), <+>[0,w] isNoCorrelation(X,Y) | w | converse of T2 (Ang & Chen 2002) |
| T3b | Decoupling in calm markets (backward) | bothCalm(X,Y) :- lowVol(X), lowVol(Y); isNoCorrelation(X,Y), <->[0,w] bothCalm(X,Y) | w | converse of T2 (Ang & Chen 2002) |
| T4 | Lead-lag / price discovery under stress | highVol(X), <+>[0,w] leadsPositively(X,Y,k) | w, k | Hasbrouck 1995; de Jong & Nijman 1997 |
| T4b | Lead-lag / price discovery under stress (backward) | leadsPositively(X,Y,k), <->[0,w] highVol(X) | w, k | Hasbrouck 1995; de Jong & Nijman 1997 |
| T5 | Hedge horizon (regime persistence) | isNegativeCorrelation(X,Y), [+][0,d] isNegativeCorrelation(X,Y) | d | Baur & Lucey 2010 (hedge vs. haven) |
| T6 | Correlation regime reversal | [-](0,d⁻] isNegativeCorrelation(X,Y), [+][a,b] isPositiveCorrelation(X,Y) | d⁻, a, b | Ang & Bekaert 2002 (regime switching) |
| T7 | Macro-conditioned co-movement | market(X,C), inflationHigh(C), <+>[0,w] isPositiveCorrelation(X,Y) | w | Bekaert, Hoerova & Lo Duca 2013 |
| T7b | Macro-conditioned co-movement (backward) | isPositiveCorrelation(X,Y), market(X,C), <->[0,w] inflationHigh(C) | w | Bekaert, Hoerova & Lo Duca 2013 |
