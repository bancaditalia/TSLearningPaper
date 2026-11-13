Rule results with the exact (Python) labels and with the classifier's labels. Every pair is the S&P 500 against the asset named. Constants are frozen on 2015--2021 and the rules scored on 2022--2024; when the two label sources fit different constants, both are shown (exact / classifier). Lift = test confidence − base rate of the conclusion, with its 95% bootstrap CI. Verdicts: sig. = lift ≥ 0.05 and CI excluding zero; pos. = lift ≥ 0.05, CI including zero; negl. = |lift| < 0.05; neg. = lift ≤ -0.05; inel. = too few condition days or episodes on the test period; — = not fitted on the training period with that label source. Not shown, ineligible under both label sources: T5, T6.

| Rule | Pair | Constants | Lift [CI], exact | Verdict, exact | Lift [CI], classifier | Verdict, classifier |
|---|---|---|---|---|---|---|
| T1b | Gold | w=1 | — | — | +0.154 [-0.299, 0.611] | pos. |
| T1b | US 10y Treasury | w=1 | -0.299 [-0.299, -0.299] | neg. | -0.249 [-0.299, -0.239] | neg. |
| T1b | US 10y zero-coupon (GSW) | w=1 | -0.301 | neg. | -0.240 | neg. |
| T1b | US Treasury 20y+ (TLT) | w=1 | — | — | -0.028 | negl. |
| T2 | Bitcoin | w=1 | +0.343 [0.343, 0.343] | pos. | +0.318 [0.318, 0.318] | pos. |
| T2 | Gold | w=1 | +0.446 [0.212, 0.782] | sig. | +0.614 [0.589, 0.689] | sig. |
| T3b | Bitcoin | w=1 | +0.326 [0.231, 0.507] | sig. | +0.312 [0.202, 0.507] | sig. |
| T3b | Gold | w=1 | +0.024 [-0.177, 0.267] | negl. | — | — |
| T3b | US Treasury 20y+ (TLT) | w=1 | +0.014 [-0.029, 0.211] | negl. | +0.014 [-0.029, 0.083] | negl. |
| T4 | Bitcoin | w=1, k=1 | -0.310 [-0.310, -0.310] | neg. | -0.310 [-0.310, -0.310] | neg. |
| T4 | Gold | w=1, k=5 | +0.115 [-0.285, 0.506] | pos. | +0.115 [-0.285, 0.506] | pos. |
| T4 | US 10y Treasury | w=1, k=2 | -0.049 [-0.469, 0.281] | negl. | -0.049 [-0.469, 0.281] | negl. |
| T4 | US 10y zero-coupon (GSW) | w=2, k=1 | +0.177 [-0.289, 0.461] | pos. | +0.177 [-0.289, 0.461] | pos. |
| T4 | US Treasury 20y+ (TLT) | w=1, k=1 | +0.186 [-0.256, 0.434] | pos. | +0.186 [-0.256, 0.434] | pos. |
| T7 | Bitcoin | w=1 | +0.159 [-0.245, 0.343] | pos. | +0.165 [-0.270, 0.318] | pos. |
| T7 | Gold | w=1 | +0.099 [-0.169, 0.589] | pos. | -0.040 [-0.309, 0.456] | negl. |
| T7 | US 10y Treasury | w=1 | +0.060 [-0.252, 0.493] | pos. | -0.102 [-0.399, 0.448] | neg. |
| T7 | US 10y zero-coupon (GSW) | w=1 | -0.087 [-0.182, 0.161] | neg. | — | — |
| T7 | US Treasury 20y+ (TLT) | w=1 | -0.126 [-0.278, 0.116] | neg. | — | — |
