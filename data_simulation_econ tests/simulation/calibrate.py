"""Calibrate the data-generating process of the simulation on the real data used by the rules.

Writes simulation/calibration.json, read by dgp.py:
  X  S&P 500 daily log returns (%), GJR-GARCH(1,1) with Student-t innovations
  Y  gold (COMEX futures, vintage of the rule results: GOLD_FUT), GARCH(1,1) with Student-t innovations
     Both estimated with arch on 2000-2024, on the days on which both returns exist; 'var' is the sample
     variance, used for variance targeting in dgp.py (the fitted models are near-integrated).
  INFL  monthly AR(1) of US CPI inflation as in the rules: month-on-month log change (annualised, %),
        estimated on 1985-2024. dgp.py smooths it with a centred three-month moving average and marks
        high inflation above the 75th percentile of the training months, as the rule pipeline does.
"""
import os
import json
import numpy as np
import pandas as pd
from arch import arch_model

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'market_data')

r = pd.read_csv(os.path.join(DATA, 'market_daily_logreturns.csv'), parse_dates=['date'], index_col='date')
d = 100 * r[['SP500', 'GOLD_FUT']].loc['2000':'2024'].dropna()
cal = {}
for key, col, o in [('X', 'SP500', 1), ('Y', 'GOLD_FUT', 0)]:
    p = arch_model(d[col], mean='Constant', vol='GARCH', p=1, o=o, q=1, dist='t').fit(disp='off').params
    cal[key] = {'mu': round(p['mu'], 4), 'alpha': round(p['alpha[1]'], 4),
                'gamma': round(p['gamma[1]'], 4) if o else 0.0, 'beta': round(p['beta[1]'], 4),
                'nu': round(p['nu'], 4), 'var': round(d[col].var(), 4)}

cpi = pd.read_csv(os.path.join(DATA, 'raw', 'CPIAUCSL.csv'), parse_dates=['date']).set_index('date').iloc[:, 0]
mom = (1200 * np.log(cpi / cpi.shift(1))).loc['1985':'2024'].dropna()
phi, const = np.polyfit(mom.values[:-1], mom.values[1:], 1)
resid = mom.values[1:] - (const + phi * mom.values[:-1])
cal['INFL'] = {'const': round(const, 4), 'phi': round(phi, 4), 'sd': round(resid.std(), 4),
               'measure': 'month-on-month log change of CPIAUCSL, annualised %, 1985-2024'}
cal['sample'] = {'returns': f'{d.index[0].date()} to {d.index[-1].date()}, {len(d)} days', 'gold': 'GOLD_FUT (Jul 2026 vintage)'}

with open(os.path.join(HERE, 'calibration.json'), 'w') as f:
    json.dump(cal, f, indent=1)
print(json.dumps(cal, indent=1))
