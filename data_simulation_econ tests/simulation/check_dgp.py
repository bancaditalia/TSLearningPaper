"""Quality check: simulated vs real moments and trigger frequencies, and planted effects at large theta."""
import sys, os
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dgp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
r = pd.read_csv(os.path.join(HERE, 'market_data', 'market_daily_logreturns.csv'), parse_dates=['date'], index_col='date')
real = r[['SP500', 'GOLD_FUT']].dropna().loc['2000':'2024']


def triggers(x, y, train_frac=0.7):
    x, y = pd.Series(x), pd.Series(y)
    ntr = int(train_frac * len(x))
    tx, ty = x[:ntr].abs().quantile(0.9), y[:ntr].abs().quantile(0.9)
    vx = (x.abs() > tx).rolling(50).mean(); vy = (y.abs() > ty).rolling(50).mean()
    hv = vx >= 0.30 - 1e-9; td = x.rolling(50).sum() < -0.02
    calm = (vx <= 0.05) & (vy <= 0.05)
    qx, qy = x[:ntr].quantile(0.05), y[:ntr].quantile(0.05)
    tail = ((x < qx) & (y < qy)).astype(float).rolling(50).sum() >= 2
    return {'highVol': hv.mean(), 'stressedDown': (hv & td).mean(), 'bothCalm': calm.mean(), 'tailDependent': tail.mean()}


def moments(x):
    x = pd.Series(x)
    return {'vol': x.std() * np.sqrt(252), 'kurt': stats.kurtosis(x), 'ac1_sq': (x ** 2).autocorr(1)}


rows = []
rows.append({'source': 'real 2000-2024', **{f'X_{k}': v for k, v in moments(real.SP500).items()},
             **{f'Y_{k}': v for k, v in moments(real.GOLD_FUT).items()}, **triggers(real.SP500.values, real.GOLD_FUT.values),
             'corr': real.corr().iloc[0, 1]})
x, y, _ = dgp.simulate('T2', 0.0, 0.0, len(real), 20, seed=7)
sim = [dict(**{f'X_{k}': v for k, v in moments(x[i]).items()}, **{f'Y_{k}': v for k, v in moments(y[i]).items()},
            **triggers(x[i], y[i]), corr=np.corrcoef(x[i], y[i])[0, 1]) for i in range(20)]
rows.append({'source': 'simulated null (mean of 20)', **pd.DataFrame(sim).mean().to_dict()})
print(pd.DataFrame(rows).set_index('source').round(3).T.to_string())

# planted effects at large theta: average correlation in the trigger state vs elsewhere
print('\nplanted effects (theta large, rho0 = 0.3), correlation of daily returns in state vs elsewhere')
for tpl, th in [('T1b', 0.6), ('T2', 0.4), ('T3b', 1.0), ('T7', 0.4)]:
    x, y, I = dgp.simulate(tpl, th, 0.3, 2520, 20, seed=11)
    vals = []
    for i in range(20):
        xs, ys = pd.Series(x[i]), pd.Series(y[i])
        tx, ty = xs.abs().quantile(0.9), ys.abs().quantile(0.9)
        vx = (xs.abs() > tx).rolling(50).mean().shift(1); vy = (ys.abs() > ty).rolling(50).mean().shift(1)
        st = {'T1b': (vx >= 0.30 - 1e-9) & (xs.rolling(50).sum().shift(1) < -0.02), 'T2': vx >= 0.30 - 1e-9,
              'T3b': (vx <= 0.05) & (vy <= 0.05), 'T7': pd.Series(I[i]) > 0.5 if I is not None else None}[tpl]
        vals.append((xs[st].corr(ys[st]), xs[~st].corr(ys[~st]), st.mean()))
    v = np.nanmean(np.array(vals), axis=0)
    print(f'{tpl}: rho in state {v[0]:+.2f}, elsewhere {v[1]:+.2f}, share of days in state {v[2]:.2f}')
