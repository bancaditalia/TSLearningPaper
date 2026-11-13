"""Monte Carlo of the econometric benchmark tests on simulated data (size and power).

For each template, effect size theta, baseline correlation rho0 and sample length n, simulate `reps` pairs
(dgp.simulate), split them 70/30 into training and test, compute the triggers with training thresholds
(as on real data, econometric_tests.py) and run the template's test on the test part. Detection = the test
finds the planted phenomenon at the 5% level; a replication with too few trigger days counts as not detected. At theta = 0 the detection rate is the size (false positives).
Results are appended to results_econ.csv (resumable).
"""
import os
import sys
import warnings
import itertools
import zlib
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from arch import arch_model
from multiprocessing import Pool

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dgp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results_econ.csv')
W, ALPHA, TRAIN_FRAC, K = 50, 0.05, 0.7, dgp.LEAD_K
REPS = int(os.environ.get('MC_REPS', 100))
THETA = {'T1b': [0, 0.15, 0.3, 0.45, 0.6], 'T2': [0, 0.1, 0.2, 0.3, 0.4], 'T3b': [0, 0.25, 0.5, 0.75, 1.0],
         'T4': [0, 0.05, 0.1, 0.15, 0.2], 'T7': [0, 0.1, 0.2, 0.3, 0.4]}
RHO0 = [0.0, 0.3, 0.6]
LENGTHS = [2520, 5040, 10080]   # 10, 20, 40 years of trading days


def triggers(x, y, ntr):
    tx, ty = x[:ntr].abs().quantile(0.9), y[:ntr].abs().quantile(0.9)
    vx = (x.abs() > tx).astype(float).rolling(W).mean()
    vy = (y.abs() > ty).astype(float).rolling(W).mean()
    hv = vx >= 0.30 - 1e-9
    qx, qy = x[:ntr].quantile(0.05), y[:ntr].quantile(0.05)
    return {'highVol': hv, 'stressedDown': hv & (x.rolling(W).sum() < -0.02),
            'bothCalm': (vx <= 0.05) & (vy <= 0.05),
            'tailDependent': ((x < qx) & (y < qy)).astype(float).rolling(W).sum() >= 2}


def test_T1b(x, y, trig, I):
    X_ = pd.DataFrame({'b': x})
    for q in [10, 5, 1]:
        X_[f'c{q}'] = x * (x < x.quantile(q / 100))
    try:
        res = arch_model(100 * y, x=100 * X_, mean='LS', vol='GARCH', p=1, q=1).fit(disp='off')
    except Exception:
        return np.nan
    P, V = res.params, res.param_cov
    # nested dummies: the slope on days below the q% quantile is b plus all shifts down to q
    for q, names in [(10, ['b', 'c10']), (5, ['b', 'c10', 'c5']), (1, ['b', 'c10', 'c5', 'c1'])]:
        est = P[names].sum()
        se = np.sqrt(V.loc[names, names].values.sum())
        if est < 0 and 2 * stats.norm.sf(abs(est / se)) < ALPHA:
            return 1.0          # strong safe haven detected
    return 0.0


def test_T2(x, y, trig, I):
    t = trig['tailDependent'].astype(bool)
    T, N = (x[t], y[t]), (x[~t], y[~t])
    if t.sum() < 20 or (~t).sum() < 20:
        return np.nan
    rT, rN = np.corrcoef(*T)[0, 1], np.corrcoef(*N)[0, 1]
    delta = T[0].var() / N[0].var() - 1
    rA = rT / np.sqrt(1 + delta * (1 - rT ** 2))
    z = (np.arctanh(rA) - np.arctanh(rN)) / np.sqrt(1 / (t.sum() - 3) + 1 / ((~t).sum() - 3))
    return float(stats.norm.sf(z) < ALPHA)


def test_T3b(x, y, trig, I):
    t = trig['bothCalm'].astype(bool)
    if t.sum() < 20:
        return np.nan
    rT = np.corrcoef(x[t], y[t])[0, 1]
    p0 = 2 * stats.norm.sf(abs(np.arctanh(rT) * np.sqrt(t.sum() - 3)))
    return float(p0 >= ALPHA)   # decoupling "detected" = no significant correlation in calm


def test_T4(x, y, trig, I):
    d = pd.DataFrame({'y': y, 'y1': y.shift(1), 'xk': x.shift(K), 'H': trig['highVol'].astype(float)}).dropna()
    d['xkH'] = d.xk * d.H
    m = sm.OLS(d.y, sm.add_constant(d[['y1', 'xk', 'xkH', 'H']])).fit(cov_type='HAC', cov_kwds={'maxlags': 10})
    C = m.cov_params()
    est = m.params['xk'] + m.params['xkH']
    se = np.sqrt(C.loc['xk', 'xk'] + C.loc['xkH', 'xkH'] + 2 * C.loc['xk', 'xkH'])
    return float(est > 0 and 2 * stats.norm.sf(abs(est / se)) < ALPHA)


def test_T7(x, y, trig, I):
    if I.mean() < 0.05 or I.mean() > 0.95:
        return np.nan
    try:
        zx = arch_model(100 * x, mean='Constant', vol='GARCH', p=1, q=1, dist='t').fit(disp='off').std_resid
        zy = arch_model(100 * y, mean='Constant', vol='GARCH', p=1, q=1, dist='t').fit(disp='off').std_resid
    except Exception:
        return np.nan
    d = pd.DataFrame({'zx': zx, 'zy': zy, 'I': I}).dropna()
    d['zxI'] = d.zx * d.I
    m = sm.OLS(d.zy, sm.add_constant(d[['zx', 'zxI', 'I']])).fit(cov_type='HAC', cov_kwds={'maxlags': 10})
    return float(m.params['zxI'] > 0 and m.pvalues['zxI'] < ALPHA)


TESTS = {'T1b': test_T1b, 'T2': test_T2, 'T3b': test_T3b, 'T4': test_T4, 'T7': test_T7}


def config_seed(tpl, theta, rho0, n):
    """Deterministic seed of a configuration (same datasets on every run and for the rule pipeline)."""
    return zlib.crc32(f'{tpl}|{theta}|{rho0}|{n}'.encode()) % (2 ** 31)


def run_config(cfg):
    tpl, theta, rho0, n = cfg
    seed = config_seed(tpl, theta, rho0, n)
    X, Y, I = dgp.simulate(tpl, theta, rho0, n, REPS, seed)
    ntr = int(TRAIN_FRAC * n)
    det = []
    for i in range(REPS):
        x, y = pd.Series(X[i]), pd.Series(Y[i])
        trig = triggers(x, y, ntr)
        sl = slice(ntr, n)
        xt, yt = x.iloc[sl].reset_index(drop=True), y.iloc[sl].reset_index(drop=True)
        tt = {k: v.iloc[sl].reset_index(drop=True) for k, v in trig.items()}
        It = pd.Series(I[i][sl]) if I is not None else None
        det.append(TESTS[tpl](xt, yt, tt, It))
    det = np.array(det, dtype=float)
    # not testable (too few trigger days) counts as not detected; the testable share is reported separately
    return {'template': tpl, 'theta': theta, 'rho0': rho0, 'n': n, 'test_days': n - ntr, 'reps': REPS,
            'testable_share': np.isfinite(det).mean(), 'detection_rate': np.nansum(det) / REPS}


def configs():
    for tpl in THETA:
        for theta, rho0, n in itertools.product(THETA[tpl], RHO0, LENGTHS):
            if tpl == 'T3b' and rho0 == 0.0:
                continue   # nothing to decouple when the background correlation is zero
            yield (tpl, theta, rho0, n)


if __name__ == '__main__':
    done = set()
    if os.path.exists(OUT):
        d0 = pd.read_csv(OUT)
        done = set(zip(d0.template, d0.theta, d0.rho0, d0.n))
    todo = [c for c in configs() if c not in done]
    print(f'{len(todo)} configurations to run, {REPS} replications each', flush=True)
    with Pool(2) as pool:
        for k, res in enumerate(pool.imap_unordered(run_config, todo), 1):
            pd.DataFrame([res]).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
            print(k, res['template'], res['theta'], res['rho0'], res['n'], round(res['detection_rate'], 3), flush=True)
