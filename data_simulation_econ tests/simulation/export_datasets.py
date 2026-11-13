"""Write simulated datasets as compressed CSV files for the rule pipeline.

One file per configuration (template, theta, rho0, n), in long format with all replications:
  rep            replication number, 0 to reps-1
  date           business days starting 2000-01-03 (placeholder calendar, same for every replication)
  X, Y           daily log returns (X plays the S&P 500, Y the second asset, gold)
  split          'train' for the first 70% of days, 'test' for 30
  inflationHigh  real-time high-inflation indicator, 0/1 (T7 only; empty for the other templates)
Files: datasets/<set>/<template>_theta<theta>_rho<rho0>_n<n>.csv.gz, plus datasets/<set>/configurations.csv
(one row per file: configuration, seed, train and test days, planted phenomenon, and the detection rate
of the econometric test on the same samples from results_econ.csv, the benchmark for the rules). pandas reads the files
directly: pd.read_csv(path); one replication is df[df.rep == i].

The datasets are identical to those used by mc_econ.py (same deterministic seed per configuration and the
same number of replications, 200), so the rules and the econometric tests are compared on the same samples.

Sets (see SETS):
  core  one configuration per template, chosen from the Monte Carlo of the econometric tests: 10 years
        (756 test days, as the real 2022-24 test window), a baseline rho0 and a strength theta at which the
        test has correct size and power near one half (T2: the largest strength, where the test still has
        power 0.13), plus the same configuration with theta = 0 (no phenomenon: false positives).
        T1b rho0 0.3 theta 0.45 (test power 0.44)   T2 rho0 0.3 theta 0.4 (0.13)
        T3b rho0 0.6 theta 1.0  (0.44; at rho0 0.3 the test's size is 0.16)
        T4  rho0 0.3 theta 0.2  (0.40)              T7 rho0 0.3 theta 0.2 (0.64)
  long  the core configurations (with and without the phenomenon) on 20 and 40 years of daily data
        (1,512 and 3,024 test days): does the method detect more when the sample is longer?
        20 files, about 11.5 MB (20 years) and 23 MB (40 years) each
  grid  every configuration of the Monte Carlo (210 files, about 2.8 GB)

Usage:
  python export_datasets.py [core] [long] [grid] [--force]
  python export_datasets.py T2 0.3 0.3 2520 [reps] [outdir]       one configuration
Existing files are skipped unless --force.
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dgp  # noqa: E402
from mc_econ import config_seed, configs, TRAIN_FRAC  # noqa: E402

REPS = 200                      # as in the Monte Carlo of the tests (results_econ.csv)
OUTDIR = os.path.join(HERE, 'datasets')
PLANTED = {
    'T1b': 'correlation rho0 - theta on days when X falls into its lower 10% tail (safe haven)',
    'T2': 'correlation rho0 + theta when X is in high volatility (contagion)',
    'T3b': 'correlation rho0 * (1 - theta) when both series are calm (decoupling)',
    'T4': 'correlation rho0; in high volatility of X, Y also loads theta on X five days earlier (lead)',
    'T7': 'correlation rho0 + theta when inflation is high, known with a two-month lag (macro)',
}


CORE = {'T1b': (0.3, 0.45), 'T2': (0.3, 0.4), 'T3b': (0.6, 1.0), 'T4': (0.3, 0.2), 'T7': (0.3, 0.2)}  # (rho0, theta)
SETS = {
    'core': [(t, th, r0, 2520) for t, (r0, th1) in CORE.items() for th in (0, th1)],
    'long': [(t, th, r0, n) for n in (5040, 10080) for t, (r0, th1) in CORE.items() for th in (0, th1)],
    'grid': list(configs()),
}


def _test_rate(tpl, theta, rho0, n):
    """Detection rate of the econometric test for this configuration (results_econ.csv), if available."""
    path = os.path.join(HERE, 'results_econ.csv')
    if not os.path.exists(path):
        return np.nan
    r = pd.read_csv(path)
    m = r[(r.template == tpl) & np.isclose(r.theta, theta) & np.isclose(r.rho0, rho0) & (r.n == n)]
    return float(m.detection_rate.iloc[0]) if len(m) else np.nan


def fname(tpl, theta, rho0, n):
    return f'{tpl}_theta{theta}_rho{rho0}_n{n}.csv.gz'


def export(tpl, theta, rho0, n, reps=REPS, outdir=OUTDIR, force=False):
    """Write one configuration; return its row of configurations.csv."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, fname(tpl, theta, rho0, n))
    seed = config_seed(tpl, theta, rho0, n)
    ntr = int(TRAIN_FRAC * n)
    row = {'file': os.path.basename(path), 'template': tpl, 'theta': theta, 'rho0': rho0, 'n': n,
           'years': round(n / 252), 'train_days': ntr, 'test_days': n - ntr, 'reps': reps, 'seed': seed,
           'planted': PLANTED[tpl], 'test_detection_rate': _test_rate(tpl, theta, rho0, n)}
    if os.path.exists(path) and not force:
        return row
    X, Y, I = dgp.simulate(tpl, theta, rho0, n, reps, seed, train_frac=TRAIN_FRAC)
    dates = pd.bdate_range('2000-01-03', periods=n).strftime('%Y-%m-%d')
    df = pd.DataFrame({'rep': np.repeat(np.arange(reps), n), 'date': np.tile(dates, reps),
                       'X': X.ravel(), 'Y': Y.ravel(),
                       'split': np.tile(np.where(np.arange(n) < ntr, 'train', 'test'), reps),
                       'inflationHigh': I.astype(int).ravel() if I is not None else ''})
    tmp = path + '.part'
    df.to_csv(tmp, index=False, float_format='%.8f', compression='gzip')
    os.replace(tmp, path)       # never leave a half-written file under the final name
    return row


def export_set(name, force=False):
    outdir = os.path.join(OUTDIR, name)
    rows = []
    for k, cfg in enumerate(SETS[name], 1):
        rows.append(export(*cfg, outdir=outdir, force=force))
        print(f'{name} {k}/{len(SETS[name])} {rows[-1]["file"]}', flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, 'configurations.csv'), index=False)
    return outdir


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--force']
    force = '--force' in sys.argv[1:]
    if not args:
        args = ['core']
    if args[0] in SETS:
        for name in args:
            print('written', export_set(name, force=force))
    else:
        tpl, theta, rho0, n = args[0], float(args[1]), float(args[2]), int(args[3])
        reps = int(args[4]) if len(args) > 4 else REPS
        outdir = args[5] if len(args) > 5 else os.path.join(OUTDIR, 'single')
        print('written', os.path.join(outdir, export(tpl, theta, rho0, n, reps, outdir, force)['file']))
