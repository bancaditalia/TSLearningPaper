"""Econometric benchmarks for the rule templates 

Train 2015-2021, test 2022-2024.

Pairs: S&P 500 - Gold, S&P 500 - Bitcoin, S&P 500 - Bond

X = S&P 500 throughout.

The triggers reproduce the predicates, with thresholds computed on the training period:
  highVol(X)     at least 30% of the 50-day window above the 90th percentile of |r_X| (train), as in the rules
  lowVol(X)      at most 5% of the window above that percentile
  trendDown(X)   sum of log returns over the window below -0.02
  stressedDown   highVol and trendDown
  tailDependent  at least 2 co-crash days in the window (crash = return below the 5th percentile, train)
  inflationHigh  as in the rules: centred 3-month average of m/m log CPI inflation above its 75th percentile
                 over 2015-2021 months; month m's state on the days of month m (ex post, no publication lag)
A window is labelled at its last day t; day t is a trigger day when the window ending at t satisfies the trigger.

Tests
  T1b safe haven   Baur and Lucey (2010): r_Y = a + b r_X + sum_q c_q r_X 1{r_X < Q_q(r_X)} + e, GARCH(1,1)
                   errors, q = 10%, 5%, 1% within the period.
                   Slope on days below the q% quantile: b + c_10 + ... + c_q (nested dummies).
                   Strong safe haven: that slope significantly negative at some q;
                   not a safe haven: significantly positive at some q; weak otherwise.
  T2 contagion     Forbes and Rigobon (2002): correlation on tailDependent days, corrected for the higher
                   variance of X, against the other days; one-sided Fisher z test.
  T3b decoupling   Correlation on bothCalm days (lowVol X and Y): Fisher z test of zero correlation.
  T4 lead/stress   r_Y,t = a + phi r_Y,t-1 + beta r_X,t-k + gamma r_X,t-k H_t + delta H_t + e, H = highVol(X),
                   HAC standard errors; X leads Y under stress if beta + gamma > 0. k as fitted by the rule.
  T7 macro         Conditional correlation regression on GARCH-standardised returns:
                   z_Y = a + b z_X + c z_X I_t + d I_t + e, I = inflationHigh, HAC standard errors.

Outputs: figures_tables_repository/tab_econ_tests.{csv,tex} (all estimates), tab_econ_tests_gold_revised.csv
(gold tests on the provider-revised series), figures_tables_paper/tab_benchmark.{csv,tex} (rules vs tests)
"""
import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from arch import arch_model

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'market_data')
PAPER, REPO = os.path.join(HERE, 'figures_tables_paper'), os.path.join(HERE, 'figures_tables_repository')
for d in (PAPER, REPO):
    os.makedirs(d, exist_ok=True)

# Sample split, window size, X and Y series, lag for T4, significance level
TRAIN, TEST = ('2015-01-01', '2021-12-31'), ('2022-01-01', '2024-12-31')
PERIODS = {'Train': TRAIN, 'Test': TEST}
W = 50
X = 'SP500' #always in pairs with Y
YS = {'GOLD_FUT': 'Gold', 'BTC': 'Bitcoin', 'ZCB10Y': 'Bond'}
K_RULE = {'GOLD_FUT': 5, 'BTC': 1, 'ZCB10Y': 1}  # T4 lag fitted by the rule (Gold 5, Bitcoin 1); Bond: 1
ALPHA = 0.05

r = pd.read_csv(os.path.join(DATA, 'market_daily_logreturns.csv'), parse_dates=['date'], index_col='date')
st = pd.read_csv(os.path.join(DATA, 'us_inflation_states_daily.csv'), parse_dates=['date'], index_col='date')
cpi = pd.read_csv(os.path.join(DATA, 'us_cpi_monthly.csv'), parse_dates=['month'], index_col='month')
r = r.loc[:TEST[1]]


# ---------------------------------------------------------------- predicates (thresholds from the training period)
def tr(s):
    return s.loc[TRAIN[0]:TRAIN[1]].dropna()


def vol_share(s):
    tau = tr(s).abs().quantile(0.90)
    exceed = (s.abs() > tau).astype(float).where(s.notna())
    return exceed.rolling(W, min_periods=int(0.8 * W)).mean()


def high_vol(s):
    return vol_share(s) >= 0.30 - 1e-9  # at least 30% (tolerance for floating rolling means)


def low_vol(s):
    v = vol_share(s)
    return (v <= 0.05) & v.notna()


def trend_down(s):
    return s.rolling(W, min_periods=int(0.8 * W)).sum() < -0.02


def tail_dependent(x, y):
    cx = (x < tr(x).quantile(0.05)) & x.notna()
    cy = (y < tr(y).quantile(0.05)) & y.notna()
    return (cx & cy).astype(float).rolling(W, min_periods=int(0.8 * W)).sum() >= 2


# Inflation state of the rule pipeline: centred 3-month average of m/m CPI inflation above its 2015-2021
# 75th percentile, month m's state on the days of month m (built in market_data/macro_states.py)
p75 = cpi.loc[TRAIN[0]:TRAIN[1], 'infl_3m_centred_ann_pct'].quantile(0.75)
infl_high = (st['inflationHigh_rules'] == 1).reindex(r.index).fillna(False)


def fisher_ci(rho, n):
    se = 1 / np.sqrt(n - 3)
    z = np.arctanh(rho)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def pstr(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'


# ---------------------------------------------------------------- tests
def t1_baur_lucey(x, y):
    d = pd.concat([x, y], axis=1, keys=['x', 'y']).dropna()
    X_ = pd.DataFrame({'b': d.x})
    for q in [10, 5, 1]:
        X_[f'c{q}'] = d.x * (d.x < d.x.quantile(q / 100))
    res = arch_model(100 * d.y, x=100 * X_, mean='LS', vol='GARCH', p=1, q=1, dist='normal').fit(disp='off')
    P, V = res.params, res.param_cov
    out = {'b': P['b'], 'p_b': 2 * stats.norm.sf(abs(P['b'] / np.sqrt(V.loc['b', 'b'])))}
    # nested dummies (a day below the 1% quantile is also below 5% and 10%): the slope on days below the
    # q% quantile is b plus all the shifts down to q
    for q, names in [(10, ['b', 'c10']), (5, ['b', 'c10', 'c5']), (1, ['b', 'c10', 'c5', 'c1'])]:
        est = P[names].sum()
        se = np.sqrt(V.loc[names, names].values.sum())
        out[f'sh{q}'] = est
        out[f'p_sh{q}'] = 2 * stats.norm.sf(abs(est / se))
    sig_neg = any(out[f'sh{q}'] < 0 and out[f'p_sh{q}'] < ALPHA for q in [10, 5, 1])
    sig_pos = any(out[f'sh{q}'] > 0 and out[f'p_sh{q}'] < ALPHA for q in [10, 5, 1])
    # Baur-Lucey: strong safe haven = significantly negative in stress; weak = not significantly different from zero
    out['verdict'] = ('not a safe haven' if sig_pos else 'strong safe haven' if sig_neg else 'weak safe haven')
    return out


def t2_forbes_rigobon(x, y, trig):
    d = pd.concat([x, y, trig], axis=1, keys=['x', 'y', 't']).dropna()
    T, N = d[d.t == 1], d[d.t == 0]
    if len(T) < 20 or len(N) < 20:
        return {'n_trig': len(T), 'verdict': 'not testable (too few trigger days)'}
    rT, rN = T.x.corr(T.y), N.x.corr(N.y)
    delta = T.x.var() / N.x.var() - 1
    rA = rT / np.sqrt(1 + delta * (1 - rT ** 2))
    z = (np.arctanh(rA) - np.arctanh(rN)) / np.sqrt(1 / (len(T) - 3) + 1 / (len(N) - 3))
    zu = (np.arctanh(rT) - np.arctanh(rN)) / np.sqrt(1 / (len(T) - 3) + 1 / (len(N) - 3))
    p, pu = stats.norm.sf(z), stats.norm.sf(zu)
    verdict = ('contagion' if p < ALPHA else 'interdependence (no contagion after correction)' if pu < ALPHA
               else 'no contagion')
    return {'n_trig': len(T), 'rho_trig': rT, 'rho_other': rN, 'rho_adj': rA, 'p': p, 'p_unadj': pu, 'verdict': verdict}


def t3_decoupling(x, y, trig):
    d = pd.concat([x, y, trig], axis=1, keys=['x', 'y', 't']).dropna()
    T, N = d[d.t == 1], d[d.t == 0]
    if len(T) < 20:
        return {'n_trig': len(T), 'verdict': 'not testable (too few calm days)'}
    rT, rN = T.x.corr(T.y), N.x.corr(N.y)
    p0 = 2 * stats.norm.sf(abs(np.arctanh(rT) * np.sqrt(len(T) - 3)))
    lo, hi = fisher_ci(rT, len(T))
    return {'n_trig': len(T), 'rho_trig': rT, 'ci': (lo, hi), 'rho_other': rN, 'p': p0,
            'verdict': 'decoupled (no correlation in calm)' if p0 >= ALPHA else
                       ('correlated in calm (positive)' if rT > 0 else 'correlated in calm (negative)')}


def t4_lead(x, y, H, k):
    d = pd.DataFrame({'y': y, 'y1': y.shift(1), 'xk': x.shift(k), 'H': H.astype(float)}).dropna()
    d['xkH'] = d.xk * d.H
    m = sm.OLS(d.y, sm.add_constant(d[['y1', 'xk', 'xkH', 'H']])).fit(cov_type='HAC', cov_kwds={'maxlags': 10})
    est = m.params['xk'] + m.params['xkH']
    se = np.sqrt(m.cov_params().loc['xk', 'xk'] + m.cov_params().loc['xkH', 'xkH'] + 2 * m.cov_params().loc['xk', 'xkH'])
    p = 2 * stats.norm.sf(abs(est / se))
    return {'n_trig': int(d.H.sum()), 'k': k, 'beta_stress': est, 'p': p,
            'verdict': ('leads positively under stress' if est > 0 and p < ALPHA else
                        'leads negatively under stress' if est < 0 and p < ALPHA else 'no lead under stress')}


def garch_std(s):
    s = s.dropna()
    res = arch_model(100 * s, mean='Constant', vol='GARCH', p=1, q=1, dist='t').fit(disp='off')
    return res.std_resid


def t7_macro(x, y, I):
    zx, zy = garch_std(x), garch_std(y)
    d = pd.concat([zx, zy, I.astype(float)], axis=1, keys=['zx', 'zy', 'I']).dropna()
    share = d.I.mean()
    if share < 0.05 or share > 0.95:
        return {'share_high': share, 'verdict': 'not testable (inflation state almost constant)'}
    d['zxI'] = d.zx * d.I
    m = sm.OLS(d.zy, sm.add_constant(d[['zx', 'zxI', 'I']])).fit(cov_type='HAC', cov_kwds={'maxlags': 10})
    c, p = m.params['zxI'], m.pvalues['zxI']
    return {'share_high': share, 'rho_low': m.params['zx'], 'rho_high': m.params['zx'] + c, 'c': c, 'p': p,
            'verdict': ('higher correlation in high inflation' if c > 0 and p < ALPHA else
                        'lower correlation in high inflation' if c < 0 and p < ALPHA else 'no inflation effect')}


# 
xs = r[X]
H_x = high_vol(xs)
stressed = H_x & trend_down(xs)
rows = []
for yc, yname in YS.items():
    y = r[yc]
    trig = {'T2': tail_dependent(xs, y), 'T3b': low_vol(xs) & low_vol(y)}
    for per, (a, b) in PERIODS.items():
        xp, yp = xs.loc[a:b], y.loc[a:b]
        res = {
            'T1b': t1_baur_lucey(xp, yp),
            'T2': t2_forbes_rigobon(xp, yp, trig['T2'].loc[a:b].astype(int)),
            'T3b': t3_decoupling(xp, yp, trig['T3b'].loc[a:b].astype(int)),
            'T4': t4_lead(xp, yp, H_x.loc[a:b], K_RULE[yc]),
            'T7': t7_macro(xp, yp, infl_high.loc[a:b]),
        }
        for t, o in res.items():
            rows.append({'template': t, 'pair': f'S&P 500 - {yname}', 'period': per, **o})
E = pd.DataFrame(rows)
E.to_csv(os.path.join(REPO, 'tab_econ_tests.csv'), index=False)

# Gold, provider-revised series (Yahoo GC=F downloaded Oct 2026, revised from May 2020): same tests, repository only.
# The paper uses the vintage of the rule results (GOLD_FUT); this checks how much the vintage matters.
rows_rev = []
y = r['GOLD_FUT_REV']
trig_rev = {'T2': tail_dependent(xs, y), 'T3b': low_vol(xs) & low_vol(y)}
for per, (a, b) in PERIODS.items():
    xp, yp = xs.loc[a:b], y.loc[a:b]
    res = {'T1b': t1_baur_lucey(xp, yp),
           'T2': t2_forbes_rigobon(xp, yp, trig_rev['T2'].loc[a:b].astype(int)),
           'T3b': t3_decoupling(xp, yp, trig_rev['T3b'].loc[a:b].astype(int)),
           'T4': t4_lead(xp, yp, H_x.loc[a:b], K_RULE['GOLD_FUT']),
           'T7': t7_macro(xp, yp, infl_high.loc[a:b])}
    for t, o in res.items():
        rows_rev.append({'template': t, 'pair': 'S&P 500 - Gold (revised series)', 'period': per, **o})
pd.DataFrame(rows_rev).to_csv(os.path.join(REPO, 'tab_econ_tests_gold_revised.csv'), index=False)
print(f'inflation p75 (3m centred, 2015-2021) = {p75:.2f}%')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(E.to_string())


# LaTeX tables
def stars(p):
    return '' if pd.isna(p) else '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''


def num(v, d=2, p=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '--'
    v = round(float(v), d)
    s = f'{0:.{d}f}' if v == 0 else f'{v:.{d}f}'
    s = s.replace('-', '$-$')
    return s + (f'$^{{{stars(p)}}}$' if p is not None and stars(p) else '')


def esc(t):
    return str(t).replace('&', '\\&').replace('%', '\\%')


def table(title, label, header, rows, align, note, colsep='3.5pt'):
    L = ['\\begin{table}[t]', '\\centering', '\\scriptsize', f'\\setlength{{\\tabcolsep}}{{{colsep}}}',
         f'\\caption{{{title}}}', f'\\label{{{label}}}', '\\begin{tabular}{' + align + '}', '\\toprule',
         ' & '.join(header) + ' \\\\', '\\midrule']
    for rw in rows:
        L.append('\\addlinespace[2pt]' if rw == 'SPACE' else ' & '.join(rw) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\par\\smallskip',
          '\\parbox{\\linewidth}{\\footnotesize ' + note + '}', '\\end{table}']
    return '\n'.join(L) + '\n'


SHORT = {'interdependence (no contagion after correction)': 'interdependence',
         'decoupled (no correlation in calm)': 'decoupled', 'correlated in calm (positive)': 'correlated (+)',
         'correlated in calm (negative)': 'correlated ($-$)', 'not testable (too few calm days)': 'not testable',
         'not testable (too few trigger days)': 'not testable', 'higher correlation in high inflation': 'higher in high infl.',
         'lower correlation in high inflation': 'lower in high infl.', 'no inflation effect': 'no effect',
         'leads positively under stress': 'leads (+)', 'leads negatively under stress': 'leads ($-$)',
         'no lead under stress': 'no lead'}
DEF = ('Train 2015--2021, test 2022--2024; thresholds of the triggers from the training period only. '
       'Bond = 10-year zero-coupon Treasury (G\\"urkaynak--Sack--Wright). $^{*}$, $^{**}$, $^{***}$: '
       'significant at 10\\%, 5\\%, 1\\%.')
pairs = [f'S&P 500 - {n}' for n in YS.values()]


def g(t, pair, per):
    return E[(E.template == t) & (E.pair == pair) & (E.period == per)].iloc[0]


def pname(p):
    return esc(p.replace('S&P 500 - ', 'S&P 500 -- '))


# ---------- repository: one table per template
out = []
rows = []
for pair in pairs:
    for per in ['Train', 'Test']:
        o = g('T1b', pair, per)
        rows.append([pname(pair) if per == 'Train' else '', per, num(o.b, 2, o.p_b), num(o.sh10, 2, o.p_sh10),
                     num(o.sh5, 2, o.p_sh5), num(o.sh1, 2, o.p_sh1), SHORT.get(o.verdict, o.verdict)])
out.append(table('Safe haven (T1b): Baur--Lucey regressions', 'tab:econ_t1',
                 ['Pair', 'Period', 'Hedge $b$', 'Slope 10\\%', 'Slope 5\\%', 'Slope 1\\%', 'Verdict'], rows, 'llrrrrl',
                 'GARCH(1,1) regression of the daily return of the second asset on the S\\&P 500 return, with slope shifts '
                 'on the days when the S\\&P 500 return is below its 10\\%, 5\\% or 1\\% quantile in the period '
                 '(Baur and Lucey, 2010). $b$: average slope (hedge if $b \\le 0$). Slope $q$\\%: slope on the days below '
                 'the $q$\\% quantile, $b$ plus the shifts of all nested quantiles down to $q$. '
                 'Strong safe haven: slope significantly negative at some $q$; not a safe haven: significantly '
                 'positive at some $q$; weak safe haven otherwise. ' + DEF))
rows = []
for pair in pairs:
    for per in ['Train', 'Test']:
        o = g('T2', pair, per)
        rows.append([pname(pair) if per == 'Train' else '', per, f'{int(o.n_trig)}', num(o.rho_trig), num(o.rho_other),
                     num(o.rho_adj, 2, o.p), pstr(o.p_unadj).replace('<', '$<$'), SHORT.get(o.verdict, o.verdict)])
out.append(table('Contagion (T2): Forbes--Rigobon test', 'tab:econ_t2',
                 ['Pair', 'Period', 'Days', '$\\rho$ trigger', '$\\rho$ other', '$\\rho^{*}$ trigger', '$p$ unadj.', 'Verdict'],
                 rows, 'llrrrrrl',
                 'Trigger days: the 50-day window ending on the day has at least two co-crash days (tailDependent, '
                 'the trigger of rule T2). $\\rho$: correlation of daily returns on trigger and other days. $\\rho^{*}$: '
                 'trigger correlation corrected for the higher variance of the S\\&P 500 (Forbes and Rigobon, 2002); '
                 'stars from a one-sided Fisher test of $\\rho^{*} > \\rho$ other (contagion). $p$ unadj.: same test '
                 'without correction; significant only before correction = interdependence. ' + DEF))
rows = []
for pair in pairs:
    for per in ['Train', 'Test']:
        o = g('T3b', pair, per)
        ci = o.ci if isinstance(o.ci, tuple) else None
        rows.append([pname(pair) if per == 'Train' else '', per, f'{int(o.n_trig)}', num(o.rho_trig, 2, o.p),
                     num(o.rho_other), SHORT.get(o.verdict, o.verdict)])
out.append(table('Decoupling in calm markets (T3b): correlation on calm days', 'tab:econ_t3',
                 ['Pair', 'Period', 'Days', '$\\rho$ calm', '$\\rho$ other', 'Verdict'], rows, 'llrrrl',
                 'Calm days: the 50-day window ending on the day has low volatility for both series (bothCalm, the '
                 'trigger of rule T3b). Stars test zero correlation on calm days (Fisher); decoupled = not significant. '
                 + DEF))
rows = []
for pair in pairs:
    for per in ['Train', 'Test']:
        o = g('T4', pair, per)
        rows.append([pname(pair) if per == 'Train' else '', per, f'{int(o.k)}', f'{int(o.n_trig)}',
                     num(o.beta_stress, 3, o.p), SHORT.get(o.verdict, o.verdict)])
out.append(table('Lead under stress (T4): predictive regressions', 'tab:econ_t4',
                 ['Pair', 'Period', 'Lag $k$', 'Stress days', '$\\beta+\\gamma$', 'Verdict'], rows, 'llrrrl',
                 'Regression of the daily return of the second asset on its own lag and on the S\\&P 500 return $k$ '
                 'days earlier, with slope shift $\\gamma$ on stress days (highVol of the S\\&P 500, the trigger of rule '
                 'T4); HAC standard errors (10 lags). $\\beta+\\gamma$: lead coefficient under stress. $k$ = lag fitted by '
                 'the rule (Gold 5, Bitcoin 1; Bond 1). ' + DEF))
rows = []
for pair in pairs:
    for per in ['Train', 'Test']:
        o = g('T7', pair, per)
        rows.append([pname(pair) if per == 'Train' else '', per, num(o.share_high), num(o.rho_low), num(o.rho_high),
                     num(o.c, 2, o.p), SHORT.get(o.verdict, o.verdict)])
out.append(table('Macro-conditioned co-movement (T7): conditional correlation', 'tab:econ_t7',
                 ['Pair', 'Period', 'Share high', '$\\rho$ low', '$\\rho$ high', 'Difference', 'Verdict'], rows, 'llrrrrl',
                 'Regression of GARCH(1,1)-standardised daily returns of the second asset on those of the S\\&P 500, '
                 'with slope shift on high-inflation days; HAC standard errors (10 lags). High inflation: centred 3-month '
                 f'average of m/m CPI inflation above its 2015--2021 75th percentile ({p75:.2f}\\%), as in the rules '
                 '(the trigger of rule T7). Share high: share of high-inflation days. $\\rho$ low/high: correlation in '
                 'low/high inflation; difference = $\\rho$ high $-$ $\\rho$ low. ' + DEF))
open(os.path.join(REPO, 'tab_econ_tests.tex'), 'w', encoding='utf-8').write('\n'.join(out))

# ---------- paper: rules (Fig. 4, exact labels) against the econometric tests, test period
RULES = [  # template, pair, rule lift and CI on 2022-2024, rule verdict (paper, Fig. 4)
    ('T1b', 'S&P 500 - Gold', '0.149 [$-$0.298, 0.692]', 'positive lift'),
    ('T1b', 'S&P 500 - Bitcoin', '$-$0.298', 'negative'),
    ('T2', 'S&P 500 - Gold', '0.293 [0.265, 0.375]', 'significant'),
    ('T2', 'S&P 500 - Bitcoin', '0.093 [0.093, 0.093]', 'near-vacuous'),
    ('T3b', 'S&P 500 - Gold', '0.031 [$-$0.202, 0.338]', 'negligible'),
    ('T4', 'S&P 500 - Gold', '0.117 [$-$0.282, 0.508]', 'positive lift'),
    ('T4', 'S&P 500 - Bitcoin', '$-$0.307 [$-$0.307, $-$0.307]', 'negative'),
    ('T7', 'S&P 500 - Gold', '0.071 [$-$0.108, 0.320]', 'positive lift'),
    ('T7', 'S&P 500 - Bitcoin', '0.093 [0.093, 0.093]', 'near-vacuous'),
]
TEST_NAME = {'T1b': 'Baur--Lucey', 'T2': 'Forbes--Rigobon', 'T3b': 'Calm correlation', 'T4': 'Predictive regr.',
             'T7': 'Cond. correlation'}
SUPPORT = {'T1b': {'strong safe haven'}, 'T2': {'contagion'}, 'T3b': {'decoupled (no correlation in calm)'},
           'T4': {'leads positively under stress'}, 'T7': {'higher correlation in high inflation'}}
AGAINST = {'T1b': {'not a safe haven'}, 'T2': set(), 'T3b': {'correlated in calm (positive)', 'correlated in calm (negative)'},
           'T4': {'leads negatively under stress'}, 'T7': {'lower correlation in high inflation'}}


def econ_class(t, v):
    return 'support' if v in SUPPORT[t] else 'against' if v in AGAINST[t] else 'none'


def agree(rule_v, ec):
    if rule_v == 'near-vacuous':
        return 'n.a.'
    if rule_v == 'significant':
        return 'yes' if ec == 'support' else 'no'
    if rule_v in ('positive lift', 'negligible'):  # no significant evidence from the rule
        return 'yes' if ec != 'support' else 'partly'
    if rule_v == 'negative':
        return 'yes' if ec == 'against' else 'partly' if ec == 'none' else 'no'
    return ''


def stat_of(t, o):
    if t == 'T1b':
        k = min([10, 5, 1], key=lambda q: o[f'p_sh{q}'])
        return f'slope {k}\\% = ' + num(o[f'sh{k}'], 2, o[f'p_sh{k}'])
    if t == 'T2':
        return '$\\rho^{*}$ = ' + num(o.rho_adj, 2, o.p) + ' (' + num(o.rho_other) + ')'
    if t == 'T3b':
        return '$\\rho$ calm = ' + num(o.rho_trig, 2, o.p)
    if t == 'T4':
        return '$\\beta+\\gamma$ = ' + num(o.beta_stress, 3, o.p)
    if t == 'T7':
        return '$\\Delta\\rho$ = ' + num(o.c, 2, o.p)


rows, csv = [], []
for t, pair, lift, rv in RULES:
    o = g(t, pair, 'Test')
    ec = econ_class(t, o.verdict)
    a = agree(rv, ec)
    rows.append([t, pname(pair).replace('S\\&P 500 -- ', ''), lift, rv, stat_of(t, o),
                 SHORT.get(o.verdict, o.verdict), a])
    csv.append({'template': t, 'pair': pair, 'rule_lift_ci': re.sub(r'\$-\$', '-', lift), 'rule_verdict': rv,
                'test': TEST_NAME[t].replace('--', '-'), 'test_verdict': o.verdict, 'agreement': a})
pd.DataFrame(csv).to_csv(os.path.join(PAPER, 'tab_benchmark.csv'), index=False)
open(os.path.join(PAPER, 'tab_benchmark.tex'), 'w', encoding='utf-8').write(table(
    'Rule verdicts against econometric tests (test period 2022--2024)', 'tab:benchmark',
    ['', 'Asset', 'Rule lift [CI]', 'Rule verdict', 'Test estimate', 'Test verdict', 'Agree'], rows, 'llllrll',
    'Rule lift and verdict: Fig.~4 of the paper (exact labels, constants fitted on 2015--2021). Each rule is '
    'compared with the standard econometric test of the same phenomenon, estimated on 2022--2024 daily returns '
    'with the trigger of the rule. T1b safe haven: Baur and Lucey (2010) regression with GARCH errors; slope '
    'on the days when the S\\&P 500 return is below its $q$\\% quantile (nested dummies), shown for the $q$ with the '
    'smallest $p$-value. T2 contagion: Forbes and Rigobon (2002) variance-corrected correlation $\\rho^{*}$ on '
    'tail-dependence days (in brackets: correlation on the other days). T3b decoupling: correlation on days when '
    'both series are calm. T4 lead under stress: predictive regression with the lag of the rule, lead coefficient '
    '$\\beta+\\gamma$ on stress days. T7 macro conditioning: difference $\\Delta\\rho$ in conditional '
    'correlation between high- and low-inflation days. Agree: yes '
    'when both find significant evidence in the same direction, or neither finds significant evidence; partly '
    'when only one does; n.a.\\ when the rule is near-vacuous. Second asset vs the S\\&P 500. Full estimates for '
    'both periods and for the Bond pair are in the repository. $^{*}$, $^{**}$, $^{***}$: significant at 10\\%, '
    '5\\%, 1\\%.', colsep='2.2pt'))
print(pd.DataFrame(csv)[['template', 'pair', 'rule_verdict', 'test_verdict', 'agreement']].to_string())