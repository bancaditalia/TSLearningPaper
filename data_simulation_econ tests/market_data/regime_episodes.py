"""Correlation-regime episodes and durations per pair, scale and regime

Input : market_daily_logreturns.csv
Output: regime_episodes_summary.csv   one row per pair x window x variant x regime
        regime_reversals.csv           sign reversals (neg->pos, pos->neg) per pair x window x variant

Concordance score on a rolling window of W trading days ending at t (label at t covers [t-W+1, t]):
    S_t = 100 * (n_same_sign - n_opposite_sign) / n_valid        (zero returns excluded)
This is the SIGN part of the paper's mixed concordance; the "same magnitude" component is not
reproduced here, pending the exact definition used in the Vadalog rules. For Gaussian returns
S = 200*arcsin(rho)/pi, so the +-10 deadband corresponds to |rho| < 0.156.

Variants
  deadband   : pos if S > 10, neg if S < -10, else no (paper's rule)
  hysteresis (two thresholds): enter pos at S >= 15 and leave below 5; enter neg at S <= -15 and leave above -5
A window needs at least 80% valid pairs of returns, otherwise the label is missing and the run breaks.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = [('SP500', 'GOLD_FUT'), ('SP500', 'BTC'), ('SP500', 'ZCB10Y'),
         ('GOLD_FUT', 'ZCB10Y')]  # bond = 10-year zero-coupon Treasury, GSW curve (from Aug 1971)
WINDOWS = [50, 250]
TEST_START = '2022-01-01'

r = pd.read_csv(os.path.join(HERE, 'market_daily_logreturns.csv'), parse_dates=['date'], index_col='date')


def score(x, y, w):
    sx, sy = np.sign(x), np.sign(y)
    valid = (sx != 0) & (sy != 0) & sx.notna() & sy.notna()
    same = ((sx == sy) & valid).astype(float).rolling(w).sum()
    opp = ((sx == -sy) & valid).astype(float).rolling(w).sum()
    n = valid.astype(float).rolling(w).sum()
    s = 100 * (same - opp) / n
    return s.where(n >= 0.8 * w)


def label_deadband(s):
    lab = pd.Series(np.where(s > 10, 'pos', np.where(s < -10, 'neg', 'no')), index=s.index)
    return lab.where(s.notna())


def label_hysteresis(s):
    out, cur = [], 'no'
    for v in s.values:
        if np.isnan(v):
            out.append(None); cur = 'no'; continue
        if v >= 15:
            cur = 'pos'
        elif v <= -15:
            cur = 'neg'
        elif cur == 'pos' and v < 5:
            cur = 'no'
        elif cur == 'neg' and v > -5:
            cur = 'no'
        out.append(cur)
    return pd.Series(out, index=s.index)


def episodes(lab):
    lab = lab.dropna()
    # a missing label breaks a run: start a new run whenever the label changes or there is a gap
    pos = pd.Series(np.arange(len(lab)), index=lab.index)
    full = pd.Series(np.arange(len(r)), index=r.index).reindex(lab.index)
    brk = (lab != lab.shift()) | (full.diff() != 1)
    run = brk.cumsum()
    g = pd.DataFrame({'lab': lab, 'run': run})
    ep = g.groupby('run').agg(regime=('lab', 'first'), start=('lab', lambda s: s.index[0]),
                              end=('lab', lambda s: s.index[-1]), length=('lab', 'size'))
    return ep.reset_index(drop=True)


summ, rev = [], []
for a, b in PAIRS:
    for w in WINDOWS:
        s = score(r[a], r[b], w)
        for variant, f in [('deadband', label_deadband), ('hysteresis', label_hysteresis)]:
            lab = f(s)
            ep = episodes(lab)
            first, last = lab.dropna().index[[0, -1]]
            for reg in ['pos', 'no', 'neg']:
                e = ep[ep.regime == reg]
                L = e.length
                summ.append(dict(pair=f'{a}-{b}', window=w, variant=variant, regime=reg,
                                 sample=f'{first.date()} to {last.date()}',
                                 share_days=round((lab == reg).sum() / lab.notna().sum(), 3),
                                 episodes=len(e), episodes_since_2022=int((e.end >= TEST_START).sum()),
                                 episodes_ge_w=int((L >= w).sum()),
                                 median_len=float(L.median()) if len(e) else np.nan,
                                 p25_len=float(L.quantile(.25)) if len(e) else np.nan,
                                 p75_len=float(L.quantile(.75)) if len(e) else np.nan,
                                 max_len=int(L.max()) if len(e) else 0))
            # sign reversals: consecutive non-'no' episodes with opposite sign (a 'no' spell in between is allowed)
            sgn = ep[ep.regime != 'no'].reset_index(drop=True)
            n2p = p2n = 0
            n2p_since = p2n_since = 0
            dates = []
            for i in range(1, len(sgn)):
                prv, cur = sgn.loc[i - 1], sgn.loc[i]
                if prv.regime != cur.regime:
                    since = cur.start >= pd.Timestamp(TEST_START)
                    if cur.regime == 'pos':
                        n2p += 1; n2p_since += since
                    else:
                        p2n += 1; p2n_since += since
                    if w == 250 and prv.length >= 250 and cur.length >= 250:
                        dates.append(f'{prv.regime}->{cur.regime} {cur.start.date()}')
            rev.append(dict(pair=f'{a}-{b}', window=w, variant=variant, neg_to_pos=n2p, pos_to_neg=p2n,
                            neg_to_pos_since_2022=n2p_since, pos_to_neg_since_2022=p2n_since,
                            long_reversals_250=('; '.join(dates) if dates else '')))

S = pd.DataFrame(summ); R = pd.DataFrame(rev)
S.to_csv(os.path.join(HERE, 'regime_episodes_summary.csv'), index=False)
R.to_csv(os.path.join(HERE, 'regime_reversals.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_colwidth', 120)
print(S.drop(columns=['sample']).to_string(index=False))
print(R.to_string(index=False))
print(S.groupby('pair')['sample'].first().to_string())
