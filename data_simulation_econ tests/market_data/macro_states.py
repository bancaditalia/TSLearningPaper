"""Daily US inflation states for the rule miner

Input : us_cpi_monthly.csv, market_daily_levels.csv (for the calendar)  -- built by build.py
Output: us_inflation_states_daily.csv

Real-time alignment. CPI for month m is published by BLS around mid-month m+1 (later in the
1960s-70s). We use a conservative rule:
the value for month m becomes usable on the first trading day of month m+2. 

Columns
  cpi_month            latest CPI month known on that day
  infl_yoy             12-month CPI inflation, %                    (known on that day)
  infl_3m              3-month trailing average of annualised m/m, % (known on that day)
  inflationHigh        main state, with two thresholds (hysteresis): enters when infl_yoy >= 3.0 and exits only
                       when infl_yoy < 2.5, so that inflation hovering around 3% does not
                       produce spurious switches (and spurious onsets)
  inflationHigh_yoy30  robustness: infl_yoy >= 3.0, with one threshold
  inflationHigh_yoy25  robustness: infl_yoy >= 2.5
  inflationHigh_yoy40  robustness: infl_yoy >= 4.0
  inflationHigh_3m30   robustness: infl_3m  >= 3.0 (faster, noisier)
  inflationHigh_p75    original rule: infl_yoy >= 75th pct of the training months
  inflationRising      infl_3m > infl_yoy (short-run inflation above the 12-month rate)
  inflationOnset       1 on the first trading day of each inflationHigh spell
  infl_cma3, inflationHigh_rules   the definition of the rule pipeline, used by the econometric tests and the
                       simulation: centred 3-month average of annualised m/m log CPI inflation, high above the
                       75th percentile of the 2015-2021 months; the state of month m holds on the days of month m
                       (ex post: no publication lag, and it uses month m+1)
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = ('2015-01-01', '2021-12-31')  # months used for the percentile threshold 
MAIN_THR, EXIT_THR = 3.0, 2.5

cpi = pd.read_csv(os.path.join(HERE, 'us_cpi_monthly.csv'), parse_dates=['month'], index_col='month')
cal = pd.read_csv(os.path.join(HERE, 'market_daily_levels.csv'), parse_dates=['date'], usecols=['date'])['date']

# release lag: month m usable from the first day of month m+2
m = cpi[['infl_yoy_pct', 'infl_3m_trailing_ann_pct']].rename(
    columns={'infl_yoy_pct': 'infl_yoy', 'infl_3m_trailing_ann_pct': 'infl_3m'})
m['cpi_month'] = m.index.strftime('%Y-%m')
m.index = m.index + pd.offsets.MonthBegin(2)
m = m[~m.index.duplicated()]
d = m.reindex(m.index.union(pd.DatetimeIndex(cal))).ffill().reindex(pd.DatetimeIndex(cal))
d.index.name = 'date'

p75 = cpi.loc[TRAIN[0]:TRAIN[1], 'infl_yoy_pct'].quantile(0.75)


def state(cond, src):
    return cond.where(src.notna()).astype('Int64')


def hysteresis(x, enter, leave):
    out, cur = [], 0
    for v in x:
        if np.isnan(v):
            out.append(np.nan)
            continue
        cur = 1 if v >= enter else (0 if v < leave else cur)
        out.append(cur)
    return pd.Series(out, index=x.index).astype('Int64')


d['inflationHigh'] = hysteresis(d.infl_yoy, MAIN_THR, EXIT_THR)
d['inflationHigh_yoy30'] = state(d.infl_yoy >= MAIN_THR, d.infl_yoy)
d['inflationHigh_yoy25'] = state(d.infl_yoy >= 2.5, d.infl_yoy)
d['inflationHigh_yoy40'] = state(d.infl_yoy >= 4.0, d.infl_yoy)
d['inflationHigh_3m30'] = state(d.infl_3m >= 3.0, d.infl_3m)
d['inflationHigh_p75'] = state(d.infl_yoy >= p75, d.infl_yoy)
d['inflationRising'] = state(d.infl_3m > d.infl_yoy, d.infl_3m + d.infl_yoy)
h = d['inflationHigh']
d['inflationOnset'] = ((h == 1) & (h.shift(1) == 0)).astype(int)

# Rule-pipeline definition (no lag): month m's centred average and state on the days of month m
cma3 = cpi['infl_3m_centred_ann_pct']
p75_rules = cma3.loc[TRAIN[0]:TRAIN[1]].quantile(0.75)
month_of_day = pd.DatetimeIndex(d.index).to_period('M').to_timestamp()
d['infl_cma3'] = cma3.reindex(month_of_day).values
d['inflationHigh_rules'] = state(d['infl_cma3'] > p75_rules, d['infl_cma3'])
print(f'rules inflation threshold (75th pct of centred 3m, 2015-2021) = {p75_rules:.2f}% p.a.')

d = d[d.infl_yoy.notna()]
d[['cpi_month', 'infl_yoy', 'infl_3m']].round(4).join(
    d.drop(columns=['cpi_month', 'infl_yoy', 'infl_3m'])).to_csv(
    os.path.join(HERE, 'us_inflation_states_daily.csv'))


def spells(s):
    s = s.dropna().astype(int)
    runs = (s != s.shift()).cumsum()
    g = s.groupby(runs)
    lens = g.size()[g.first() == 1]
    return len(lens), lens


rows = []
for col in ['inflationHigh', 'inflationHigh_yoy30', 'inflationHigh_yoy25', 'inflationHigh_yoy40', 'inflationHigh_3m30',
            'inflationHigh_p75', 'inflationRising']:
    n, lens = spells(d[col])
    rows.append((col, round(d[col].mean(), 3), n, int(lens.median()) if n else 0, int(lens.min()) if n else 0))
summ = pd.DataFrame(rows, columns=['state', 'share_of_days', 'spells', 'median_len_days', 'min_len_days'])
print(f'calendar {d.index[0].date()} - {d.index[-1].date()}  |  p75 threshold (train {TRAIN[0][:4]}-{TRAIN[1][:4]}) = {p75:.2f}%')
print(summ.to_string(index=False))
