"""Build the clean dataset for the temporal-rule-mining

Inputs : raw/<SERIES>.csv  
Outputs: out/market_daily_levels.csv      levels on the NYSE (S&P 500) trading calendar
         out/market_daily_logreturns.csv  daily log returns on the same calendar
         out/us_cpi_monthly.csv           CPI level and inflation measures
         out/raw/*.csv                    raw
"""
import os
import shutil
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'raw')
OUT = HERE  # outputs written next to this script (previous versions kept in _previous_2026-10-17/)
END = '2024-12-31'  # sample end: all outputs stop at 31 Dec 2024

RAW_FILES = {  # file -> (output column, description); GSW zero-coupon curve read separately below
    'GSPC': ('SP500', 'S&P 500 index close (price index, no dividends), Yahoo ^GSPC'),
    'GC': ('GOLD_FUT_REV', 'COMEX gold front-month futures settle, Yahoo GC=F (USD/oz), downloaded Oct 2026: '
                          'series revised by the provider from May 2020 (roll dates)'),
    'BTC': ('BTC', 'Bitcoin close at 00:00 UTC, Yahoo BTC-USD (USD)'),
    # Not used in the paper: raw files moved to notused/ (kept out of git). Uncomment, and move the files back
    # to raw/, to rebuild the ETF total returns, the FRED yields and the par-bond returns below.
    # 'IEF': ('IEF_TR', 'iShares 7-10y Treasury ETF adjusted close (total return), Yahoo IEF'),
    # 'TLT': ('TLT_TR', 'iShares 20+y Treasury ETF adjusted close (total return), Yahoo TLT'),
    # 'DGS2': ('DGS2', '2-year Treasury constant-maturity yield, % p.a., FRED DGS2'),
    # 'DGS10': ('DGS10', '10-year Treasury constant-maturity yield, % p.a., FRED DGS10'),
    # 'DGS30': ('DGS30', '30-year Treasury constant-maturity yield, % p.a., FRED DGS30'),
}


def load(name):
    d = pd.read_csv(os.path.join(RAW, name + '.csv'), parse_dates=['date'])
    return d.set_index('date').iloc[:, 0].astype(float)


raw = {k: load(k) for k in RAW_FILES}
# Gold used in the paper = the vintage of the rule results: Yahoo GC=F downloaded 10 Jul 2026 by the coauthors
# (raw/GC_jul2026.csv, from TSLearning_raw_2026-11-11.zip, data/raw/Gold_raw.csv, 2005-2024). Before 2005 the
# October download is used; the two vintages coincide until May 2020.
gc_jul = load('GC_jul2026')

# Bitcoin: 2026-10-17 was an intraday quote at retrieval time -> drop the incomplete day.
raw['BTC'] = raw['BTC'][raw['BTC'].index < '2026-10-17']

# ---- calendar: NYSE trading days = days with an S&P 500 close -------------------------
cal = raw['GSPC'].dropna().index
cal = cal[cal <= END]
levels = pd.DataFrame(index=cal)
for k, (col, _) in RAW_FILES.items():
    s = raw[k].dropna()
    if k == 'BTC':
        # BTC trades every day; take its 00:00 UTC close dated on each S&P trading day.
        # Weekend/holiday moves are therefore folded into the next trading-day return.
        levels[col] = s.reindex(cal)
    else:
        levels[col] = s.reindex(cal)  # NaN where that market was closed / no print
levels['GOLD_FUT'] = gc_jul.combine_first(raw['GC']).reindex(cal)
levels.index.name = 'date'


# ---- Treasury par-bond total returns from constant-maturity yields ----------------------
def par_bond_return(y_pct, maturity, max_gap_days=7):
    """Daily total return of a par bond: bought at par with coupon y_{t-1} and maturity M,
    repriced at the next observed yield y_t with maturity M - dt, plus coupon accrual.
    Semi-annual compounding, annuity formula with fractional periods (Swinkels 2019 style).

    Returns are computed between consecutive OBSERVED yields on the S&P calendar. A day on which
    the bond market is closed while stocks trade (Columbus Day, Veterans Day) has no return, and
    the next return spans the closure, as over a weekend. Gaps longer than max_gap_days calendar
    days are not bridged (e.g. the 30-year yield is absent from Feb 2002 to Feb 2006, when the
    Treasury suspended 30-year issuance)."""
    y = (y_pct / 100.0).dropna()
    out = pd.Series(np.nan, index=y_pct.index)
    for i in range(1, len(y)):
        t0, t1 = y.index[i - 1], y.index[i]
        if (t1 - t0).days > max_gap_days:
            continue
        y0, y1 = y.iloc[i - 1], y.iloc[i]
        dt = (t1 - t0).days / 365.25
        n = 2 * (maturity - dt)
        c = y0
        price = (c / y1) * (1 - (1 + y1 / 2) ** (-n)) + (1 + y1 / 2) ** (-n) if y1 > 0 else 1 + c / 2 * n
        out.loc[t1] = price - 1 + c * dt
    return out


bond_ret = {}
# Par-bond returns from the FRED yields: not used in the paper (see RAW_FILES above).
# for k, m in [('DGS2', 2), ('DGS10', 10), ('DGS30', 30)]:
#     r = par_bond_return(levels[k], m)  # yields on the trading calendar
#     bond_ret['BOND%dY' % m] = np.log1p(r)

# ---- Treasury zero-coupon returns from Gurkaynak, Sack and Wright (2007) ----------------------
# raw/feds200628.csv: Federal Reserve staff dataset "The U.S. Treasury Yield Curve: 1961 to the
# Present", downloaded 20 Oct 2026. Fitted Svensson curve, continuously compounded zero yields (%).
# GSW report a maturity only when Treasuries of that maturity were outstanding:
# 2y and 5y from Jun 1961, 10y from Aug 1971, 30y from Nov 1985.
# Daily log return of an n-year zero bought at t-1 and valued at t with maturity n - dt:
#     r_t = n * y_n(t-1) - (n - dt) * y_{n-dt}(t)
# where y_{n-dt}(t) is evaluated exactly from the Svensson parameters of day t. This is the
# construction used by Campbell, Pflueger and Viceira (2020) and David and Veronesi (2013), who
# use the 5-year maturity. Same closure rule as above: returns span bond-market holidays, gaps
# longer than 7 calendar days are not bridged.
gsw = pd.read_csv(os.path.join(RAW, 'feds200628.csv'), skiprows=9, parse_dates=['Date'],
                  index_col='Date', na_values=['NA'])
gsw = gsw[gsw['BETA0'].notna()]


def svensson(par, m):
    b0, b1, b2, b3, t1, t2 = (par[c] for c in ['BETA0', 'BETA1', 'BETA2', 'BETA3', 'TAU1', 'TAU2'])
    x1 = m / t1
    f1 = (1 - np.exp(-x1)) / x1
    y = b0 + b1 * f1 + b2 * (f1 - np.exp(-x1))
    nss = (b3 != 0) & (t2 > 0)  # Nelson-Siegel days have BETA3 = 0 and TAU2 = -999.99
    x2 = m / t2.where(nss, 1.0)
    f2 = (1 - np.exp(-x2)) / x2
    return y + (b3 * (f2 - np.exp(-x2))).where(nss, 0.0)


def gsw_zero_return(n, max_gap_days=7):
    col = 'SVENY%02d' % n
    g = gsw[gsw.index.isin(cal)]  # GSW business days that are also S&P trading days
    g = g[g[col].notna()]
    dt = pd.Series(g.index, index=g.index).diff().dt.days / 365.25
    y_prev = g[col].shift(1) / 100
    y_now = svensson(g, (n - dt).fillna(n)) / 100
    r = n * y_prev - (n - dt) * y_now
    r[(dt * 365.25 > max_gap_days) | dt.isna()] = np.nan
    return r.reindex(cal)


for n in [2, 5, 10, 30]:
    bond_ret['ZCB%dY' % n] = gsw_zero_return(n)
    levels['GSW_Y%02d' % n] = gsw['SVENY%02d' % n].reindex(cal)

# ---- log returns on the trading calendar --------------------------------------------------
rets = pd.DataFrame(index=cal)
for col in ['SP500', 'GOLD_FUT', 'GOLD_FUT_REV', 'BTC']:  # + 'IEF_TR', 'TLT_TR' if rebuilt
    p = levels[col]
    rets[col] = np.log(p / p.shift(1))  # NaN if either day is missing: no bridging gaps
for col, r in bond_ret.items():
    rets[col] = r
# Yield changes: not used in the paper (see RAW_FILES above).
# for k in ['DGS2', 'DGS10', 'DGS30']:
#     y = levels[k]
#     rets['d' + k] = (y - y.shift(1))  # yield change, percentage points
rets.index.name = 'date'

# ---- CPI ---------------------------------------------------------------------------------
cpi = pd.read_csv(os.path.join(RAW, 'CPIAUCSL.csv'), parse_dates=['date']).set_index('date')['CPIAUCSL']
# Inflation as in the rule pipeline: m/m log change, centred 3-month moving average (months m-1, m, m+1).
# Computed before the cut, so that December 2024 uses the published January 2025 CPI.
_mom_full = 1200 * np.log(cpi / cpi.shift(1))
cma3_full = _mom_full.rolling(3, center=True).mean()
cpi = cpi[:END]  # no imputation: the sample ends before the unpublished October 2025 CPI
cpi_df = pd.DataFrame({'CPIAUCSL': cpi})
cpi_df['infl_yoy_pct'] = 100 * (cpi / cpi.shift(12) - 1)
mom = 1200 * np.log(cpi / cpi.shift(1))  # annualised m/m log change, %
cpi_df['infl_mom_ann_pct'] = mom
cpi_df['infl_3m_trailing_ann_pct'] = mom.rolling(3).mean()  # trailing, not centred
cpi_df['infl_3m_centred_ann_pct'] = cma3_full.reindex(cpi_df.index)  # centred, as in the rules (uses month m+1)
cpi_df.index.name = 'month'

levels.round(6).to_csv(os.path.join(OUT, 'market_daily_levels.csv'))
rets.round(8).to_csv(os.path.join(OUT, 'market_daily_logreturns.csv'))
cpi_df.round(4).to_csv(os.path.join(OUT, 'us_cpi_monthly.csv'))
print(levels.shape, rets.shape, cpi_df.shape)
print(levels.apply(lambda s: s.first_valid_index()).to_string())
