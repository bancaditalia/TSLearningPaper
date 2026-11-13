"""Descriptive tables and figures for temporal-rule-mining

Inputs (market_data/): market_daily_logreturns.csv, us_inflation_states_daily.csv, us_cpi_monthly.csv,
                       regime_episodes_summary.csv, regime_episodes.py (concordance score)
Outputs
  figures_tables_paper/       tables and figures for the main text
    tab_comovement.{csv,tex}      Pearson and sign concordance by period and inflation state
    tab_scale.{csv,tex}           daily correlation label x window trend agreement (50 days)
    fig_stock_bond_50d.pdf        50-day stock-bond correlation of daily returns, regime labels, inflation, NBER
  figures_tables_repository/  material published with the code
    tab_series_stats.{csv,tex}    moments, autocorrelation and ADF unit-root test per series
    tab_volatility.{csv,tex}      annualised volatility by period and inflation state
    tab_inflation_by_decade.{csv,tex}
    fig_inflation_realtime.pdf, fig_stock_bond_250d.pdf, fig_pair_<A>_<B>.pdf (figures are PDF only)
  The regime tables (figures_tables_paper/tab_regimes.tex; figures_tables_repository/tab_regime_episodes.tex, tab_regime_reversals.tex)
  are written by regime_tables.py.
"""
import os
import re
import importlib.util
import contextlib
import io
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'market_data')
PAPER = os.path.join(HERE, 'figures_tables_paper')        # tables and figures used in the main text
REPO = os.path.join(HERE, 'figures_tables_repository')    # everything else, published with the code
for _d in (PAPER, REPO):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------- style 
FONTS = os.path.join(HERE, 'fonts')
if os.path.isdir(FONTS):
    for fn in os.listdir(FONTS):
        if fn.lower().endswith(('.ttf', '.otf')):
            font_manager.fontManager.addfont(os.path.join(FONTS, fn))
installed = {f.name for f in font_manager.fontManager.ttflist}
family = next(f for f in ['Avenir Next', 'Avenir Next LT Pro', 'Avenir', 'Nunito Sans Light', 'Inter', 'DejaVu Sans']
              if f in installed)
plt.rcParams.update({
    'font.family': family, 'font.weight': 'light', 'axes.labelweight': 'light',
    'font.size': 8.5, 'axes.titlesize': 9, 'axes.titleweight': 'normal',
    'axes.edgecolor': '#52514e', 'axes.linewidth': 0.6,
    'xtick.color': '#52514e', 'ytick.color': '#52514e', 'text.color': '#0b0b0b',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42,
})
INK, GRID, BLUE, ORANGE, SHADE, MUTED, SUB = '#0b0b0b', '#e6e5e1', '#2a78d6', '#eb6834', '#f1ede4', '#8a8984', '#52514e'
REG_COL = {'pos': BLUE, 'no': '#d9d8d3', 'neg': ORANGE}
NAMES = {'SP500': 'S&P 500', 'GOLD_FUT': 'Gold', 'BTC': 'Bitcoin', 'ZCB10Y': '10y Treasury'}
TNAMES_FULL = {'S&P 500': 'S&P 500', 'Gold': 'Gold', 'Bitcoin': 'Bitcoin', '10y Treasury': 'Bond'}
TNAMES = {'SP500': 'S&P 500', 'GOLD_FUT': 'Gold', 'BTC': 'Bitcoin', 'ZCB10Y': 'Bond'}  # table labels; bond defined in captions
SERIES = ['SP500', 'GOLD_FUT', 'BTC', 'ZCB10Y']
PAIRS = [('SP500', 'ZCB10Y'), ('SP500', 'GOLD_FUT'), ('SP500', 'BTC'), ('GOLD_FUT', 'ZCB10Y')]
PERIODS = [('Start–1999', '1962-01-01', '1999-12-31'), ('2000–2021', '2000-01-01', '2021-12-31'),
           ('2022–2024', '2022-01-01', '2024-12-31')]
SPLIT = pd.Timestamp('2022-01-01')  # train/test split
# NBER recessions (months), https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
NBER = [('1969-12', '1970-11'), ('1973-11', '1975-03'), ('1980-01', '1980-07'), ('1981-07', '1982-11'),
        ('1990-07', '1991-03'), ('2001-03', '2001-11'), ('2007-12', '2009-06'), ('2020-02', '2020-04')]

r = pd.read_csv(os.path.join(DATA, 'market_daily_logreturns.csv'), parse_dates=['date'], index_col='date')
st = pd.read_csv(os.path.join(DATA, 'us_inflation_states_daily.csv'), parse_dates=['date'], index_col='date')
cpi = pd.read_csv(os.path.join(DATA, 'us_cpi_monthly.csv'), parse_dates=['month'], index_col='month')
eps = pd.read_csv(os.path.join(DATA, 'regime_episodes_summary.csv'))
spec = importlib.util.spec_from_file_location('regime_episodes', os.path.join(DATA, 'regime_episodes.py'))
re_mod = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(re_mod)  # also refreshes regime_episodes_summary.csv / regime_reversals.csv
high = st['inflationHigh'].reindex(r.index)


TEX_ESCAPES = [('\\', '\\textbackslash{}'), ('&', '\\&'), ('%', '\\%'), ('_', '\\_'), ('#', '\\#'),
               ('<', '$<$'), ('>', '$>$'), ('²', '$^2$')]


def tex_escape(text):
    """Escape LaTeX special characters in header and body cells (captions are written as LaTeX already)."""
    text = str(text)
    for a, b in TEX_ESCAPES:
        text = text.replace(a, b)
    return text


def caption_escape(text):
    """Captions may contain LaTeX ($\\pm$, --); escape only bare % and & signs."""
    return re.sub(r'(?<!\\)([%&])', r'\\\1', text)


def to_tex(df, path, caption, label, index=True, panels=None, colsep=None, ncols_l=1, note=None):
    """Minimal booktabs table (no jinja2 needed). panels = [(title, df), ...] stacks sub-tables with
    a common header; ncols_l = number of left-aligned leading columns when index=False."""
    blocks = panels if panels is not None else [(None, df)]
    d = blocks[0][1]
    cols = [tex_escape(c) for c in ([d.index.name or ''] if index else []) + [str(c) for c in d.columns]]
    align = ('l' if index else 'l' * ncols_l) + 'r' * (d.shape[1] - (0 if index else ncols_l))
    lines = ['\\begin{table}[t]', '\\centering', '\\scriptsize']
    if colsep:
        lines.append(f'\\setlength{{\\tabcolsep}}{{{colsep}}}')
    lines += [f'\\caption{{{caption_escape(caption)}}}', f'\\label{{{label}}}',
              '\\begin{tabular}{' + align + '}', '\\toprule', ' & '.join(cols) + ' \\\\']
    for title, b in blocks:
        lines.append('\\midrule')
        if title:
            lines += [f'\\multicolumn{{{len(cols)}}}{{l}}{{\\textit{{{tex_escape(title)}}}}} \\\\', '\\addlinespace[2pt]']
        for i, row in b.iterrows():
            cells = ([str(i)] if index else []) + ['' if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row]
            lines.append(' & '.join(tex_escape(c) for c in cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    if note:
        lines += ['\\par\\smallskip', f'\\parbox{{\\linewidth}}{{\\footnotesize {caption_escape(note)}}}']
    lines += ['\\end{table}']
    open(path, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')


def save(df, name, caption, index=True, tex_df=None, dest=None, **kw):
    """CSV keeps the full layout; the LaTeX version may use a compact layout (tex_df / panels)."""
    df.to_csv(os.path.join(dest, name + '.csv'), index=index)
    to_tex(df if tex_df is None else tex_df, os.path.join(dest, name + '.tex'), caption, 'tab:' + name, index=index, **kw)


def shade(ax, s, color=SHADE):
    s = s.dropna().astype(int)
    runs = (s != s.shift()).cumsum()
    for _, g in s.groupby(runs):
        if g.iloc[0] == 1:
            ax.axvspan(g.index[0], g.index[-1], color=color, lw=0, zorder=0)


def style(ax):
    ax.grid(axis='y', color=GRID, lw=0.5, zorder=1)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=2.5)


def savefig(fig, name, dest):
    for ext in ['pdf']:  # PDF only
        fig.savefig(os.path.join(dest, f'{name}.{ext}'), dpi=300)
    plt.close(fig)


# ================================================================ Table 1: statistics
def fp(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def fs(x, d):
    """Fixed decimals with sign; values that round to zero are shown without a sign."""
    v = round(float(x), d)
    return f'{0:.{d}f}' if v == 0 else f'{v:+.{d}f}'


def fd(x, d):
    """Fixed decimals, minus sign only (no forced plus)."""
    v = round(float(x), d)
    return f'{0:.{d}f}' if v == 0 else f'{v:.{d}f}'


rows = []
for c in SERIES:
    x = r[c].dropna()
    lb_r = acorr_ljungbox(x, lags=[10])['lb_pvalue'].iloc[0]
    lb_r2 = acorr_ljungbox(x ** 2, lags=[10])['lb_pvalue'].iloc[0]
    jb = stats.jarque_bera(x).statistic
    adf_p = adfuller(x, maxlag=20, autolag=None)[1]
    rows.append({'Series': TNAMES[c], 'From': x.index[0].year, 'N': f'{len(x):,}',
                 'Mean (% p.a.)': fd(252 * 100 * x.mean(), 2), 'Vol (% p.a.)': fd(np.sqrt(252) * 100 * x.std(), 2),
                 'Skew': fd(stats.skew(x), 2), 'Ex. kurt.': fd(stats.kurtosis(x), 2),
                 'Min (%)': fd(100 * x.min(), 2), 'Max (%)': fd(100 * x.max(), 2),
                 'AC1 r': fd(x.autocorr(1), 3), 'AC1 r²': fd((x ** 2).autocorr(1), 3),
                 'LB(10) r p': fp(lb_r), 'LB(10) r² p': fp(lb_r2),
                 'JB': f'{jb:,.0f}', 'ADF p': fp(adf_p)})
t1 = pd.DataFrame(rows).set_index('Series')
t1_tex = t1.T
t1_tex.index.name = 'Statistic'
save(t1, 'tab_series_stats', 'Daily log returns: moments, autocorrelation, stationarity', dest=REPO,
     tex_df=t1_tex,
     note='Daily log returns over each series\' full available sample, ending 31 Dec 2024. Bond = 10-year zero-coupon '
          'Treasury (G\\"urkaynak--Sack--Wright). p.a. = per annum: the mean is annualised as 252 $\\times$ the daily mean '
          'and the volatility as $\\sqrt{252}$ $\\times$ the daily standard deviation, both in %. Min and Max are the '
          'smallest and largest daily returns (%). Ex. kurt. = excess kurtosis. AC1 = first-order autocorrelation of '
          'returns (r) and squared returns (r$^2$). LB(10) = p-value of the Ljung--Box test of no autocorrelation up to '
          '10 lags. JB = Jarque--Bera statistic for normality. ADF = p-value of the augmented Dickey--Fuller test with '
          'a constant and 20 lags; the null hypothesis is a unit root (non-stationarity).')

# ================================================================ Table 2: volatility by period and inflation state
rows = []
for c in SERIES:
    x = r[c]
    row = {'Series': TNAMES[c]}
    for lab, a, b in PERIODS:
        xs = x[a:b].dropna()
        row[lab] = fd(np.sqrt(252) * 100 * xs.std(), 1) if len(xs) > 250 else ''
    for lab, v in [('Infl. high', 1), ('Infl. low', 0)]:
        xs = x[high == v].dropna()
        row[lab] = fd(np.sqrt(252) * 100 * xs.std(), 1)
    rows.append(row)
t2 = pd.DataFrame(rows).set_index('Series')
save(t2, 'tab_volatility', 'Volatility by period and inflation state', dest=REPO,
     note='Annualised volatility of daily log returns ($\\sqrt{252}$ $\\times$ the daily standard deviation), in %. '
          'Start--1999 runs from the first observation of each series (S&P 500: 1962; Bond: 1971) to 1999; empty cells: '
          'series not yet available. Infl. high/low: days in and out of the main inflation state (12-month CPI '
          'inflation, entering at 3.0% and leaving below 2.5%; CPI for month m used from month m+2). Bond = 10-year '
          'zero-coupon Treasury (G\\"urkaynak--Sack--Wright). Sample ends 31 Dec 2024.')


# ================================================================ Table 3: correlations
def concord(x, y):
    sx, sy = np.sign(x), np.sign(y)
    v = (sx != 0) & (sy != 0)
    return 100 * ((sx == sy) & v).sum() / v.sum() - 100 * ((sx == -sy) & v).sum() / v.sum()


rows = []
for a, b in PAIRS:
    d = r[[a, b]].dropna()
    row = {'Pair': f'{TNAMES[a]} – {TNAMES[b]}'}
    subsets = [('Full', d)] + [(lab, d[s:e]) for lab, s, e in PERIODS] + \
              [('Infl. high', d[high.reindex(d.index) == 1]), ('Infl. low', d[high.reindex(d.index) == 0])]
    for lab, dd in subsets:
        row[lab] = (f'{fs(dd[a].corr(dd[b]), 2)} / {fs(concord(dd[a], dd[b]), 0)}') if len(dd) > 250 else ''
    rows.append(row)
t3 = pd.DataFrame(rows).set_index('Pair')
t3_p = t3.apply(lambda c: c.map(lambda v: v.split(' / ')[0] if v else ''))
t3_s = t3.apply(lambda c: c.map(lambda v: v.split(' / ')[1] if v else ''))
save(t3, 'tab_comovement', 'Daily co-movement by period and inflation state', dest=PAPER,
     panels=[('Panel A: Pearson correlation', t3_p), ('Panel B: sign concordance S', t3_s)], colsep='4pt',
     note='Panel A: Pearson correlation of daily log returns. Panel B: sign concordance score S = 100 $\\times$ '
          '(share of days on which both returns have the same sign $-$ share with opposite signs), zero returns '
          'excluded; for Gaussian returns S = 200 arcsin($\\rho$)/$\\pi$. Start--1999 runs from the first observation '
          'of each pair (S&P 500 -- Bond: 1971). Infl. high/low: days in and out of the high-inflation state (12-month '
          'CPI inflation, entering at 3.0% and leaving below 2.5%; CPI for month m used from month m+2). Empty cells: '
          'fewer than 250 joint observations. Bond = 10-year zero-coupon Treasury (G\\"urkaynak--Sack--Wright). '
          'Sample ends 31 Dec 2024.')

# Table 4 removed: the 50-day label baselines are reported in tab7_regime_episodes (regime_tables.py)

# ================================================================ Table 5: scale contingency
rows = []
for a, b in PAIRS[:4]:
    s = re_mod.score(r[a], r[b], 50)
    lab = re_mod.label_deadband(s)
    ta, tb = r[a].rolling(50, min_periods=40).sum(), r[b].rolling(50, min_periods=40).sum()
    da = np.where(ta > 0.02, 1, np.where(ta < -0.02, -1, 0))
    db = np.where(tb > 0.02, 1, np.where(tb < -0.02, -1, 0))
    tr = pd.Series(np.where((da != 0) & (da == db), 'same trend', np.where((da != 0) & (da == -db), 'opposite trend',
                                                                            'no common trend')), index=r.index)
    ok = lab.notna() & ta.notna() & tb.notna()
    ct = pd.crosstab(lab[ok], tr[ok], normalize='index').reindex(['pos', 'no', 'neg'])
    ct = ct[['same trend', 'opposite trend', 'no common trend']]
    for reg in ct.index:
        rows.append({'Pair': f'{TNAMES[a]} – {TNAMES[b]}' if reg == 'pos' else '', 'Daily label': reg,
                     **{k: f'{100 * v:.0f}%' for k, v in ct.loc[reg].items()}})
t5 = pd.DataFrame(rows)
t5_tex = t5.rename(columns={'same trend': 'Same trend', 'opposite trend': 'Opposite trend', 'no common trend': 'No common trend'})
save(t5, 'tab_scale', 'Daily co-movement versus window trends', dest=PAPER,
     index=False, tex_df=t5_tex, ncols_l=2,
     note='Rows: correlation regime of the pair on a 50-day window (deadband labels: positive if S $>$ 10, negative '
          'if S $<$ $-$10, none otherwise). Columns: share of those windows in which the two series trend in the same '
          'direction, in opposite directions, or have no common trend. A series trends up (down) when the sum of its '
          'daily log returns over the window exceeds 0.02 (falls below $-$0.02). Rows sum to 100%.')

# ================================================================ Table 6: inflation by decade
s = st.copy()
s['decade'] = (s.index.year // 10 * 10).astype(str) + 's'
g = s.groupby('decade')
t6 = pd.DataFrame({'Mean infl. yoy (%)': g.infl_yoy.mean().map(lambda v: fd(v, 1)),
                   'Max infl. yoy (%)': g.infl_yoy.max().map(lambda v: fd(v, 1)),
                   'Share high (main)': g.inflationHigh.mean().map(lambda v: fd(v, 2)),
                   'Share high (p75 rule)': g.inflationHigh_p75.mean().map(lambda v: fd(v, 2)),
                   'Share rising': g.inflationRising.mean().map(lambda v: fd(v, 2)), 'Onsets': g.inflationOnset.sum()})
t6.index.name = 'Decade'
save(t6, 'tab_inflation_by_decade', 'Inflation states by decade', dest=REPO,
     tex_df=t6.rename(columns={'Mean infl. yoy (%)': 'Mean (%)', 'Max infl. yoy (%)': 'Max (%)', 'Share high (main)': 'High (main)',
                               'Share high (p75 rule)': 'High (p75)', 'Share rising': 'Rising'}),
     note='US CPI inflation as known in real time (CPI for month m used from month m+2). Mean and Max: 12-month CPI '
          'inflation (%). High (main): share of days in the main state (entering at 3.0% and leaving below 2.5%). '
          'High (p75): share of days above the original threshold of the paper (2.33%, the 75th percentile of '
          '2015--2021). Rising: share of days with 3-month inflation above 12-month inflation. Onsets: entries into '
          'the main state. The 2020s cover 2020--2024.')

# ================================================================ Figure: inflation, reference month vs real time
fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.4), gridspec_kw={'height_ratios': [1.2, 1.0, 0.75], 'hspace': 0.5})
ax = axes[0]
shade(ax, st['inflationHigh'])
ax.plot(cpi.index, cpi.infl_yoy_pct, color=MUTED, lw=0.6, label='12-month, by reference month')
ax.step(st.index, st.infl_yoy, where='post', color=ORANGE, lw=1.2, label='12-month, as known on each day')
ax.axhline(3.0, color=INK, lw=0.6, ls=(0, (4, 2)), label='enter 3.0%')
ax.axhline(2.5, color=INK, lw=0.6, ls=(0, (1, 2)), label='exit 2.5%')
ax.set_xlim(pd.Timestamp('1962-01-01'), pd.Timestamp('2024-12-31')); ax.set_ylim(-3, 16)
ax.set_title('a  US CPI inflation (%), 1962–2024; shaded = high-inflation state', loc='left', pad=3)
ax.legend(frameon=False, ncol=2, fontsize=7.2, loc='upper right', handlelength=2.2)
ax.xaxis.set_major_locator(mdates.YearLocator(10)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
style(ax)

ax = axes[1]
z0, z1 = pd.Timestamp('2020-01-01'), pd.Timestamp('2024-12-31')
shade(ax, st['inflationHigh'][z0:z1])
c = cpi[z0 - pd.DateOffset(months=1):]
ax.plot(c.index, c.infl_yoy_pct, color=MUTED, lw=0.9, marker='o', ms=1.6, label='12-month, by reference month')
ax.step(st[z0:].index, st[z0:].infl_yoy, where='post', color=ORANGE, lw=1.3, label='12-month, as known')
ax.step(st[z0:].index, st[z0:].infl_3m, where='post', color=BLUE, lw=0.8, label='3-month trailing ann., as known')
ax.axhline(3.0, color=INK, lw=0.6, ls=(0, (4, 2))); ax.axhline(2.5, color=INK, lw=0.6, ls=(0, (1, 2)))
ax.set_xlim(z0, z1); ax.set_ylim(-4, 11)
ax.set_title('b  Zoom 2020–2024: the real-time series lags the reference month by two months', loc='left', pad=3)
ax.legend(frameon=False, ncol=2, fontsize=7.2, loc='lower right', handlelength=2.2)
ax.xaxis.set_major_locator(mdates.YearLocator(1)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
style(ax)

ax = axes[2]
variants = [('inflationHigh', 'Main (3.0 in, 2.5 out)'), ('inflationHigh_yoy25', 'yoy 2.5+'),
            ('inflationHigh_yoy30', 'yoy 3.0+'), ('inflationHigh_yoy40', 'yoy 4.0+'),
            ('inflationHigh_3m30', '3-month 3.0+'), ('inflationHigh_p75', 'p75 rule (2.33)'),
            ('inflationRising', 'Rising (3m > yoy)')]
for i, (col, lab) in enumerate(variants):
    v = st[col].dropna().astype(int)
    runs = (v != v.shift()).cumsum()
    for _, gg in v.groupby(runs):
        if gg.iloc[0] == 1:
            ax.axvspan(gg.index[0], gg.index[-1], ymin=1 - (i + 0.9) / len(variants), ymax=1 - (i + 0.1) / len(variants),
                       color=ORANGE if i == 0 else (BLUE if col == 'inflationRising' else '#c9a48f'), lw=0)
ax.set_yticks([1 - (i + 0.5) / len(variants) for i in range(len(variants))])
ax.set_yticklabels([lab for _, lab in variants], fontsize=7); ax.set_ylim(0, 1)
ax.set_xlim(pd.Timestamp('1962-01-01'), pd.Timestamp('2024-12-31'))
ax.xaxis.set_major_locator(mdates.YearLocator(10)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.spines[['top', 'right', 'left']].set_visible(False); ax.tick_params(axis='y', length=0)
ax.set_title('c  Inflation-state variants (days in state)', loc='left', pad=3)
fig.text(0.02, 0.04, 'Real-time rule: CPI for month m is used from the first trading day of month m+2.\n3-month trailing = '
         'average of the last three annualised m/m changes (no future months). Source: FRED CPIAUCSL, retrieved 17 Oct 2026.',
         fontsize=6.8, color=SUB, va='top')
fig.subplots_adjust(left=0.2, right=0.98, top=0.96, bottom=0.09)
savefig(fig, 'fig_inflation_realtime', REPO)

# ================================================================ Figure: stock-bond 250-day correlation
d = r[['SP500', 'ZCB10Y']].dropna()
rho = d.SP500.rolling(250, min_periods=200).corr(d.ZCB10Y)
fig, ax = plt.subplots(figsize=(7.2, 3.2))
shade(ax, st['inflationHigh'])
for a, b in NBER:
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b) + pd.offsets.MonthEnd(0), ymin=0.94, ymax=1.0, color=MUTED, lw=0)
ax.axhline(0, color=MUTED, lw=0.6)
ax.plot(rho.index, rho, color=BLUE, lw=1.0)
ax.set_xlim(pd.Timestamp(f'{rho.first_valid_index().year}-01-01'), pd.Timestamp('2024-12-31')); ax.set_ylim(-0.8, 0.8)
ax.xaxis.set_major_locator(mdates.YearLocator(10)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_title('S&P 500 – 10-year Treasury: 250-day rolling correlation of daily returns', loc='left', pad=3)
style(ax)
ax.legend(handles=[Patch(facecolor=SHADE, edgecolor=MUTED, lw=0.5, label='High-inflation state (real time)'),
                   Patch(color=MUTED, label='NBER recession')],
          frameon=True, facecolor='white', edgecolor='none', framealpha=1, fontsize=7.2, loc='lower left', ncol=2)
fig.text(0.08, 0.075, 'Bond = 10-year zero-coupon Treasury (from Aug 1971) from the Gürkaynak–Sack–Wright (2007) curve. Recessions: NBER business-cycle dates.\n'
         'Inflation state: 12-month CPI, enter 3.0%, exit 2.5%, CPI used from month m+2.', fontsize=6.8, color=SUB, va='top')
fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.22)
savefig(fig, 'fig_stock_bond_250d', REPO)

# ================================================================ Figure: stock-bond 50-day correlation of daily returns
d = r[['SP500', 'ZCB10Y']].dropna()
rho50 = r['SP500'].rolling(50, min_periods=40).corr(r['ZCB10Y'])
rho50 = rho50[d.index[0]:]
lab50 = re_mod.label_hysteresis(re_mod.score(r['SP500'], r['ZCB10Y'], 50))[d.index[0]:]
x0, x1 = pd.Timestamp(f'{d.index[0].year}-01-01'), pd.Timestamp('2024-12-31')
fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True, gridspec_kw={'height_ratios': [1, 0.11], 'hspace': 0.32})
ax = axes[0]
shade(ax, st['inflationHigh'])
for a, b in NBER:
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b) + pd.offsets.MonthEnd(0), ymin=0.95, ymax=1.0, color=MUTED, lw=0)
ax.axhline(0, color=MUTED, lw=0.6)
ax.plot(rho50.index, rho50, color=BLUE, lw=0.5)
ax.set_ylim(-1, 1)
ax.set_title('S&P 500 – 10-year Treasury: 50-day rolling correlation of daily returns', loc='left', pad=3)
style(ax)
ax.legend(handles=[Patch(facecolor=SHADE, edgecolor=MUTED, lw=0.5, label='High-inflation state (real time)'),
                   Patch(color=MUTED, label='NBER recession')],
          frameon=True, facecolor='white', edgecolor='none', framealpha=1, fontsize=7.2, loc='lower left', ncol=2)
ax = axes[1]
lb = lab50.dropna()
runs = (lb != lb.shift()).cumsum()
for _, gg in lb.groupby(runs):
    ax.axvspan(gg.index[0], gg.index[-1], color=REG_COL[gg.iloc[0]], lw=0)
ax.set_yticks([]); ax.spines[['top', 'right', 'left']].set_visible(False)
ax.set_title('Correlation regime (hysteresis labels)', loc='left', pad=3, fontsize=8)
ax.legend(handles=[Patch(color=REG_COL[k], label=l) for k, l in [('pos', 'positive'), ('no', 'none'), ('neg', 'negative')]],
          frameon=False, fontsize=7, ncol=3, loc='lower right', bbox_to_anchor=(1.0, 0.95), handlelength=1.2)
ax.set_xlim(x0, x1)
ax.xaxis.set_major_locator(mdates.YearLocator(10)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.text(0.08, 0.105, 'Bond = 10-year zero-coupon Treasury (Gürkaynak–Sack–Wright curve, from Aug 1971). Recessions: NBER business-cycle dates.\n'
         'Regime labels from the sign concordance score on the same 50-day windows, entering at ±15 and leaving inside ±5.\n'
         'Inflation state: 12-month CPI, enter 3.0%, exit 2.5%, CPI used from month m+2.', fontsize=6.8, color=SUB, va='top')
fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.25)
savefig(fig, 'fig_stock_bond_50d', PAPER)

# ================================================================ Figures per pair
for a, b in PAIRS[:4]:
    d = r[[a, b]].dropna(how='all')
    start = max(r[a].first_valid_index(), r[b].first_valid_index())
    d = d[start:]
    lv = np.exp(d.fillna(0).cumsum()) * 100
    j = r[[a, b]][start:]
    rho = j[a].rolling(50, min_periods=40).corr(j[b])
    sc = re_mod.score(r[a], r[b], 50)[start:]
    lab = re_mod.label_hysteresis(sc)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.0), sharex=True,
                             gridspec_kw={'height_ratios': [1.3, 1.0, 0.18], 'hspace': 0.35})
    for ax in axes[:2]:
        shade(ax, st['inflationHigh'][start:])
        ax.axvline(SPLIT, color=INK, lw=0.6, ls=(0, (4, 2)))
        style(ax)
    ax = axes[0]
    ax.plot(lv.index, lv[a], color=BLUE, lw=0.8, label=NAMES[a])
    ax.plot(lv.index, lv[b], color=ORANGE, lw=0.8, label=NAMES[b])
    ax.set_yscale('log')
    ax.set_title(f'{NAMES[a]} and {NAMES[b]}: levels from cumulative log returns (start = 100, log scale)', loc='left', pad=3)
    ax.legend(frameon=False, fontsize=7.2, loc='upper left')
    ax = axes[1]
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.plot(rho.index, rho, color=INK, lw=0.6, label='Pearson, 50 days')
    ax.plot(sc.index, sc / 100, color=BLUE, lw=0.6, alpha=0.8, label='Sign concordance S/100, 50 days')
    ax.set_ylim(-1, 1)
    ax.set_title('Daily co-movement on 50-day windows', loc='left', pad=3)
    ax.legend(frameon=False, fontsize=7.2, loc='lower left', ncol=2)
    ax = axes[2]
    lb = lab.dropna()
    runs = (lb != lb.shift()).cumsum()
    for _, gg in lb.groupby(runs):
        ax.axvspan(gg.index[0], gg.index[-1], color=REG_COL[gg.iloc[0]], lw=0)
    ax.set_yticks([]); ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.set_title('Correlation regime (hysteresis labels)', loc='left', pad=3, fontsize=8)
    ax.legend(handles=[Patch(color=REG_COL[k], label=l) for k, l in [('pos', 'positive'), ('no', 'none'), ('neg', 'negative')]],
              frameon=False, fontsize=7, ncol=3, loc='lower right', bbox_to_anchor=(1.0, 0.95), handlelength=1.2)
    span = (lv.index[-1] - lv.index[0]).days / 365.25
    loc = mdates.YearLocator(10 if span > 40 else (5 if span > 15 else 2))
    ax.xaxis.set_major_locator(loc); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(lv.index[0], lv.index[-1])
    fig.text(0.08, 0.035, 'Shaded: high-inflation state (real time). Dashed line: paper\'s train/test split (1 Jan 2022). '
             'Labels: sign concordance, enter at ±15, leave inside ±5.', fontsize=6.8, color=SUB, va='top')
    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.1)
    savefig(fig, f'fig_pair_{a}_{b}', REPO)

print('font:', family)
for t in [t1, t2, t3, t5, t6]:
    print(t.to_string(), '\n')
