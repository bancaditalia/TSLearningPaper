"""Figure for the paper: two rules on real data, 2015-2024.

Output: figures_tables_paper/fig_rule_examples.pdf
  columns  S&P 500 - gold (contagion, T2) and S&P 500 - 10-year zero-coupon Treasury (safe haven, T1b)
  top      the two series: levels from cumulative log returns (1 Jan 2015 = 100)
  bottom   50-day sign concordance with the no-correlation band (+-10)
  shading  light: days on which the condition of the rule holds
             T2: the two series are tail-dependent (at least 2 joint crash days in the window)
             T1b: the S&P 500 is stressed (high volatility and downward trend)
           dark: the condition holds and the concordance is in the regime the rule predicts
             (T2: positive, above +10; T1b: negative, below -10)
Concordance (Section 3.1): s_t = 100/50 * sum 1[sgn d|r_X| = sgn d|r_Y|] sgn(r_X) sgn(r_Y) over the
50-day window ending at t: same-sign minus opposite-sign share among magnitude-concordant days. Thresholds
(crash = 5th percentile of each series, high volatility = at least 30% of the window above the 90th
percentile of |r_X|, trend = window sum below -0.02) are computed on the training period 2015-2021.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, 'figures_tables_paper')
os.makedirs(PAPER, exist_ok=True)
FONTS = os.path.join(HERE, 'fonts')
if os.path.isdir(FONTS):
    for fn in os.listdir(FONTS):
        if fn.lower().endswith(('.ttf', '.otf')):
            font_manager.fontManager.addfont(os.path.join(FONTS, fn))
installed = {f.name for f in font_manager.fontManager.ttflist}
family = next(f for f in ['Avenir Next', 'Avenir Next LT Pro', 'Nunito Sans Light', 'Inter', 'DejaVu Sans'] if f in installed)
plt.rcParams.update({'font.family': family, 'font.weight': 'light', 'font.size': 6.8, 'axes.titlesize': 7.3,
                     'axes.edgecolor': '#52514e', 'axes.linewidth': 0.5, 'xtick.color': '#52514e',
                     'ytick.color': '#52514e', 'xtick.major.size': 2, 'ytick.major.size': 2, 'pdf.fonttype': 42})
INK, MUTED, GRID = '#2b2a28', '#8a8984', '#e6e5e1'
BLUE, DARK = '#2a78d6', '#2b2a28'
COND, HOLDS, BAND = '#f5ddcb', '#e3936a', '#e9e8e4'

W, DEAD = 50, 10
START, END, SPLIT = '2015-01-01', '2024-12-31', pd.Timestamp('2022-01-01')
r = pd.read_csv(os.path.join(HERE, 'market_data', 'market_daily_logreturns.csv'), parse_dates=['date'],
                index_col='date').loc[START:END]


def pair(y):
    d = r[['SP500', y]].dropna()
    x, yy = d.SP500, d[y]
    tr = d.index < SPLIT
    mag = (np.sign(x.abs().diff()) == np.sign(yy.abs().diff())).astype(float)   # both |r| rise or both fall
    s = 100 * (mag * np.sign(x) * np.sign(yy)).rolling(W).mean()
    crash = (x < x[tr].quantile(0.05)) & (yy < yy[tr].quantile(0.05))
    tail = crash.astype(float).rolling(W).sum() >= 2
    high_vol = (x.abs() > x[tr].abs().quantile(0.90)).astype(float).rolling(W).mean() >= 0.30 - 1e-9
    stressed = high_vol & (x.rolling(W).sum() < -0.02)
    levels = 100 * np.exp(d.cumsum())
    return levels, s, tail, stressed


def spans(ax, mask, color, z):
    on = mask.fillna(False).astype(int).values
    d = np.diff(np.r_[0, on, 0])
    idx = mask.index
    for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        ax.axvspan(idx[a], idx[min(b, len(idx) - 1)], color=color, lw=0, zorder=z)


cols = [('GOLD_FUT', 'Gold', 'Contagion (T2): S&P 500 and gold', 'tail', 'pos',
         'both crash together', 'and regime positive'),
        ('ZCB10Y', '10y zero-coupon', 'Safe haven (T1b): S&P 500 and bond', 'stressed', 'neg',
         'S&P 500 stressed', 'and regime negative')]
fig, axes = plt.subplots(2, 2, figsize=(4.8, 3.3), sharex=True, gridspec_kw={'height_ratios': [1, 1.15]})
for j, (y, yname, title, cname, head, clab, hlab) in enumerate(cols):
    lv, s, tail, stressed = pair(y)
    cond = tail if cname == 'tail' else stressed
    holds = cond & ((s > DEAD) if head == 'pos' else (s < -DEAD))
    for ax in axes[:, j]:
        spans(ax, cond, COND, 0)
        spans(ax, holds, HOLDS, 1)
        ax.axvline(SPLIT, color=MUTED, lw=0.6, ls=(0, (4, 2)), zorder=4)
        ax.spines[['top', 'right']].set_visible(False)
        ax.margins(x=0)
    ax = axes[0, j]
    ax.plot(lv.index, lv.SP500, color=BLUE, lw=0.7, zorder=3)
    ax.plot(lv.index, lv[y], color=DARK, lw=0.7, zorder=3)
    ax.set_title(title, loc='left', pad=8)
    ax.text(SPLIT, 1.0, 'train | test', transform=ax.get_xaxis_transform(), ha='center', va='bottom',
            fontsize=5.8, color=MUTED)
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=0.8, label='S&P 500'),
                       Line2D([], [], color=DARK, lw=0.8, label=yname)],
              frameon=False, fontsize=5.8, loc='upper left', handlelength=1.4, borderaxespad=0.1)
    ax.grid(axis='y', color=GRID, lw=0.4)
    ax = axes[1, j]
    ax.axhspan(-DEAD, DEAD, color=BAND, lw=0, zorder=2)
    ax.plot(s.index, s.values, color=INK, lw=0.5, zorder=3)
    ax.set_ylim(-80, 80)
    ax.set_yticks([-50, 0, 50])
    ax.grid(axis='y', color=GRID, lw=0.4)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(handles=[Patch(color=COND, label=clab), Patch(color=HOLDS, label=hlab)],
              frameon=False, fontsize=5.8, loc='upper left', ncol=2, handlelength=1.1, borderaxespad=0,
              bbox_to_anchor=(0, -0.2), columnspacing=1.0)
axes[0, 0].set_ylabel('level (2015 = 100)')
axes[1, 0].set_ylabel('concordance')
fig.tight_layout(w_pad=0.8, h_pad=0.4)
out = os.path.join(PAPER, 'fig_rule_examples.pdf')
fig.savefig(out)
print('font:', family, '| written', out)
