"""Plots of simulated time series (one replication per figure), for the repository.

Outputs (figures_tables_repository/):
  fig_sim_vs_real.pdf        real S&P 500 and gold returns (last 10 years of the sample, to 2024) next to one
                             simulated null sample of the same length (no phenomenon planted, rho0 = 0):
                             returns and 50-day rolling correlation
  fig_sim_series_<T>.pdf     one simulated 10-year sample per template with the phenomenon planted
                             (rho0 = 0.3, strength theta as in THETA below): returns of X (S&P-like) and
                             Y (gold-like), and the 50-day rolling correlation against the planted one,
                             with the days in the trigger state shaded; train/test split at 70%

Fixed seeds: the figures are reproducible. 
"""
import os
import sys
import zlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.join(ROOT, 'figures_tables_repository')
os.makedirs(REPO, exist_ok=True)
sys.path.insert(0, HERE)
import dgp  # noqa: E402

FONTS = os.path.join(ROOT, 'fonts')
if os.path.isdir(FONTS):
    for fn in os.listdir(FONTS):
        if fn.lower().endswith(('.ttf', '.otf')):
            font_manager.fontManager.addfont(os.path.join(FONTS, fn))
installed = {f.name for f in font_manager.fontManager.ttflist}
family = next(f for f in ['Avenir Next', 'Avenir Next LT Pro', 'Nunito Sans Light', 'Inter', 'DejaVu Sans'] if f in installed)
plt.rcParams.update({'font.family': family, 'font.weight': 'light', 'font.size': 8, 'axes.titlesize': 8.5,
                     'axes.edgecolor': '#52514e', 'axes.linewidth': 0.6, 'xtick.color': '#52514e',
                     'ytick.color': '#52514e', 'pdf.fonttype': 42})
INK, MUTED, GRID = '#2b2a28', '#8a8984', '#e6e5e1'
BLUE, ORANGE, SHADE = '#2a78d6', '#eb6834', '#fbe3d6'

N = 2520                 # 10 years of daily returns
TRAIN = 0.7
RHO0 = 0.3
THETA = {'T1b': 0.45, 'T2': 0.4, 'T3b': 1.0, 'T4': 0.2, 'T7': 0.3}
TITLE = {'T1b': 'T1b safe haven', 'T2': 'T2 contagion', 'T3b': 'T3b decoupling', 'T4': 'T4 lead in stress',
         'T7': 'T7 macro-conditioned (inflation)'}
PLANT = {'T1b': 'correlation {r0} on normal days, {r1} on the days when X falls into its lower 10% tail',
         'T2': 'correlation {r0} on normal days, {r1} when X is in high volatility',
         'T3b': 'correlation {r0} on normal days, {r1} when both series are calm',
         'T4': 'correlation {r0}; when X is in high volatility, Y also loads {th} on X five days earlier',
         'T7': 'correlation {r0} on normal days, {r1} when inflation is high (known with a two-month lag)'}
STATE = {'T1b': None, 'T2': 'highVol', 'T3b': 'bothCalm', 'T4': 'highVol', 'T7': 'infl'}
STATE_LAB = {'highVol': 'X in high volatility', 'bothCalm': 'both series calm', 'infl': 'high inflation'}


def seed_of(tag):
    return zlib.crc32(tag.encode()) % (2 ** 31)


def shade(ax, years, on):
    """Shade the spans where the boolean array `on` is true."""
    on = np.asarray(on, bool)
    if not on.any():
        return
    d = np.diff(np.r_[0, on.astype(int), 0])
    for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        ax.axvspan(years[a], years[min(b, len(years) - 1)], color=SHADE, lw=0, zorder=0)


def style(ax, ylabel):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color=GRID, lw=0.5)
    ax.set_ylabel(ylabel)
    ax.margins(x=0)


def split_line(axes, x, top_ax):
    for ax in axes:
        ax.axvline(x, color=MUTED, lw=0.7, ls=(0, (4, 2)), zorder=3)
    top_ax.text(x, 1.02, 'train | test', transform=top_ax.get_xaxis_transform(), ha='center', va='bottom',
                fontsize=6.5, color=MUTED)


YLIM = {'X': 15.0, 'Y': 12.0}  # display range (%), a little above the largest real daily move 2000-2024 (12.8, 9.8)


def returns_panel(ax, xs, v, lim):
    """Plot daily returns on a fixed range; days beyond it are marked at the edge with their value."""
    ax.plot(xs, np.clip(v, -lim, lim), color=INK, lw=0.35)
    ax.set_ylim(-lim * 1.08, lim * 1.08)
    off = np.where(np.abs(v) > lim)[0]
    for i in off:
        sgn = np.sign(v[i])
        ax.plot(xs[i], sgn * lim, marker='^' if sgn > 0 else 'v', ms=3, color=ORANGE, zorder=4)
    if len(off):
        big = off[np.argmax(np.abs(v[off]))]
        ax.text(0.995, 0.97, f'{len(off)} day{"s" if len(off) > 1 else ""} beyond \u00b1{lim:.0f}% (largest {v[big]:+.0f}%)',
                transform=ax.transAxes, ha='right', va='top', fontsize=6.5, color=ORANGE)


def footnote(fig, text):
    fig.text(0.01, 0.008, text, fontsize=6.5, color='#52514e', va='bottom', wrap=False)


# ---------------------------------------------------------------- simulated vs real (null)
r = pd.read_csv(os.path.join(ROOT, 'market_data', 'market_daily_logreturns.csv'), parse_dates=['date'], index_col='date')
real = r[['SP500', 'GOLD_FUT']].dropna().loc[:'2024-12-31'].iloc[-N:]
x0, y0, _ = dgp.simulate('T2', 0.0, 0.0, N, 1, seed=seed_of('sim_vs_real'))
sim = pd.DataFrame({'SP500': x0[0], 'GOLD_FUT': y0[0]})

fig, axes = plt.subplots(3, 2, figsize=(7.2, 5.0), sharey='row', sharex='col')
yr_sim = np.arange(N) / 252
for j, (df, xs, head) in enumerate([(real, real.index, f'Real: S&P 500 and gold, {real.index[0]:%b %Y} to {real.index[-1]:%b %Y}'),
                                    (sim, yr_sim, 'Simulated: nothing planted, correlation 0')]):
    returns_panel(axes[0, j], np.asarray(xs), 100 * df.SP500.values, YLIM['X'])
    returns_panel(axes[1, j], np.asarray(xs), 100 * df.GOLD_FUT.values, YLIM['Y'])
    rc = df.SP500.rolling(50).corr(df.GOLD_FUT)
    axes[2, j].plot(xs, rc.values, color=BLUE, lw=0.9)
    axes[2, j].axhline(0, color=MUTED, lw=0.5)
    axes[0, j].set_title(head, loc='left', pad=4)
    for i, lab in enumerate(['S&P 500 / X\ndaily return (%)', 'gold / Y\ndaily return (%)', '50-day rolling\ncorrelation']):
        style(axes[i, j], lab if j == 0 else '')
axes[2, 1].set_xlabel('years')
axes[2, 1].set_ylim(-1, 1)
footnote(fig, 'Left: daily log returns of the S&P 500 and COMEX gold futures, last 2,520 trading days to Dec 2024.\n'
              'Right: one replication of the simulation design (GJR-GARCH-t for X, GARCH-t for Y, calibrated on 2000-2024)\n'
              'with constant zero correlation, same length. Days beyond the plotted range are marked at the edge with their value.')
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(os.path.join(REPO, 'fig_sim_vs_real.pdf'))
plt.close(fig)
print('fig_sim_vs_real.pdf  real corr', round(real.corr().iloc[0, 1], 3), '| simulated corr', round(sim.corr().iloc[0, 1], 3))

# ---------------------------------------------------------------- one sample per template
years = np.arange(N) / 252
for tpl, th in THETA.items():
    rec = {}
    x, y, infl = dgp.simulate(tpl, th, RHO0, N, 1, seed=seed_of(f'series|{tpl}|{th}|{RHO0}'), train_frac=TRAIN, record=rec)
    x, y = 100 * x[0], 100 * y[0]
    xs, ys = pd.Series(x), pd.Series(y)
    state = None
    if STATE[tpl] == 'infl':
        state = infl[0] > 0.5
    elif STATE[tpl]:
        state = rec[STATE[tpl]][0] > 0.5

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 4.9), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1.25]})
    for ax in axes:
        if state is not None:
            shade(ax, years, state)
    returns_panel(axes[0], years, x, YLIM['X'])
    returns_panel(axes[1], years, y, YLIM['Y'])
    if tpl == 'T1b':  # mark the extreme-down days of X in both return panels
        ed = rec['extremeDown'][0] > 0.5
        for ax, v in [(axes[0], x), (axes[1], y)]:
            lim = ax.get_ylim()[1] / 1.08
            ax.scatter(years[ed], np.clip(v[ed], -lim, lim), s=2.5, color=ORANGE, lw=0, zorder=3)

    ax = axes[2]
    if tpl == 'T4':
        lagc = ys.rolling(50).corr(xs.shift(dgp.LEAD_K))
        ax.plot(years, lagc.values, color=BLUE, lw=0.9)
        ax.plot(years, pd.Series(rec['lead'][0]).rolling(50).mean().values, color=ORANGE, lw=1.0, ls=(0, (3, 1.5)))
        ylab, real_lab, plant_lab = '50-day rolling correlation\nof Y with X five days earlier', \
            'realised (rolling, Y on X five days earlier)', 'planted lead loading (50-day average)'
    else:
        ax.plot(years, xs.rolling(50).corr(ys).values, color=BLUE, lw=0.9)
        ax.plot(years, pd.Series(rec['rho'][0]).rolling(50).mean().values, color=ORANGE, lw=1.0, ls=(0, (3, 1.5)))
        ylab, real_lab, plant_lab = '50-day rolling\ncorrelation', 'realised (rolling)', 'planted (50-day average)'
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.set_ylim(-1, 1)
    style(axes[0], 'X (S&P-like)\ndaily return (%)')
    style(axes[1], 'Y (gold-like)\ndaily return (%)')
    style(ax, ylab)
    ax.set_xlabel('years')
    split_line(axes, TRAIN * N / 252, axes[0])
    axes[0].set_title(f'{TITLE[tpl]}: one simulated sample', loc='left', pad=4)

    handles = [Line2D([], [], color=BLUE, lw=0.9, label=real_lab),
               Line2D([], [], color=ORANGE, lw=1.0, ls=(0, (3, 1.5)), label=plant_lab)]
    if state is not None:
        handles.append(Patch(color=SHADE, label=STATE_LAB[STATE[tpl]]))
    if tpl == 'T1b':
        handles.append(Line2D([], [], color=ORANGE, marker='o', ms=2.5, lw=0, label='X in its lower 10% tail'))
    ax.legend(handles=handles, frameon=False, fontsize=6.5, loc='lower left', ncol=len(handles),
              bbox_to_anchor=(0, -0.62))

    r1 = RHO0 - th if tpl == 'T1b' else RHO0 + th if tpl in ('T2', 'T7') else RHO0 * (1 - th)
    planted = PLANT[tpl].format(r0=f'{RHO0:.1f}', r1=f'{r1:.2f}'.rstrip('0').rstrip('.') if r1 else '0', th=f'{th:.1f}')
    footnote(fig, f'One replication of the simulation design, 10 years (2,520 days), calibrated on the S&P 500 (X) and gold (Y).\n'
                  f'Planted phenomenon: {planted}.\nStates from the 50-day window ending the previous day. Dashed vertical line: 70/30 train/test split. '
                  f'Days beyond the plotted range are marked at the edge.')
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(REPO, f'fig_sim_series_{tpl}.pdf'))
    plt.close(fig)
    share = state.mean() if state is not None else rec['extremeDown'][0].mean()
    print(f'fig_sim_series_{tpl}.pdf  theta {th}  share of days in state {share:.3f}')
