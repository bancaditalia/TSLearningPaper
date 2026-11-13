"""Daily log returns (1962-2024) with US CPI inflation; high-inflation periods shaded.

Shading uses the main daily state `inflationHigh` from market_data/us_inflation_states_daily.csv:
real-time aligned (CPI for month m used from month m+2) and with hysteresis (enter at 3.0%,
exit below 2.5% 12-month inflation). The inflation panel shows inflation as known on each day,
so line and shading line up.
Uses Avenir Next (light) when installed, otherwise Nunito Sans Light from ./fonts, otherwise Inter Light."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'market_data')
FONTS = os.path.join(HERE, 'fonts')  # optional local font files (e.g. Nunito Sans Light, OFL licence)
if os.path.isdir(FONTS):
    for fn in os.listdir(FONTS):
        if fn.lower().endswith(('.ttf', '.otf')):
            font_manager.fontManager.addfont(os.path.join(FONTS, fn))
installed = {f.name for f in font_manager.fontManager.ttflist}
family = next((f for f in ['Avenir Next', 'Avenir Next LT Pro', 'Avenir', 'Nunito Sans Light', 'Inter', 'DejaVu Sans']
               if f in installed))
plt.rcParams.update({
    'font.family': family, 'font.weight': 'light', 'axes.labelweight': 'light',
    'font.size': 8.5, 'axes.titlesize': 9, 'axes.titleweight': 'normal',
    'axes.edgecolor': '#52514e', 'axes.linewidth': 0.6,
    'xtick.color': '#52514e', 'ytick.color': '#52514e', 'text.color': '#0b0b0b',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'pdf.fonttype': 42,
})
INK, GRID, BLUE, ORANGE, SHADE, MUTED = '#0b0b0b', '#e6e5e1', '#2a78d6', '#eb6834', '#f1ede4', '#8a8984'

r = pd.read_csv(os.path.join(DATA, 'market_daily_logreturns.csv'), parse_dates=['date'], index_col='date')
st = pd.read_csv(os.path.join(DATA, 'us_inflation_states_daily.csv'), parse_dates=['date'], index_col='date')

start, end = pd.Timestamp('1962-01-01'), pd.Timestamp('2024-12-31')
high = st['inflationHigh'].dropna().astype(int)
# contiguous high-inflation spells -> (start, end) intervals on the trading calendar
runs = (high != high.shift()).cumsum()
spells = [(g.index[0], g.index[-1]) for _, g in high.groupby(runs) if g.iloc[0] == 1]

panels = [('SP500', 'S&P 500'), ('GOLD_FUT', 'Gold (COMEX futures)'),
          ('BTC', 'Bitcoin'), ('ZCB10Y', '10-year zero-coupon Treasury (Gürkaynak–Sack–Wright curve)')]

fig, axes = plt.subplots(len(panels) + 1, 1, figsize=(7.2, 9.7), sharex=True,
                         gridspec_kw={'height_ratios': [1] * len(panels) + [1.25], 'hspace': 0.42})
for ax in axes:
    for a, b in spells:
        ax.axvspan(a, b, color=SHADE, lw=0, zorder=0)
    ax.grid(axis='y', color=GRID, lw=0.5, zorder=1)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=2.5)

for ax, (col, name) in zip(axes, panels):
    s = 100 * r[col].dropna()
    ax.plot(s.index, s.values, color=BLUE, lw=0.35, zorder=2)
    ax.axhline(0, color=MUTED, lw=0.5, zorder=1)
    lim = np.nanpercentile(np.abs(s), 99.9) * 1.15
    ax.set_ylim(-lim, lim)
    ax.set_title(f'{name}, daily log return (%)  ·  from {s.index[0].year}', loc='left', pad=3)

ax = axes[-1]
ax.step(st.index, st['infl_3m'], where='post', color=MUTED, lw=0.6, zorder=2, label='3-month trailing, annualised')
ax.step(st.index, st['infl_yoy'], where='post', color=ORANGE, lw=1.4, zorder=3, label='12-month')
ax.axhline(3.0, color=INK, lw=0.6, ls=(0, (4, 2)), zorder=2, label='enter 3.0%')
ax.axhline(2.5, color=INK, lw=0.6, ls=(0, (1, 2)), zorder=2, label='exit 2.5%')
ax.axhline(0, color=MUTED, lw=0.5, zorder=1)
ax.set_ylim(-6, 16)
ax.set_title('US CPI inflation as known on each day (%)', loc='left', pad=3)
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.66, 1.0), ncol=2, handlelength=2.2, fontsize=7.5)
ax.xaxis.set_major_locator(mdates.YearLocator(10))
ax.xaxis.set_minor_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_xlim(start, end)

fig.text(0.08, 0.06,
         f'Shaded: high-inflation state ({len(spells)} spells). Enters when 12-month CPI inflation reaches 3.0% and exits below 2.5%;\n'
         'CPI for month m is used from the first trading day of month m+2 (no figure before its release).\n'
         '3-month trailing inflation clipped to the axis range. Return axes clipped at 1.15× the 99.9th percentile of |r| (19 Oct 1987, −20.5%, is off-scale).\n'
         'Sources: Yahoo Finance (^GSPC, GC=F, BTC-USD), FRED (CPIAUCSL), Federal Reserve (GSW yield curve). Retrieved 20 Oct 2026.',
         fontsize=6.8, color='#52514e', va='top')
fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.105)
os.makedirs(os.path.join(HERE, 'figures_tables_repository'), exist_ok=True)  # not used in the main text
for ext in ['pdf']:  # PDF only
    fig.savefig(os.path.join(HERE, 'figures_tables_repository', f'returns_inflation.{ext}'), dpi=300)
print('font:', family, '| spells', len(spells))
