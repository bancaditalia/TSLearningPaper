"""Figure and table of the Monte Carlo of the econometric tests (results_econ.csv from mc_econ.py).

Outputs (figures_tables_repository/):
  fig_sim_power_econ.pdf      detection rate against theta, by template (rows) and sample length (columns),
                              one line per baseline correlation rho0; dashed line at 5%
  tab_sim_econ.tex            size (theta = 0) and power at the largest theta, by template, rho0 and length
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.join(ROOT, 'figures_tables_repository')
os.makedirs(REPO, exist_ok=True)
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
COL = {0.0: '#2a78d6', 0.3: '#eb6834', 0.6: '#52514e'}
NAMES = {'T1b': 'T1b safe haven', 'T2': 'T2 contagion', 'T3b': 'T3b decoupling', 'T4': 'T4 lead in stress',
         'T7': 'T7 macro (inflation)'}
XLAB = {'T1b': 'drop in correlation on extreme-down days', 'T2': 'rise in correlation in high volatility',
        'T3b': 'share of correlation removed in calm', 'T4': 'lead loading in high volatility',
        'T7': 'rise in correlation in high inflation'}
YEARS = {2520: '10 years', 5040: '20 years', 10080: '40 years'}

R = pd.read_csv(os.path.join(HERE, 'results_econ.csv'))
tpls = [t for t in NAMES if t in set(R.template)]
lengths = sorted(R.n.unique())
fig, axes = plt.subplots(len(tpls), len(lengths), figsize=(7.2, 1.7 * len(tpls) + 0.6), sharey=True, squeeze=False)
for i, t in enumerate(tpls):
    for j, n in enumerate(lengths):
        ax = axes[i, j]
        for rho0, g in R[(R.template == t) & (R.n == n)].groupby('rho0'):
            g = g.sort_values('theta')
            ax.plot(g.theta, g.detection_rate, marker='o', ms=2.5, lw=1.0, color=COL[rho0], label=rf'background correlation $\rho_0$ = {rho0:.1f}')
        ax.axhline(0.05, color='#8a8984', lw=0.6, ls=(0, (4, 2)))
        ax.set_ylim(-0.02, 1.02)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#e6e5e1', lw=0.5)
        if i == 0:
            ax.set_title(YEARS.get(n, f'n = {n}') + f' ({int(0.3 * n)} test days)', pad=3)
        if j == 0:
            ax.set_ylabel(NAMES[t] + '\ndetection rate')
        if i == len(tpls) - 1 or True:
            ax.set_xlabel(XLAB[t] if j == 1 else '', fontsize=7)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.035))
fig.text(0.01, 0.005, 'Detection rate of the econometric test of each phenomenon on the test part (30%) of simulated samples, '
         '200 replications per point.\nAt zero strength the rate is the size (dashed line: 5%).', fontsize=6.5, color='#52514e')
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(os.path.join(REPO, 'fig_sim_power_econ.pdf'))

# table: size and power at the largest theta
rows = []
for t in tpls:
    th_max = R[R.template == t].theta.max()
    for rho0 in sorted(R[R.template == t].rho0.unique()):
        row = [NAMES[t] if rho0 == sorted(R[R.template == t].rho0.unique())[0] else '', f'{rho0:.1f}']
        for kind, th in [('size', 0.0), ('power', th_max)]:
            for n in lengths:
                v = R[(R.template == t) & (R.rho0 == rho0) & (R.n == n) & (R.theta == th)].detection_rate
                row.append(f'{v.iloc[0]:.2f}' if len(v) else '--')
        rows.append(row)
hdr = ' & & \\multicolumn{%d}{c}{Size ($\\theta = 0$)} & \\multicolumn{%d}{c}{Power (largest $\\theta$)} \\\\' % (len(lengths), len(lengths))
L = ['\\begin{table}[t]', '\\centering', '\\scriptsize', '\\setlength{\\tabcolsep}{3pt}',
     '\\caption{Simulation: size and power of the econometric tests}', '\\label{tab:sim_econ}',
     '\\begin{tabular}{ll' + 'r' * 2 * len(lengths) + '}', '\\toprule', hdr,
     f'\\cmidrule(lr){{3-{2 + len(lengths)}}}\\cmidrule(lr){{{3 + len(lengths)}-{2 + 2 * len(lengths)}}}',
     'Template & $\\rho_0$ & ' + ' & '.join([YEARS.get(n, str(n)) for n in lengths] * 2) + ' \\\\', '\\midrule']
L += [' & '.join(r) + ' \\\\' for r in rows]
L += ['\\bottomrule', '\\end{tabular}', '\\par\\smallskip',
      '\\parbox{\\linewidth}{\\footnotesize Share of 200 simulated samples in which the econometric test of each phenomenon '
      'detects it at the 5\\% level on the test part (30\\% of the sample). Size: no phenomenon planted (target 0.05). '
      'Power: largest strength of the grid (T1b: correlation 0.6 lower on extreme-down days; T2: 0.4 higher in high '
      'volatility; T3b: correlation removed in calm; T4: lead loading 0.2; T7: 0.4 higher in high inflation). '
      '$\\rho_0$: background correlation. Samples of 10, 20 and 40 years of daily returns calibrated on the S\\&P 500 and '
      'gold (GJR-GARCH/GARCH with Student-t innovations). Replications with too few trigger days count as not detected.}',
      '\\end{table}']
open(os.path.join(REPO, 'tab_sim_econ.tex'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('font', family, '| configurations', len(R))
