"""LaTeX tables (booktabs) for the correlation-regime episodes and reversals.

Inputs : market_data/regime_episodes_summary.csv, market_data/regime_reversals.csv  (from regime_episodes.py)
Outputs: figures_tables_paper/tab_regimes.tex                 main text: hysteresis labels, episodes and reversals in one table
         figures_tables_repository/tab_regime_episodes.tex     full episode table, hysteresis and deadband labels
         figures_tables_repository/tab_regime_reversals.tex    full reversal table, hysteresis and deadband labels
Requires \\usepackage{booktabs} in the paper.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'market_data')
PAPER, REPO = os.path.join(HERE, 'figures_tables_paper'), os.path.join(HERE, 'figures_tables_repository')
for d in (PAPER, REPO):
    os.makedirs(d, exist_ok=True)
NAMES = {'SP500': 'S\\&P 500', 'GOLD_FUT': 'Gold', 'BTC': 'Bitcoin', 'ZCB10Y': 'Bond'}  # bond defined in the notes
ORDER = ['SP500-ZCB10Y', 'SP500-GOLD_FUT', 'SP500-BTC', 'GOLD_FUT-ZCB10Y']
VARIANTS = [('hysteresis', 'Panel A: hysteresis labels (enter at $\\pm$15, leave inside $\\pm$5)'),
            ('deadband', 'Panel B: deadband labels ($\\pm$10)')]
S = pd.read_csv(os.path.join(DATA, 'regime_episodes_summary.csv'))
R = pd.read_csv(os.path.join(DATA, 'regime_reversals.csv'))
DEF_S = ('the sign concordance score S = 100 $\\times$ (share of days on which both returns have the same sign $-$ '
         'share with opposite signs), computed on rolling windows (Win.) of 50 or 250 trading days')
BOND = 'Bond = 10-year zero-coupon Treasury (G\\"urkaynak--Sack--Wright). Sample ends 31 Dec 2024.'


def pair(p):
    a, b = p.split('-')
    return f'{NAMES[a]} -- {NAMES[b]}'


def span(sample):
    a, b = [pd.Timestamp(x) for x in sample.split(' to ')]
    return f'{a.year}--{b.year}'


def fmt(v):
    return '--' if pd.isna(v) else f'{int(round(v))}'


def note(text):
    return ['\\par\\smallskip', '\\parbox{\\linewidth}{\\footnotesize ' + text + '}', '\\end{table}']


def write(path, lines):
    open(path, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')


# ---------------------------------------------------------------- Episodes
L = ['\\begin{table}[t]', '\\centering', '\\scriptsize', '\\setlength{\\tabcolsep}{2.8pt}',
     '\\caption{Correlation-regime episodes}', '\\label{tab:regime_episodes}',
     '\\begin{tabular}{llcrrrrrrrrr}', '\\toprule',
     ' & & & \\multicolumn{3}{c}{Positive} & \\multicolumn{3}{c}{None} & \\multicolumn{3}{c}{Negative} \\\\',
     '\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-12}',
     'Pair & Window & Sample & Share & Ep. & Med. & Share & Ep. & Med. & Share & Ep. & Med. \\\\']
for v, title in VARIANTS:
    L += ['\\midrule', f'\\multicolumn{{12}}{{l}}{{\\textit{{{title}}}}} \\\\', '\\addlinespace[2pt]']
    for p in ORDER:
        for w in [50, 250]:
            g = S[(S.pair == p) & (S.window == w) & (S.variant == v)].set_index('regime')
            if g.empty:
                continue
            cells = [pair(p) if w == 50 else '', f'{w} d', span(g['sample'].iloc[0])]
            for reg in ['pos', 'no', 'neg']:
                cells += [f"{g.loc[reg, 'share_days']:.2f}", f"{int(g.loc[reg, 'episodes'])}", fmt(g.loc[reg, 'median_len'])]
            L.append(' & '.join(cells) + ' \\\\')
L += ['\\bottomrule', '\\end{tabular}']
L += note(f'Correlation regime of each pair from {DEF_S}. Share = share of labelled days in the regime, which is the '
          'baseline of a rule whose hypothesis is that regime. Ep.\\ = number of episodes (runs of consecutive days with '
          'the same label). Med.\\ = median episode length in trading days. Sample = first and last labelled year. '
          + BOND)
write(os.path.join(REPO, 'tab_regime_episodes.tex'), L)

# ---------------------------------------------------------------- Reversals
L = ['\\begin{table}[t]', '\\centering', '\\scriptsize',
     '\\caption{Sign reversals of the correlation regime}', '\\label{tab:regime_reversals}',
     '\\begin{tabular}{llrrrr}', '\\toprule',
     ' & & \\multicolumn{2}{c}{Full sample} & \\multicolumn{2}{c}{2022--2024} \\\\',
     '\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}',
     'Pair & Window & Neg $\\to$ Pos & Pos $\\to$ Neg & Neg $\\to$ Pos & Pos $\\to$ Neg \\\\']
for v, title in VARIANTS:
    L += ['\\midrule', f'\\multicolumn{{6}}{{l}}{{\\textit{{{title}}}}} \\\\', '\\addlinespace[2pt]']
    for p in ORDER:
        for w in [50, 250]:
            g = R[(R.pair == p) & (R.window == w) & (R.variant == v)]
            if g.empty:
                continue
            g = g.iloc[0]
            L.append(' & '.join([pair(p) if w == 50 else '', f'{w} d', str(g.neg_to_pos), str(g.pos_to_neg),
                                 str(g.neg_to_pos_since_2022), str(g.pos_to_neg_since_2022)]) + ' \\\\')
L += ['\\bottomrule', '\\end{tabular}']
L += note('Number of switches between a positive and a negative correlation episode (consecutive episodes of opposite '
          'sign, possibly separated by a no-correlation spell), over the full sample of each pair and in the test period '
          '2022--2024 (switches whose new episode starts in 2022 or later). Labels and windows as in '
          'Table~\\ref{tab:regime_episodes}. ' + BOND)
write(os.path.join(REPO, 'tab_regime_reversals.tex'), L)

# ---------------------------------------------------------------- Episodes and reversals, hysteresis labels
L = ['\\begin{table}[t]', '\\centering', '\\scriptsize', '\\setlength{\\tabcolsep}{2.8pt}',
     '\\caption{Correlation regimes: baselines, episodes and reversals}', '\\label{tab:regimes}',
     '\\begin{tabular}{llrrrrrrrrrrr}', '\\toprule',
     ' & & \\multicolumn{3}{c}{Share} & \\multicolumn{2}{c}{Episodes} & \\multicolumn{2}{c}{Med.\\ days}'
     ' & \\multicolumn{2}{c}{Reversals} & \\multicolumn{2}{c}{2022--24} \\\\',
     '\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}\\cmidrule(lr){12-13}',
     'Pair & Win. & Pos & None & Neg & Pos & Neg & Pos & Neg & ${-}{\\to}{+}$ & ${+}{\\to}{-}$ & ${-}{\\to}{+}$ & ${+}{\\to}{-}$ \\\\',
     '\\midrule']
for p in ORDER:
    for w in [50, 250]:
        g = S[(S.pair == p) & (S.window == w) & (S.variant == 'hysteresis')].set_index('regime')
        rv = R[(R.pair == p) & (R.window == w) & (R.variant == 'hysteresis')].iloc[0]
        cells = [pair(p) if w == 50 else '', f'{w}',
                 f"{g.loc['pos', 'share_days']:.2f}", f"{g.loc['no', 'share_days']:.2f}", f"{g.loc['neg', 'share_days']:.2f}",
                 f"{int(g.loc['pos', 'episodes'])}", f"{int(g.loc['neg', 'episodes'])}",
                 fmt(g.loc['pos', 'median_len']), fmt(g.loc['neg', 'median_len']),
                 str(rv.neg_to_pos), str(rv.pos_to_neg), str(rv.neg_to_pos_since_2022), str(rv.pos_to_neg_since_2022)]
        L.append(' & '.join(cells) + ' \\\\')
L += ['\\bottomrule', '\\end{tabular}']
samples = '; '.join(f"{pair(p)}: {span(S[(S.pair == p) & (S.window == 50)]['sample'].iloc[0])}" for p in ORDER)
L += note(f'Correlation regime of each pair from {DEF_S}, labelled with hysteresis: a regime starts when S reaches '
          '$\\pm$15 and ends when S falls inside $\\pm$5. Share = share of labelled days in each regime, which is '
          'the baseline of a rule whose hypothesis is that regime. Episodes = runs of consecutive days with the same '
          'label; Med.\\ days = median episode length in trading days. Reversals = switches between a negative and a positive episode, '
          'possibly through a no-correlation spell; 2022--24 counts switches whose new episode starts in the test '
          f'period. Samples (50-day windows): {samples}. Results with the deadband labels of the original rule '
          '($\\pm$10) are in the repository. ' + BOND)
write(os.path.join(PAPER, 'tab_regimes.tex'), L)
print('written figures_tables_paper/tab_regimes.tex, figures_tables_repository/tab_regime_episodes.tex, figures_tables_repository/tab_regime_reversals.tex')
