# Runs — robustness checks without touching the paper's results

The paper's setup is the **base run**: `configs/config.yaml` alone, outputs in
`data/processed/` and `results/` as always.

Any other setting (another period, window length or stride) is a **named run**: a small
YAML file here that overrides only what changes. Every script accepts `--run <name>`.
Under a run, every output goes to its own folders:

```
data/processed/runs/<name>/     DDPM samples, transfer sets, labels, thresholds, classifier, ...
results/runs/<name>/            rules, verification, fidelity, tables
```

## Which run is which

- **Give a run a short name** with `scripts/runs.py new --name <name>` (e.g. `main`,
  `window25`). Without `--name`, the name is built from the parameters
  (`tr2005-2014_te2015-2021_w50_s1` = train 2005–2014, test 2015–2021, window 50, stride 1).
- **Every run folder has a `run_config.yaml`** with the run's COMPLETE configuration
  (`configs/config.yaml` merged with the run's YAML), written automatically and kept up to
  date. The run's own YAML here lists only what differs.
- **`python scripts/runs.py`** lists every run with its periods, window, stride and folders.
  **`python scripts/runs.py show <name>`** prints all its parameters.
- **Every run folder has a `run.json`** with the run's name and full parameters.
  If you edit a run's YAML after it has produced outputs, scripts **stop** instead of
  mixing two settings in one folder. Make a new run instead, or delete the folder.
  Two settings are exempt, because they are set during a run's life:
  `classifier.weights` and `paper.pairs`. They are recorded under `info` and kept up
  to date.
- **Under a run, every script prints one line on start**, e.g.
  `[run 'tr2005-2014_te2015-2021_w50_s1': train 2005-01-01..2014-12-31, test 2015-01-01..2021-12-31, window 50, real-pair stride 1]`.
- **Table captions take their years from the run** ("constants frozen on 2005--2014 and
  scored on 2015--2021"), so a table never shows the paper's years by mistake.

## Create a run

```bash
# another period (Bitcoin starts in 2017, so leave it out)
python scripts/runs.py new --train 2005-01-01:2014-12-31 --test 2015-01-01:2021-12-31 \
    --pairs sp500_gold,sp500_nasdaq

# another window length: set the window-dependent constants too (see below)
python scripts/runs.py new --window 75 --trend 0.0245 --max-w 38 --min-days 45

# another stride for the real-pair labels
python scripts/runs.py new --stride 5
```

`--pairs` also removes the fidelity entries of the pairs left out. A run's YAML can set
anything in `configs/config.yaml` except `paths:`. A key set to `null` removes it, e.g.
`paper: {fidelity: {sp500_bitcoin: null}}`.

## Run it: the full rerun, in order

Every command below takes `--run $R`. It reads and writes only the run's folders; a path
like `data/processed/...` in a command is moved to the run's folder automatically.
Without `--run`, the same commands rebuild the **base run**, e.g. after the DDPM split
change.

```bash
R=tr2005-2014_te2015-2021_w50_s1             # the run (python scripts/runs.py lists them)
P=sp500_gold,sp500_nasdaq                     # the run's paper.pairs

# 1. Data: windows of the run's length; DDPM training split at the run's test date (seconds)
python scripts/data_prep.py -d all --run $R

# 2. DDPM samples, one per series (minutes each; they depend on window and period)
for s in SP500 Gold NASDAQ; do python scripts/synthesize.py -d $s --run $R; done   # + Bitcoin if in P

# 3. Transfer sets, the recipe of the paper's classifier (labelled in Python, no engine)
python scripts/build_realistic_transfer.py -p $P --strength boundary --run $R

# 4. Classifier (same recipe and seed as classifier_v7p3_s42)
python scripts/distill.py --pairs $P --transfer-suffix _boundary --seed 42 \
    --save data/processed/classifier_v7p3_s42.pt --run $R
#    then add to configs/runs/$R.yaml:
#      classifier: {weights: data/processed/classifier_v7p3_s42.pt}
#    (that is the run's own file: data/processed/runs/$R/classifier_v7p3_s42.pt)

# 5. Exact labels on the real pairs + macro state (thresholds refitted on the run's train period)
python scripts/prep_real_pairs.py -p $P --run $R

# 6. Classifier labels on the real pairs
python scripts/infer.py -p $P --run $R

# 7. [ENGINE] reasoner labels on the real test windows (fidelity targets; cached, resumable)
caffeinate -dims python scripts/eval.py --metric fidelity --dataset real --pair sp500_gold \
    --step 5 --label-first --run $R
#    (bitcoin, if in P: ... --step 10 --label-first --run $R)

# 8. Rules, [ENGINE] verification, multiple testing, fidelity, classifier timing
caffeinate -dims python scripts/paper.py results --run $R    # --no-engine: skip verification

# 9. Every table, with the consistency checks -> results/runs/$R/
python scripts/paper.py tables --run $R
```

**Optional: the LLM arm** (README step 10) works the same way with `--run $R`; its
prompt uses the run's train period and definitions.

**Optional: the reasoner timing** (README step 14, each command with `--run $R`). Step 8
times only the classifier. Without the reasoner timing, the run's speedup table stays
empty. The speedup mostly matters for the paper's own window, so you can skip it for
robustness runs.

**The classifier.** `classifier.weights: data/processed/...` points into the **run's**
folder. A run therefore needs its own classifier: train it with `distill.py --run $R`, then
set `classifier: {weights: data/processed/<file>.pt}` in the run's YAML. Weights trained
on another window length are refused.

A run that changes only the stride can reuse the paper's classifier on purpose:
`classifier: {weights: "base:data/processed/classifier_v7p3_s42.pt"}`.

**What a period-only or stride-only run can skip.**
- **Stride only:** steps 2–5 (DDPM, transfer sets, classifier). Run step 1
  (`data_prep.py -d all --run $R`, seconds, from the cached raw files), then go on from
  step 6, with `classifier.weights` pointing at the base classifier as above.
- **Period only:** none. The thresholds tau/qLow, the DDPM split and the transfer sets
  all depend on the training period.

## Comparing runs (robustness table)

```bash
python scripts/compare_runs.py --runs <run A>,<run B>                 # Python labels
python scripts/compare_runs.py --runs <run A>,<run B> --source classifier
python scripts/compare_runs.py --runs <run A>,<run B> --all           # also rules scored in one run only
```

This writes one row per rule and pair scored on the test period in every run. Each row
has each run's lift [CI] and verdict, and whether the sign agrees. The caption is
built from the runs' own periods and thresholds. Output goes to
`results/compare/robustness[_classifier].{csv,md,tex}`, plus `compare.json` naming the
runs. It reads the runs' rule tables and recomputes nothing, so run it after
`paper.py tables` of each run.

## Changing the window length: the constants that depend on it

The paper's values were chosen for a 50-day window. With another length W, set these in
the run's YAML (`runs.py new` flags in brackets). The scaling below is a suggestion, not
a derivation: say in the paper which values you used.

| key | paper (W=50) | suggested for W | why |
|---|---|---|---|
| `labels.trend` (`--trend`) | 0.02 | 0.02·√(W/50) | a sum of W daily returns; its spread grows like √W |
| `labels.tail_min` (`--tail-min`) | 2 | `auto` | see "The tail-dependence threshold" below |
| `labels.high_vol_pct`, `low_vol_pct` | 30, 5 | unchanged | shares of days, already scale-free |
| `rules.max_w` (`--max-w`) | 25 | ≈ W/2 | the largest lag tried when fitting `<+>[0,w]` |
| `rules.min_condition_days` (`--min-days`) | 30 | ≈ 0.6·W | minimum condition days for a rule to be scored |

### The reasoner program for the run

The base program, `tslearning/symbolic/preLabeling.vada`, has the paper's values written
into it. For any run whose window or cut-offs differ, a copy is generated with the run's
values substituted:

- **The window box `/\50` → `/\W`.** Every per-window aggregate uses it: the counts behind
  same% / diff%, the share of big days, the sum of returns, the co-crash count. There are
  15 of these boxes.
- **The cut-offs:** `Sum >= 0.02` / `<= -0.02` (trend), `Pct >= 30.0` / `<= 5.0`
  (volatility) and `Max >= 2` (tail).

The deadband is not substituted: it already reaches the program as a parameter.

**Two copies are saved,** each starting with a comment that names the run and lists the
values:
- `data/processed/runs/<name>/vada/preLabeling.vada`: the one the engine reads;
- `results/runs/<name>/preLabeling.vada`: kept next to the results, tracked by git.

The rule-verification programs (`verify_rules.py`) are built on the same generated copy.

**A safety check:** every substitution must match exactly one line of the base program.
If someone edits `preLabeling.vada` so a pattern no longer matches, generation stops with
an error instead of applying only part of the changes.

### The tail-dependence threshold

`tailDependent(X,Y)` holds when the window has at least `tail_min` days on which both
series fall below their qLow, the 5th percentile of returns. The paper uses 2, chosen for
50-day windows.

**`tail_min: auto` derives it from the window.** It is the smallest count that two
**independent** series would reach with probability ≤ `labels.tail_alpha` (default 1%):

- for independent series, a day is a joint crash with probability 0.05 × 0.05 = 0.0025;
- the number of joint-crash days in W days is therefore Binomial(W, 0.0025);
- `tail_min` is the first k with P(count ≥ k) ≤ 1%.

So "tail dependent" means "more joint crashes than chance would give".

| window W | tail_min (auto) |
|---|---|
| up to 59 | 2 (the paper's value at W=50) |
| 60–175 | 3 |
| 176–330 | 4 |

Set it per run (`--tail-min auto` or a number), or in `configs/config.yaml` for the base
run. With `auto` the base run stays at 2, so the paper is unchanged. The synthetic
transfer sets follow it: a "tail" pair gets `tail_min` or `tail_min + 1` injected
joint-crash days, which is 2–3 at W=50 as before.

## The tau / qLow thresholds: train or rolling

`tau` (a "big move" day, for highVol/lowVol) and `qLow` (a "crash" day, for
tailDependent) are set by `labels.thresholds` (`runs.py new --thresholds ...`):

- **`train`** (the paper): one value per series, fitted on the run's training period.
  After a crisis-heavy training period, a calm test period can then have no volatile
  windows at all. On 2005–2013 / 2014–2019, the S&P 500 had 0 highVol test windows.
- **`rolling`**: per window, the percentile of the series' returns in the 500 trading
  days **before** the window starts. It uses only the past, so there is no look-ahead,
  and "big" means big compared with the last two years. The same run then has 217 highVol
  test windows. Windows with under 250 days of history use the train value, and so do
  synthetic windows, which have no calendar date.

Both the Python labeller and the reasoner receive the thresholds **per window**. Next to
each batch, `_thr/tau_<batch>.csv` and `_thr/qlow_<batch>.csv` hold one row per window.
Until 2026-10-21 the reasoner was given one row per *series*, which never matched the
window names in the batch. As a result, no engine run ever applied tau or qLow (every
window came out lowVol, none highVol, no crash day). Engine results cached before the
fix are therefore recomputed automatically. The correlation regime does not use tau or
qLow, so it was not affected.

## What is shared between runs

- **Shared:** only `data/raw/` (the downloaded series, 2005–2024; Bitcoin from 2017).
- **Run-specific:** everything else is rebuilt per run, from `data_prep` on.
- **Periods outside 2005–2024** need the download dates changed in
  `tslearning/datasets/ingest_financial.py`. `runs.py new` warns when a run's period
  falls outside the data on disk.
