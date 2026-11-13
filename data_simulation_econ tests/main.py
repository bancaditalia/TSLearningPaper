"""Run the whole pipeline: real data, descriptive tables and figures, econometric tests, simulation, CSV export.

Usage (from any folder):
  python main.py                         every step; exports the core simulated datasets
  python main.py data tables             only the named steps (in pipeline order)
  python main.py export --sets core long grid
  python main.py --force-simulation      regenerate the simulation even if simulated datasets already exist

The simulation does not start if simulated datasets already exist in simulation/datasets/: the calibration, the
Monte Carlo and the export are skipped (the summaries and plots are still redrawn from the stored results). To
regenerate everything, uncomment FORCE_SIMULATION = True below, or pass --force-simulation, or answer 'y' to the
question asked when main.py is run from a terminal. Regenerating recalibrates the model, reruns the Monte Carlo from scratch (old results kept as
simulation/results_econ_previous.csv) and overwrites the datasets.

Steps, in order:
  data        market_data/build.py            daily levels and log returns from market_data/raw (sample to 2024)
              market_data/macro_states.py     real-time inflation states
              market_data/regime_episodes.py  correlation-regime episodes and reversals
  tables      regime_tables.py                regime tables
              descriptives.py                 descriptive tables and figures
              plot_returns_inflation.py       returns and inflation figure
              plot_rule_examples.py           concordance of S&P-gold and S&P-bond with rule conditions
  tests       econometric_tests.py            econometric benchmark tests, rules vs tests
  simulation  simulation/calibrate.py         calibration of the DGP on the series of the rules -> calibration.json
              simulation/check_dgp.py         simulated vs real moments (printed only)
              simulation/mc_econ.py           Monte Carlo of the tests, 200 replications (resumable: only
                                              configurations missing from results_econ.csv are run; hours)
              simulation/summarize_econ.py    power figure and size/power table
              simulation/plot_sim_series.py   plots of single simulated samples
  export      simulation/export_datasets.py   simulated datasets as .csv.gz for the rule pipeline
                                              (simulation/datasets/<set>/; existing files are kept)

Outputs: figures_tables_paper/ (main text) and figures_tables_repository/ (everything else).
Stops at the first failing script.
"""
import os
import sys
import time
import shutil
import argparse
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
STEPS = {
    'data': ['market_data/build.py', 'market_data/macro_states.py', 'market_data/regime_episodes.py'],
    'tables': ['regime_tables.py', 'descriptives.py', 'plot_returns_inflation.py', 'plot_rule_examples.py'],
    'tests': ['econometric_tests.py'],
    'simulation': ['simulation/calibrate.py', 'simulation/check_dgp.py', 'simulation/mc_econ.py',
                   'simulation/summarize_econ.py', 'simulation/plot_sim_series.py'],
    'export': ['simulation/export_datasets.py'],
}
REQUIRED = ['numpy', 'pandas', 'scipy', 'matplotlib', 'statsmodels', 'arch']  # see requirements.txt
# Simulation switch. By default the simulation is NOT regenerated when simulated datasets already exist in
# simulation/datasets/ (calibration, Monte Carlo and export are skipped). Uncomment the second line to regenerate
# it anyway (recalibration, Monte Carlo from scratch, datasets overwritten; about an hour), then comment it again.
FORCE_SIMULATION = False
# FORCE_SIMULATION = True

GENERATE = {'simulation/calibrate.py', 'simulation/mc_econ.py', 'simulation/export_datasets.py'}  # skipped if data exist
MC_REPS = '200'   # replications per configuration in the Monte Carlo (as in results_econ.csv)


def run(script, args=(), env=None):
    path = os.path.join(ROOT, script)
    print(f'\n=== {script} {" ".join(args)}', flush=True)
    t0 = time.time()
    res = subprocess.run([sys.executable, path, *args], cwd=os.path.dirname(path), env=env)
    print(f'--- {script}: {"ok" if res.returncode == 0 else "FAILED"} ({time.time() - t0:.0f} s)', flush=True)
    if res.returncode != 0:
        sys.exit(res.returncode)


def simulated_data_exist():
    base = os.path.join(ROOT, 'simulation', 'datasets')
    return any(f.endswith('.csv.gz') for _, _, files in os.walk(base) for f in files)


def ask_rerun():
    """The 'button': when run from a terminal, ask whether to regenerate the existing simulation."""
    if not sys.stdin.isatty():
        return False
    ans = input('Simulated datasets already exist in simulation/datasets/. Regenerate the simulation '
                '(recalibration, Monte Carlo from scratch, datasets overwritten; about an hour)? [y/N] ')
    return ans.strip().lower() in ('y', 'yes', 's', 'si', 'sì')


def main():
    p = argparse.ArgumentParser(description='Run the pipeline (see the docstring of main.py).')
    p.add_argument('steps', nargs='*', help='steps to run: ' + ', '.join(STEPS) + ' (default: all)')
    p.add_argument('--sets', nargs='+', default=['core'], help='simulated datasets to export: core (default), long, grid')
    p.add_argument('--force-simulation', action='store_true',
                   help='regenerate the simulation even if simulated datasets exist: recalibrate, rerun the Monte Carlo '
                        'from scratch (old results kept as results_econ_previous.csv) and overwrite the datasets')
    a = p.parse_args()
    missing = [m for m in REQUIRED if __import__('importlib.util').util.find_spec(m) is None]
    if missing:
        sys.exit(f'Missing packages: {", ".join(missing)}. Install them in this environment with\n'
                 f'    {sys.executable} -m pip install -r requirements.txt')
    bad = [s for s in a.steps if s not in STEPS]
    if bad:
        p.error(f'unknown step(s) {bad}; choose from {list(STEPS)}')
    steps = [s for s in STEPS if s in (a.steps or STEPS)]

    env = dict(os.environ, MC_REPS=MC_REPS, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', MPLBACKEND='Agg')
    regenerate = True
    if {'simulation', 'export'} & set(steps) and simulated_data_exist():
        regenerate = FORCE_SIMULATION or a.force_simulation or ask_rerun()
        if not regenerate:
            print('Simulated datasets found in simulation/datasets/: calibration, Monte Carlo and export are skipped. '
                  'Use --force-simulation to regenerate them.')
    if regenerate and 'simulation' in steps:
        res = os.path.join(ROOT, 'simulation', 'results_econ.csv')
        if os.path.exists(res) and simulated_data_exist():   # a full regeneration starts the Monte Carlo afresh
            shutil.move(res, os.path.join(ROOT, 'simulation', 'results_econ_previous.csv'))

    t0 = time.time()
    for s in steps:
        for script in STEPS[s]:
            if script in GENERATE and not regenerate:
                print(f'\n=== {script}: skipped (simulated datasets exist)')
                continue
            args = (list(a.sets) + (['--force'] if s == 'export' and simulated_data_exist() else [])) if s == 'export' else ()
            run(script, args, env)
    print(f'\nDone: {", ".join(steps)} ({(time.time() - t0) / 60:.1f} min)')


if __name__ == '__main__':
    main()
