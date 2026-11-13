"""Data-generating processes for the simulation study of the rule templates.

Each replication is a pair of daily log-return series (X like the S&P 500, Y like gold) with GJR-GARCH /
GARCH(1,1) volatility and Student-t innovations, calibrated on S&P 500 and COMEX gold returns, 2000-2024. The two
series are linked by a time-varying correlation rho_t of their standardised innovations. A template's
phenomenon is planted through rho_t (or a lead term) with strength `theta`; theta = 0 is the null.

States that drive rho_t are computed from the 50-day window ending the previous day (predetermined), with
the same definitions as the predicates:
  highVol(X)    at least 30% of the window above tau_X, the 90th percentile of |r_X| (as in the rules)
  lowVol        at most 5% of the window above tau (each series)
  trendDown(X)  sum of returns over the window below -0.02
Phenomena (rho0 = background correlation, the "baseline" dimension):
  T1b safe haven      rho_t = rho0 - theta * extremeDown_X,t       (X's shock in its lower 10% tail on day t:
                                                                   the Baur-Lucey definition of a safe haven)
  T2  contagion       rho_t = rho0 + theta * highVol_X             (null: constant rho, but X's variance
                                                                   still rises in crises: the Forbes-Rigobon trap)
  T3b decoupling      rho_t = rho0 * (1 - theta * bothCalm)        (theta = 1: zero correlation in calm)
  T4  lead in stress  z_Y,t = rho0 z_X,t + theta * highVol_X * z_X,t-k + noise   (k = 5)
  T7  macro           rho_t = rho0 + theta * inflationHigh_t       (as in the rules: monthly AR(1) of the
                                                                   m/m log change of CPI, calibrated on
                                                                   1985-2024, smoothed with a centred 3-month
                                                                   moving average; high = above the 75th
                                                                   percentile of the training months; the
                                                                   state of month m holds on its days)
Calibration (marginals and inflation) is read from calibration.json, written by calibrate.py on the series
used by the rules (gold: the July 2026 vintage).
"""
import os
import json
import numpy as np

W = 50
TRADING_MONTH = 21
# Daily returns in %. S&P 500: GJR-GARCH(1,1)-t (volatility rises more after falls); gold: GARCH(1,1)-t.
# Estimated on 2000-2024 by calibrate.py (arch, dist='t'); omega set by variance targeting so that the
# unconditional variance equals the sample variance, since the fitted models are near-integrated.
_cal = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration.json')))
CAL = {'X': dict(_cal['X']), 'Y': dict(_cal['Y'])}
for _c in CAL.values():
    _c['omega'] = _c['var'] * (1 - _c['alpha'] - _c['gamma'] / 2 - _c['beta'])
INFL = {k: _cal['INFL'][k] for k in ('const', 'phi', 'sd')}  # monthly AR(1) of m/m CPI inflation (% p.a.)
LEAD_K = 5
BURN = 500


def std_t(rng, nu, size):
    return rng.standard_t(nu, size) * np.sqrt((nu - 2) / nu)


def _thresholds(seed=12345, n=200_000):
    """tau = 90th percentile of |r| for X and Y under the calibrated marginals (null, rho = 0)."""
    x, y = _garch_pair(np.random.default_rng(seed), 1, n, rho_fn=None)
    return np.quantile(np.abs(x), 0.90), np.quantile(np.abs(y), 0.90)


def _garch_pair(rng, reps, n, rho_fn=None, lead_fn=None, infl=None, record=None):
    """Core simulator. rho_fn(states) -> rho_t (reps,), lead_fn(states) -> lead loading (reps,).
    record: optional dict filled with the planted rho_t, lead loading and states (for plots; no effect on draws)."""
    cx, cy = CAL['X'], CAL['Y']
    T = n + BURN
    ux, uy = std_t(rng, cx['nu'], (reps, T)), std_t(rng, cy['nu'], (reps, T))
    rx, ry = np.zeros((reps, T)), np.zeros((reps, T))
    zx = np.zeros((reps, T))
    keep = {k: np.zeros((reps, T)) for k in ('rho', 'lead', 'highVol', 'bothCalm', 'extremeDown')} if record is not None else None
    s2x, s2y = np.full(reps, cx['var']), np.full(reps, cy['var'])
    ex_x, ex_y, sum_x = np.zeros(reps), np.zeros(reps), np.zeros(reps)  # rolling window counters
    # thresholds in % (returns are simulated in %); TAU is stored in log-return units
    tau_x, tau_y = (100 * TAU[0], 100 * TAU[1]) if rho_fn is not None or lead_fn is not None else (np.inf, np.inf)
    for t in range(T):
        if t > 0:
            ax, ay = rx[:, t - 1], ry[:, t - 1]
            ex, ey = ax - cx['mu'], ay - cy['mu']
            s2x = cx['omega'] + (cx['alpha'] + cx['gamma'] * (ex < 0)) * ex ** 2 + cx['beta'] * s2x
            s2y = cy['omega'] + (cy['alpha'] + cy['gamma'] * (ey < 0)) * ey ** 2 + cy['beta'] * s2y
            ex_x += np.abs(ax) > tau_x; ex_y += np.abs(ay) > tau_y; sum_x += ax / 100
            if t - 1 - W >= 0:
                bx, by = rx[:, t - 1 - W], ry[:, t - 1 - W]
                ex_x -= np.abs(bx) > tau_x; ex_y -= np.abs(by) > tau_y; sum_x -= bx / 100
        full = t >= W
        states = {
            'extremeDown': ux[:, t] < Q10_X,
            'highVol': full & (ex_x / W >= 0.30),
            'bothCalm': full & (ex_x / W <= 0.05) & (ex_y / W <= 0.05),
            'stressedDown': full & (ex_x / W >= 0.30) & (sum_x < -0.02),
            'infl': infl[:, t] if infl is not None else None,
        }
        rho = np.zeros(reps) if rho_fn is None else np.clip(rho_fn(states), -0.95, 0.95)
        lead = np.zeros(reps) if lead_fn is None else lead_fn(states)
        if keep is not None:
            keep['rho'][:, t], keep['lead'][:, t] = rho, lead
            for k in ('highVol', 'bothCalm', 'extremeDown'):
                keep[k][:, t] = states[k]
        zx[:, t] = ux[:, t]
        zlag = zx[:, t - LEAD_K] if t >= LEAD_K else 0.0
        zy = rho * ux[:, t] + lead * zlag + np.sqrt(np.maximum(1 - rho ** 2 - lead ** 2, 0.01)) * uy[:, t]
        rx[:, t] = cx['mu'] + np.sqrt(s2x) * ux[:, t]
        ry[:, t] = cy['mu'] + np.sqrt(s2y) * zy
    if keep is not None:
        record.update({k: v[:, BURN:] for k, v in keep.items()})
    return rx[:, BURN:] / 100, ry[:, BURN:] / 100


def _inflation(rng, reps, n, train_frac):
    """Daily high-inflation indicator as in the rules: monthly AR(1) of m/m inflation, centred 3-month moving
    average, high above the 75th percentile of the training months; the state of month m holds on its days."""
    months = (n + BURN) // TRADING_MONTH + 30
    pi = np.zeros((reps, months))
    pi[:, 0] = INFL['const'] / (1 - INFL['phi'])
    e = rng.normal(0, INFL['sd'], (reps, months))
    for m in range(1, months):
        pi[:, m] = INFL['const'] + INFL['phi'] * pi[:, m - 1] + e[:, m]
    cma = np.full_like(pi, np.nan)
    cma[:, 1:-1] = (pi[:, :-2] + pi[:, 1:-1] + pi[:, 2:]) / 3        # centred: months m-1, m, m+1
    day_month = np.arange(n + BURN) // TRADING_MONTH + 24          # 24 months of warm-up
    state = cma[:, day_month]                                        # month m's state on its own days
    first, last = day_month[BURN], day_month[BURN + int(train_frac * n) - 1]
    p75 = np.nanquantile(cma[:, first:last + 1], 0.75, axis=1)      # training months of the sample
    return (state > p75[:, None]).astype(float)


def simulate(template, theta, rho0, n, reps, seed, train_frac=0.7, record=None):
    """Return (X, Y, I): arrays (reps, n) of daily log returns, and the high-inflation indicator (T7 only)."""
    rng = np.random.default_rng(seed)
    infl = None
    if template == 'T1b':
        rho_fn, lead_fn = (lambda s: rho0 - theta * s['extremeDown']), None
    elif template == 'T2':
        rho_fn, lead_fn = (lambda s: rho0 + theta * s['highVol']), None
    elif template == 'T3b':
        rho_fn, lead_fn = (lambda s: rho0 * (1 - theta * s['bothCalm'])), None
    elif template == 'T4':
        rho_fn, lead_fn = (lambda s: np.full(reps, rho0)), (lambda s: theta * s['highVol'])
    elif template == 'T7':
        infl = _inflation(rng, reps, n, train_frac)
        rho_fn, lead_fn = (lambda s: rho0 + theta * s['infl']), None
    else:
        raise ValueError(template)
    x, y = _garch_pair(rng, reps, n, rho_fn=rho_fn, lead_fn=lead_fn, infl=infl, record=record)
    return x, y, (infl[:, BURN:] if infl is not None else None)


from scipy import stats as _st
Q10_X = _st.t.ppf(0.10, CAL['X']['nu']) * np.sqrt((CAL['X']['nu'] - 2) / CAL['X']['nu'])  # 10% quantile of z_X
TAU = (np.inf, np.inf)
TAU = _thresholds()
