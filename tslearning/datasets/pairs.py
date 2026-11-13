"""
Pair resolution — turn a pair spec on the command line into two real series.

The problem this solves: pairs used to be parsed with `pair.split("_", 1)`, which
works for financial names (`sp500_gold` -> sp500, gold) and silently breaks for
macro ones, because macro series contain underscores themselves:

    "us_inflation_us_unemployment".split("_", 1)  ->  ("us", "inflation_us_unemployment")

So instead of splitting blindly, we try every underscore position and keep the split
where BOTH halves are series the config knows about. `us_inflation_us_unemployment`
resolves correctly, and `sp500_gold` still resolves exactly as before.

A colon form is also accepted for cases where the underscore form would be ambiguous
(two different valid splits) or where you'd rather be explicit:

    US_Inflation:US_Unemployment
    SP500:Gold

Beyond parsing, this module answers the questions every stage needs before it can
process a pair: which pkl holds these series (financial vs macro), what window length
applies, and whether the two sides are even the same frequency.
"""
from __future__ import annotations

from tslearning.config import CFG

MACRO_FREQS = {"macro_monthly", "macro_quarterly"}


def _lut() -> dict:
    """lowercase name -> canonical config name."""
    return {n.lower(): n for n in CFG["series_freq"]}


def is_macro(series: str) -> bool:
    return CFG["series_freq"].get(series) in MACRO_FREQS


def freq_of(series: str) -> str:
    """The configured frequency bucket of a series ('financial', 'macro_monthly', ...)."""
    try:
        return CFG["series_freq"][series]
    except KeyError:
        raise KeyError(f"'{series}' has no series_freq entry in configs/config.yaml")


def resolve_pair(spec: str) -> "tuple[str, str, str]":
    """Parse a pair spec into (series_a, series_b, slug).

    Accepts 'SP500:Gold', 'sp500_gold', 'US_Inflation:US_Unemployment' and
    'us_inflation_us_unemployment'. Case-insensitive.

    The slug ('sp500_gold', 'us_inflation_us_unemployment') is what names output
    folders and CSVs. It is only ever a LABEL — nothing re-parses it — so the
    underscore ambiguity stops at this function.
    """
    lut = _lut()

    if ":" in spec:
        left, right = spec.split(":", 1)
        a_key, b_key = left.strip().lower(), right.strip().lower()
        if a_key not in lut or b_key not in lut:
            missing = [k for k in (a_key, b_key) if k not in lut]
            raise KeyError(f"unknown series {missing} — known: {sorted(lut)}")
        a, b = lut[a_key], lut[b_key]
    else:
        low = spec.strip().lower()
        candidates = []
        for i, ch in enumerate(low):
            if ch != "_":
                continue
            left, right = low[:i], low[i + 1:]
            if left in lut and right in lut:
                candidates.append((lut[left], lut[right]))
        if not candidates:
            raise KeyError(
                f"cannot resolve pair '{spec}' — no split into two known series. "
                f"Use the explicit form, e.g. 'US_Inflation:US_Unemployment'. "
                f"Known series: {sorted(lut)}")
        if len(candidates) > 1:
            raise KeyError(
                f"pair '{spec}' is ambiguous ({candidates}) — use the colon form "
                f"to disambiguate, e.g. '{candidates[0][0]}:{candidates[0][1]}'")
        a, b = candidates[0]

    return a, b, f"{a.lower()}_{b.lower()}"


def check_same_frequency(a: str, b: str) -> str:
    """Both sides must share a frequency; returns it.

    Mixed granularity (daily equity vs monthly CPI) is template T7 in
    genuinely interesting, and genuinely a different
    computation, since one window of X spans many windows of Y. Rather than
    silently aligning a daily series onto ~250 monthly points and producing
    nonsense, this refuses and says so.
    """
    fa, fb = freq_of(a), freq_of(b)
    if fa != fb:
        raise ValueError(
            f"mixed granularity: {a} is {fa}, {b} is {fb}. Pairing across "
            "frequencies is the T7 macro-conditioning template and is not "
            "supported by this path — pair like with like "
            "(financial x financial, or macro x macro at the same frequency).")
    return fa


def load_series(names) -> dict:
    """Load the transformed series for `names`, from whichever pkl holds each.

    Financial series live in transformed.pkl, macro in transformed_macro.pkl.
    Returns {canonical_name: pd.Series}, reading each file at most once.
    """
    from tslearning.datasets.data_utils import load_transformed, load_transformed_macro

    names = list(names)
    need_macro = any(is_macro(n) for n in names)
    need_fin = any(not is_macro(n) for n in names)

    pool = {}
    if need_fin:
        pool.update(load_transformed())
    if need_macro:
        pool.update(load_transformed_macro())

    missing = [n for n in names if n not in pool]
    if missing:
        which = "data_prep.py -d macro" if any(is_macro(n) for n in missing) \
                else "data_prep.py -d financial"
        raise KeyError(f"{missing} not found in the transformed pkl — run `{which}` first")

    return {n: pool[n] for n in names}


def window_for_pair(a: str, b: str) -> "tuple[int, int]":
    """(length, step) for a pair — both sides share a frequency, so either side's
    window config applies.

    Reads the config directly rather than going through data_utils.window_for, which
    would drag in scipy/statsmodels for what is a two-key dictionary lookup.
    """
    freq = check_same_frequency(a, b)
    w = CFG["windows"][freq]
    return w["length"], w.get("step", 1)


def describe(spec: str) -> dict:
    """Everything a stage needs to know about a pair spec, in one call."""
    a, b, slug = resolve_pair(spec)
    freq = check_same_frequency(a, b)
    length, step = window_for_pair(a, b)
    return {"a": a, "b": b, "slug": slug, "freq": freq,
            "a_key": a.lower(), "b_key": b.lower(),
            "seq_len": length, "step": step, "is_macro": is_macro(a)}
