"""
Synthesizer registry — which dataset trains with which hyperparameters, and where
its outputs go.

nb_06 / nb_06b hardcode one cell per series (SP500, NASDAQ, Bitcoin, Electricity,
macro), each repeating the same training loop with different variable names. This
module holds only what actually differs between them, so `scripts/synthesize.py -d
<name>` can serve any series with one code path.

`normalize` is the one substantive difference between the financial and the
seasonal/macro cases:
  "global"     - one mean/std over the whole training set (nb_06: SP500, NASDAQ,
                 Bitcoin). Keeps the relative scale of windows, which the
                 stylized-fact fingerprints depend on.
  "per_window" - each window z-scored on its own (nb_06: Electricity). Removes the
                 level so the model learns shape only; needed when windows sit at
                 very different levels.

The aim is "wrap synthesizers behind one fit/sample interface" — this
registry plus ddpm.py is the DDPM instance of that interface. A second model
(Diffusion-TS, TimeGAN) slots in by adding a `model` key here and a module beside
ddpm.py exposing the same fit()/sample() pair.
"""
from __future__ import annotations

from tslearning.config import CFG

# Defaults from nb_06 cell 6 (SP500/NASDAQ/Bitcoin) and cells 19-21 (Electricity).
DEFAULTS = {
    "model": "ddpm",
    "diffusion_steps": 500,
    "batch_size": 128,
    "epochs": 200,
    "lr": 1e-3,
    "n_samples": 1000,
    "normalize": "global",
    "source": "financial",   # which windowed pkl to read: financial | macro
}

# Per-dataset overrides only. Anything absent falls back to DEFAULTS.
REGISTRY = {
    "SP500":       {},
    "NASDAQ":      {},
    "Bitcoin":     {},
    "Gold":        {},
    "UST10Y":      {},
    "TLT":         {},
    "UST10Z":      {},
    "VIX":         {},
    "Electricity": {"normalize": "per_window", "epochs": 100, "n_samples": 100},
    "Airline":     {"normalize": "per_window", "epochs": 100, "n_samples": 100},
}

# Macro series are registered from the config so the list stays in one place.
MACRO_FREQS = {"macro_monthly", "macro_quarterly"}


def macro_series() -> "list[str]":
    """Every series whose configured frequency is a macro one."""
    return [name for name, freq in CFG["series_freq"].items() if freq in MACRO_FREQS]


def is_macro(dataset: str) -> bool:
    return CFG["series_freq"].get(dataset) in MACRO_FREQS


def spec(dataset: str, **overrides) -> dict:
    """Resolved hyperparameter spec for one dataset.

    Precedence: explicit `overrides` (CLI flags) > REGISTRY entry > DEFAULTS.
    A macro dataset not listed in REGISTRY still resolves — it gets the defaults
    with source="macro" and per-window normalization (macro windows are short and
    their levels drift, so shape-only is the safer default).
    """
    if dataset not in CFG["series_freq"]:
        raise KeyError(
            f"unknown dataset '{dataset}'. Add it to series_freq in configs/config.yaml. "
            f"Known: {sorted(CFG['series_freq'])}")

    out = dict(DEFAULTS)
    if is_macro(dataset):
        out.update({"source": "macro", "normalize": "per_window",
                    "epochs": 300, "n_samples": 500, "batch_size": 32})
    out.update(REGISTRY.get(dataset, {}))
    out.update({k: v for k, v in overrides.items() if v is not None})
    out["dataset"] = dataset
    return out


def known_datasets() -> "list[str]":
    """Everything `synthesize.py -d` accepts."""
    return sorted(set(REGISTRY) | set(macro_series()))


# ── Output naming ──────────────────────────────────────────────────────────────
# Kept byte-compatible with what nb_06 wrote, so nb_10 and friends keep loading
# the same filenames: synthetic_ddpm_sp500.npy, ddpm_unet1d_sp500.pt.

def sample_path(dataset: str, model: str = "ddpm"):
    from tslearning.config import PATHS
    return PATHS["data_processed"] / f"synthetic_{model}_{dataset.lower()}.npy"


def weights_path(dataset: str, model: str = "ddpm"):
    from tslearning.config import PATHS
    arch = "unet1d" if model == "ddpm" else model
    return PATHS["data_processed"] / f"{model}_{arch}_{dataset.lower()}.pt"
