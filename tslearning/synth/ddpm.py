"""
DDPM synthesizer — the library half of `scripts/synthesize.py`.

The model definition and the training/sampling loops are shared across series, driven
by `synth/registry.py`.

Interface (the E1 "one fit/sample interface" shape):
    schedule = make_noise_schedule(T)
    model    = fit(windows, seq_len, ...)
    samples  = sample(model, n_samples, seq_len, ...)

Known limitation carried over from the notebook, and worth stating in the paper: Gaussian noise gives thin tails, so this DDPM under-produces
big moves — highVol / tailDependent are rare in its output. That is why the
transfer set is engineered rather than drawn from independent per-series samples.

torch is imported lazily so the module can be imported (and the CLI's --help run)
in an environment without it.
"""
from __future__ import annotations

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  Noise schedule + forward process
# ══════════════════════════════════════════════════════════════════════════════

def make_noise_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    """Linear beta schedule. Returns (betas, alphas, alpha_bars) as torch tensors.

    betas      noise added at each step
    alphas     signal retained at each step (1 - beta)
    alpha_bars cumulative signal retained (cumprod of alphas)
    """
    import torch
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def add_noise(x, t, alpha_bars):
    """Closed-form forward diffusion: x_t = sqrt(ab)*x_0 + sqrt(1-ab)*eps.

    x: (batch, seq_len); t: (batch,) — a different timestep per sample.
    Returns (x_noisy, noise) — the noise is the training target.
    """
    import torch
    ab = alpha_bars[t].unsqueeze(-1)
    noise = torch.randn_like(x)
    return torch.sqrt(ab) * x + torch.sqrt(1 - ab) * noise, noise


# ══════════════════════════════════════════════════════════════════════════════
#  Model
# ══════════════════════════════════════════════════════════════════════════════

def _build_modules():
    """Define the nn.Modules at call time, so `import ddpm` doesn't need torch."""
    import torch
    import torch.nn as nn

    class SinusoidalEmbedding(nn.Module):
        """Encode the diffusion timestep t as a vector the network can condition on."""

        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, t):
            half = self.dim // 2
            freqs = torch.exp(-torch.arange(half, dtype=torch.float32)
                              * (np.log(10000) / half))
            args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
            return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    class UNet1D(nn.Module):
        """MLP U-Net with skip connections; predicts the noise added to a window."""

        def __init__(self, seq_len, time_emb_dim=32):
            super().__init__()
            self.time_emb = SinusoidalEmbedding(time_emb_dim)
            self.time_mlp = nn.Sequential(nn.Linear(time_emb_dim, 64), nn.ReLU())
            self.enc1 = nn.Sequential(nn.Linear(seq_len, 128), nn.ReLU())
            self.enc2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
            self.bottleneck = nn.Sequential(nn.Linear(64, 32), nn.ReLU(),
                                            nn.Linear(32, 64), nn.ReLU())
            self.dec2 = nn.Sequential(nn.Linear(64 + 64 + 64, 128), nn.ReLU())
            self.dec1 = nn.Sequential(nn.Linear(128 + 128, seq_len))

        def forward(self, x, t):
            t_emb = self.time_mlp(self.time_emb(t))
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            b = self.bottleneck(e2)
            d2 = self.dec2(torch.cat([b, e2, t_emb], dim=-1))
            return self.dec1(torch.cat([d2, e1], dim=-1))

    return SinusoidalEmbedding, UNet1D


def build_model(seq_len: int, time_emb_dim: int = 32):
    """Instantiate an untrained UNet1D for windows of length `seq_len`."""
    _, UNet1D = _build_modules()
    return UNet1D(seq_len=seq_len, time_emb_dim=time_emb_dim)


# ══════════════════════════════════════════════════════════════════════════════
#  Normalization
# ══════════════════════════════════════════════════════════════════════════════

def normalize(windows: np.ndarray, method: str = "global"):
    """Return (normalized, denorm) where denorm(x) undoes the transform.

    "global"     — one mean/std for the whole set (nb_06 financial series).
    "per_window" — each window z-scored on its own (nb_06 Electricity). Level is
                   unrecoverable per sample, so denorm uses the global stats as a
                   rough rescale, exactly as the notebook did.
    """
    w = np.asarray(windows, dtype=np.float32)
    mu, sd = float(w.mean()), float(w.std())

    if method == "global":
        return (w - mu) / sd, (lambda x: x * sd + mu)
    if method == "per_window":
        z = (w - w.mean(axis=1, keepdims=True)) / (w.std(axis=1, keepdims=True) + 1e-8)
        return z, (lambda x: x * sd + mu)
    raise ValueError(f"unknown normalize method: {method!r}")


# ══════════════════════════════════════════════════════════════════════════════
#  Train / sample
# ══════════════════════════════════════════════════════════════════════════════

def fit(windows_norm: np.ndarray, seq_len: int, diffusion_steps: int = 500,
        epochs: int = 200, batch_size: int = 128, lr: float = 1e-3,
        seed: "int | None" = None, log_every: int = 20, verbose: bool = True):
    """Train the denoiser on already-normalized windows.

    Returns (model, schedule, losses) — schedule is the (betas, alphas, alpha_bars)
    tuple, needed again at sampling time.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    betas, alphas, alpha_bars = make_noise_schedule(diffusion_steps)
    model = build_model(seq_len)
    opt = optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(TensorDataset(torch.tensor(np.asarray(windows_norm, np.float32))),
                        batch_size=batch_size, shuffle=True)
    if verbose:
        print(f"training DDPM: {len(windows_norm)} windows, seq_len={seq_len}, "
              f"T={diffusion_steps}, {sum(p.numel() for p in model.parameters()):,} params")

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for (x,) in loader:
            opt.zero_grad()
            t = torch.randint(0, diffusion_steps, (x.size(0),))
            x_noisy, noise = add_noise(x, t, alpha_bars)
            loss = nn.functional.mse_loss(model(x_noisy, t), noise)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss)
        if verbose and (epoch + 1) % log_every == 0:
            print(f"  epoch {epoch+1}/{epochs}  loss {epoch_loss:.4f}")

    return model, (betas, alphas, alpha_bars), losses


def sample(model, n_samples: int, seq_len: int, schedule, seed: "int | None" = None):
    """Reverse diffusion from pure noise. Returns a (n_samples, seq_len) array."""
    import torch

    betas, alphas, alpha_bars = schedule
    T = len(betas)
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    with torch.no_grad():
        x = torch.randn(n_samples, seq_len)
        for t in reversed(range(T)):
            t_batch = torch.full((n_samples,), t, dtype=torch.long)
            noise_pred = model(x, t_batch)
            ab, a, b = alpha_bars[t], alphas[t], betas[t]
            x = (1 / torch.sqrt(a)) * (x - (b / torch.sqrt(1 - ab)) * noise_pred)
            if t > 0:
                x = x + torch.sqrt(b) * torch.randn_like(x)
    return x.numpy()


# ══════════════════════════════════════════════════════════════════════════════
#  Stage runner
# ══════════════════════════════════════════════════════════════════════════════

def run(dataset: str, epochs: "int | None" = None, n_samples: "int | None" = None,
        diffusion_steps: "int | None" = None, batch_size: "int | None" = None,
        lr: "float | None" = None, save: bool = True, verbose: bool = True) -> dict:
    """Train a DDPM for one dataset and save samples + weights.

    Reads windowed.pkl / windowed_macro.pkl (whichever the registry says), writes
    synthetic_ddpm_<dataset>.npy and ddpm_unet1d_<dataset>.pt — the same filenames
    nb_06 produced, so every downstream consumer is unaffected.
    """
    import torch
    from tslearning.config import CFG
    from tslearning.datasets.data_utils import (load_windowed, load_windowed_macro,
                                                window_for)
    from tslearning.synth import registry

    sp = registry.spec(dataset, epochs=epochs, n_samples=n_samples,
                       diffusion_steps=diffusion_steps, batch_size=batch_size, lr=lr)
    seed = CFG["seed"]

    windowed = load_windowed_macro() if sp["source"] == "macro" else load_windowed()
    if dataset not in windowed:
        raise KeyError(f"{dataset} not in {sp['source']} windowed data. "
                       f"Run data_prep first. Available: {sorted(windowed)}")

    train = np.asarray(windowed[dataset]["train"], dtype=np.float32)
    seq_len = int(windowed[dataset].get("window_len") or window_for(dataset)[0])

    norm, denorm = normalize(train, sp["normalize"])
    model, schedule, losses = fit(norm, seq_len,
                                 diffusion_steps=sp["diffusion_steps"],
                                 epochs=sp["epochs"], batch_size=sp["batch_size"],
                                 lr=sp["lr"], seed=seed, verbose=verbose)

    samples = denorm(sample(model, sp["n_samples"], seq_len, schedule, seed=seed))

    paths = {}
    if save:
        s_path = registry.sample_path(dataset, sp["model"])
        w_path = registry.weights_path(dataset, sp["model"])
        s_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(s_path, samples)
        torch.save(model.state_dict(), w_path)
        paths = {"samples": s_path, "weights": w_path}
        if verbose:
            print(f"wrote -> {s_path.name}, {w_path.name}")

    if verbose:
        print(f"synthetic std {samples.std():.4f}  (real {train.std():.4f})  "
              f"final loss {losses[-1]:.4f}")

    return {"dataset": dataset, "spec": sp, "samples": samples,
            "losses": losses, "paths": paths, "seq_len": seq_len}
