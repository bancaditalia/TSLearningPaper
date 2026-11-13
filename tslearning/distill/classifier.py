"""
The distilled student — V7, the 2-channel interaction encoder.

Only V7 is here. V1-V6 stay in the notebook on purpose: they are the paper's
negative ablation — V1/V2 show per-series generative embeddings
cannot carry pair information, V3/V4 document the circularity trap of feeding
reasoner features at inference. Porting them would imply they are still part of
the pipeline; they are evidence, not code.

Design notes worth keeping straight:
  * Input is (batch, 2, L) — the two windows as CHANNELS, so the convolutions can
    compare them timepoint-by-timepoint. The earlier flat MLP on concatenated
    windows scored macro-F1 ~0.53 against V7's ~0.97; the architecture, not the
    capacity, was the problem.
  * Each window is z-scored ON ITS OWN before stacking (§4.3). That is what lets
    the student transfer across series that live on different scales — and it is
    also why the student cannot predict trend or volatility (standardization
    deletes exactly the scale information those need). Those come from
    labeling/exact_flags.py instead, computed exactly.
  * Single head: the 3-class correlation regime. It amortizes a definition; it does
    not discover correlation.
"""
from __future__ import annotations

import numpy as np

from tslearning.config import CFG


def regime_labels() -> "list[str]":
    """The softmax classes, in the order the head's logits are in."""
    return list(CFG["classifier"]["softmax_group"])


def build_classifier(n_corr: "int | None" = None, ch: int = 32, p: float = 0.3):
    """Instantiate V7. `n_corr` defaults to len(config classifier.softmax_group)."""
    import torch.nn as nn

    n_corr = n_corr if n_corr is not None else len(regime_labels())

    class CorrelationClassifierV7(nn.Module):
        """Correlation regime only. Input (batch, 2, L) -> logits over n_corr classes."""

        def __init__(self, n_corr, ch=32, p=0.3):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv1d(2, 16, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(16),
                nn.Conv1d(16, ch, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(ch),
                nn.AdaptiveAvgPool1d(1),
            )
            self.trunk = nn.Sequential(nn.Flatten(), nn.Dropout(p),
                                       nn.Linear(ch, ch), nn.ReLU())
            self.corr_head = nn.Linear(ch, n_corr)

        def forward(self, x):
            return self.corr_head(self.trunk(self.enc(x)))

    return CorrelationClassifierV7(n_corr, ch=ch, p=p)


def default_weights():
    """The paper's classifier: `classifier.weights` in configs/config.yaml.

    The ONE place the default is decided. Before 2026-09-30 four scripts each hard-coded
    classifier_v7_dominance.pt, so any run without --classifier silently used v7 — which
    is how the fidelity CSVs ended up on v7 while the rest of the paper was on v7b.
    """
    from tslearning.config import CFG, resolve_path

    w = (CFG.get("classifier") or {}).get("weights") or \
        "data/processed/classifier_v7_dominance.pt"
    return resolve_path(w)          # data/processed/... -> the current run's folder


def check_window(weights_path) -> None:
    """Refuse weights trained on another window length than the current run's.

    The architecture accepts any length (adaptive pooling), so a 50-day classifier
    would run on 75-day windows without error — and quietly measure something else.
    The training record (<weights>.json) stores seq_len since 2026-10-19; weights
    without it are the 50-day ones the paper used.
    """
    import json
    from pathlib import Path
    from tslearning.config import CFG

    side = Path(weights_path).with_suffix(".json")
    trained = 50
    if side.exists():
        trained = int(json.loads(side.read_text()).get("seq_len", 50))
    want = int(CFG["windows"]["financial"]["length"])
    if trained != want:
        raise ValueError(f"{Path(weights_path).name} was trained on {trained}-day windows, "
                         f"but run '{CFG['run']['name']}' uses {want}. Train a classifier "
                         f"for this run (distill.py --run ...) and set classifier.weights "
                         f"in its configs/runs/ file.")


def load_classifier(weights_path=None, n_corr: "int | None" = None, device: str = "cpu",
                    verbose: bool = True):
    """Build V7 and load saved weights; returns the model in eval mode.

    `weights_path=None` means the config default (`default_weights()`). The file actually
    loaded is printed, so a log always says which classifier produced a number.
    """
    import torch
    from pathlib import Path

    weights_path = Path(weights_path) if weights_path else default_weights()
    if not weights_path.exists():
        raise FileNotFoundError(f"classifier weights not found: {weights_path}")
    if verbose:
        print(f"classifier weights: {weights_path.name}")
    check_window(weights_path)
    # an ensemble manifest (.ens): load every member, average their probabilities
    from tslearning.distill import ensemble
    if ensemble.is_ensemble(weights_path):
        members = ensemble.read_members(weights_path)
        if verbose:
            print(f"  ensemble of {len(members)}: {', '.join(m.name for m in members)}")
        return ensemble.build([load_classifier(m, n_corr, device, verbose=False)
                               for m in members])
    state = torch.load(weights_path, map_location=device)
    # V8 files carry an auxiliary head (training only); its forward() is V7's, so the
    # rest of the pipeline is unchanged. V7 files load exactly as before.
    from tslearning.distill import classifier_v8
    if classifier_v8.is_v8_state(state):
        model = classifier_v8.from_state(state)
    else:
        model = build_classifier(n_corr)
    model.load_state_dict(state)
    model.eval()
    return model


def build_input(pair_names, lookup, standardize: bool = True) -> np.ndarray:
    """Stack pairs into the (N, 2, L) classifier input.

    pair_names : [(name_a, name_b), ...]; lookup : {window_name: 1-D array}.
    Thin wrapper over data_utils.build_pair_cnn so callers don't have to remember
    that the CNN variant (not build_pair_input, which flattens) is the right one.
    """
    from tslearning.datasets.data_utils import build_pair_cnn
    return build_pair_cnn(pair_names, lookup, standardize=standardize)


def build_timeline_input(windows_a, windows_b, standardize: bool = True) -> np.ndarray:
    """Same shape, but for two already-aligned window arrays (the real timeline
    case, nb_10_part_2 cell B.1) where there are no window names to look up."""
    from tslearning.datasets.data_utils import zscore_window

    z = zscore_window if standardize else (lambda x: np.asarray(x, float))
    n = min(len(windows_a), len(windows_b))
    return np.stack([np.stack([z(windows_a[i]), z(windows_b[i])], axis=0)
                     for i in range(n)]).astype(np.float32)


def predict_regime(model, X, batch_size: int = 4096, return_probs: bool = False):
    """Run the student over (N, 2, L) inputs. Returns argmax indices, or
    (argmax, softmax probabilities) if `return_probs`."""
    import torch

    model.eval()
    logits = []
    with torch.no_grad():
        Xt = torch.as_tensor(np.asarray(X, np.float32))
        for i in range(0, len(Xt), batch_size):
            logits.append(model(Xt[i:i + batch_size]))
        logits = torch.cat(logits) if logits else torch.empty(0)
        pred = logits.argmax(1).numpy()
        if return_probs:
            return pred, torch.softmax(logits, dim=1).numpy()
    return pred
