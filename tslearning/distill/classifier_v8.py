"""
V8 — V7 plus an AUXILIARY head used only during training.

Same encoder and trunk as V7 (classifier.py, untouched). The extra head regresses the
reasoner's intermediate quantities for the window — by default same%, diff% and their
difference, exactly what the dominance rule decides the regime on. Its loss is added to
the regime loss with weight λ (`aux_weight`) so the shared layers learn WHY a window gets
its label.

At inference the auxiliary head is ignored: `forward(x)` returns the regime logits only,
like V7, so infer.py, eval.py, classifier_timing.py and the rule pipeline use a V8 file
exactly as they use a V7 one. `forward_with_aux(x)` is for training.

The intermediate quantities are training TARGETS, never inputs: at inference they would
have to come from the reasoner, which is what the classifier replaces.
"""
from __future__ import annotations

ARCH = "v8"


def build_classifier_v8(n_corr: int, n_aux: int, ch: int = 32, p: float = 0.3):
    import torch.nn as nn

    class CorrelationClassifierV8(nn.Module):
        """Regime logits (forward) + auxiliary regression (forward_with_aux)."""

        def __init__(self, n_corr, n_aux, ch=32, p=0.3):
            super().__init__()
            # identical to V7, layer for layer
            self.enc = nn.Sequential(
                nn.Conv1d(2, 16, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(16),
                nn.Conv1d(16, ch, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(ch),
                nn.AdaptiveAvgPool1d(1),
            )
            self.trunk = nn.Sequential(nn.Flatten(), nn.Dropout(p),
                                       nn.Linear(ch, ch), nn.ReLU())
            self.corr_head = nn.Linear(ch, n_corr)
            self.aux_head = nn.Linear(ch, n_aux)      # training only

        def forward(self, x):
            return self.corr_head(self.trunk(self.enc(x)))

        def forward_with_aux(self, x):
            h = self.trunk(self.enc(x))
            return self.corr_head(h), self.aux_head(h)

    return CorrelationClassifierV8(n_corr, n_aux, ch=ch, p=p)


def is_v8_state(state: dict) -> bool:
    """A saved state_dict is V8 if it carries the auxiliary head."""
    return any(k.startswith("aux_head.") for k in state)


def from_state(state: dict):
    """Build a V8 sized from its saved weights."""
    n_corr = state["corr_head.weight"].shape[0]
    n_aux = state["aux_head.weight"].shape[0]
    return build_classifier_v8(n_corr, n_aux)
