"""
Ensemble classifier: several trained members (same recipe, different seeds) whose
regime PROBABILITIES are averaged. Used to report what the method does, not what one
lucky (or unlucky) seed does — on real data the spread across seeds is large.

An ensemble is a small manifest file, `<name>.ens` (JSON), listing the member weight
files with their SHA-256, so the exact members are pinned:

    {"members": [{"path": "data/processed/classifier_v8r_s42.pt", "sha256": "...",
                  "training_record": {...}}, ...],
     "combine": "mean_probability", "created": "..."}

`classifier.load_classifier()` accepts a `.ens` path like a `.pt` one, so infer.py,
eval.py, classifier_timing.py and the config's `classifier.weights` work unchanged.
`forward(x)` returns log(mean softmax): argmax is the ensemble's prediction and
softmax(forward(x)) gives back the averaged probabilities.

Synthetic fidelity is NOT defined for an ensemble: each member held out a different 20%
of the transfer set (different seed -> different split), so no window is unseen by all
members. Report the members' synthetic fidelity (mean ± sd) instead.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

SUFFIX = ".ens"


def is_ensemble(path) -> bool:
    return Path(path).suffix == SUFFIX


def _sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _abs(p) -> Path:
    from tslearning.config import resolve_path
    return resolve_path(p)          # data/processed/... -> the current run's folder


def _rel(p) -> str:
    from tslearning.config import REPO_ROOT
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def write_manifest(members, out_path, overwrite: bool = False) -> Path:
    """Pin the members (path + SHA-256 + their training record) into `out_path`."""
    out_path = Path(out_path)
    if out_path.suffix != SUFFIX:
        raise ValueError(f"an ensemble manifest must end in {SUFFIX}: {out_path}")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists — pass a new name, or --force")
    rows = []
    for m in members:
        p = _abs(m)
        if not p.exists():
            raise FileNotFoundError(p)
        rec = p.with_suffix(".json")
        rows.append({"path": _rel(p), "sha256": _sha256(p),
                     "training_record": json.loads(rec.read_text()) if rec.exists() else None})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"members": rows, "combine": "mean_probability",
                                    "created": f"{datetime.now():%Y-%m-%d %H:%M}"},
                                   indent=2))
    return out_path


def read_members(path) -> list:
    """Member weight paths, after checking each file still has its pinned SHA-256."""
    man = json.loads(Path(path).read_text())
    out = []
    for m in man["members"]:
        p = _abs(m["path"])
        if not p.exists():
            raise FileNotFoundError(f"ensemble member missing: {p}")
        if m.get("sha256") and _sha256(p) != m["sha256"]:
            raise ValueError(f"ensemble member changed since the manifest was written: {p} "
                             f"(SHA-256 mismatch) — rebuild the manifest on purpose")
        out.append(p)
    return out


def build(models):
    import torch
    import torch.nn as nn

    class MeanProbabilityEnsemble(nn.Module):
        def __init__(self, members):
            super().__init__()
            self.members = nn.ModuleList(members)

        def forward(self, x):
            p = torch.stack([torch.softmax(m(x), dim=1) for m in self.members]).mean(0)
            return torch.log(p.clamp_min(1e-12))

    return MeanProbabilityEnsemble(models).eval()
