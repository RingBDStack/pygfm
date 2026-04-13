"""Version-safe torch.load for graph .pt files (may contain PyG objects)."""
from __future__ import annotations

import torch


def load_dataset_pt(path: str, map_location=None):
    """PyTorch 2.6+ defaults weights_only=True; graph data needs full unpickling."""
    kw = {}
    if map_location is not None:
        kw["map_location"] = map_location
    try:
        return torch.load(path, **kw, weights_only=False)
    except TypeError:
        return torch.load(path, **kw)
