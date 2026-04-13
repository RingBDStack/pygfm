"""PyTorch >= 2.6 defaults torch.load(..., weights_only=True), which breaks PyG Data pickles."""

import inspect
from typing import Any, Optional

import torch


def torch_load_compat(path: str, map_location: Optional[Any] = None, **kwargs: Any):
    load_kw = dict(kwargs)
    if map_location is not None:
        load_kw["map_location"] = map_location
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kw.setdefault("weights_only", False)
    return torch.load(path, **load_kw)
