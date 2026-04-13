"""Load graph tensors used by the attack pipeline."""

from __future__ import annotations

import torch

from pygfm.baseline_models.sa2gfm.paths import paths


def load_graph(dataset_name: str) -> torch.Tensor:
    """
    Load ``Data`` from ``resolve_ori_graph_pt`` (``ori/``, flat ``*.pt``, or ``sa2gfm/*.pt``).
    Downstream attacks expect at least: ``enhanced_x_64``, ``edge_index``, ``y``.
    """
    p = paths.resolve_ori_graph_pt(dataset_name)
    # PyTorch 2.6+ defaults weights_only=True; PyG Data objects need weights_only=False.
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(p, map_location="cpu")
