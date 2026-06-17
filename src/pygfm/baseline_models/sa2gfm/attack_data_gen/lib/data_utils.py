"""Load graph tensors used by the attack pipeline."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from pygfm.baseline_models.sa2gfm.paths import paths
from pygfm.public.utils.runtime import _maybe_hetero_to_homogeneous, _torch_load_to_single_data


def load_graph(dataset_name: str) -> Data:
    """
    Load a single :class:`~torch_geometric.data.Data` from ``resolve_ori_graph_pt``.

    ``*.pt`` may store raw ``Data``, ``HeteroData``, or a PyG InMemory ``(dict, slices, cls)``
    tuple — all are normalized here. Downstream expects at least ``edge_index`` (and usually
    ``x``, ``y``).
    """
    p = paths.resolve_ori_graph_pt(dataset_name)
    try:
        raw = torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(p, map_location="cpu")
    raw = _maybe_hetero_to_homogeneous(raw)
    return _torch_load_to_single_data(raw)
