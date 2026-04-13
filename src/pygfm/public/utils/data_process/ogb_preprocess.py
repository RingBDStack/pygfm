"""OGB graph structure loading (structure only, no DGL)."""
from __future__ import annotations

import os

import torch as th


def load_ogb_graph_structure_only(ogb_name, raw_data_path, save_path="NA"):
    """
    Load OGB graph structure using PyG (no DGL).

    Returns:
      pyg_data: torch_geometric.data.Data
      labels: np.ndarray[int] shape (num_nodes,)
      split_idx: dict with keys train/valid/test (torch tensors of node indices)
    """
    from ..others.runtime import init_path

    graph_path = os.path.join(save_path, "pyg_graph.pt")
    info_path = os.path.join(save_path, "graph_info.pt")
    if save_path == "NA" or (save_path is not None and not os.path.exists(save_path)):
        from ogb.nodeproppred import PygNodePropPredDataset

        data = PygNodePropPredDataset(ogb_name, root=init_path(raw_data_path))
        pyg_data, labels = data[0]
        split_idx = data.get_idx_split()
        labels = labels.squeeze()
        labels_np = labels.cpu().numpy()
        pyg_data.y = labels if labels.dim() == 1 else labels.squeeze()
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            th.save(pyg_data, graph_path)
            th.save({"split_idx": split_idx, "labels": labels_np, "meta_info": data.meta_info}, info_path)
    else:
        pyg_data = th.load(graph_path, map_location="cpu")
        info_dict = th.load(info_path, map_location="cpu")
        split_idx, labels_np = info_dict["split_idx"], info_dict["labels"]
        if getattr(pyg_data, "y", None) is None:
            pyg_data.y = th.tensor(labels_np, dtype=th.long)
    pyg_data.y = pyg_data.y.to(th.long)
    labels_np = pyg_data.y.cpu().numpy()
    return pyg_data, labels_np, split_idx
