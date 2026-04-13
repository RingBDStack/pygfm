"""Adapt PyG-style .pt (dict/Data/tuple) for LLaGA training fields."""
from __future__ import annotations

import os
from typing import Any, List, Tuple

import torch

from .io_utils import load_dataset_pt
from .utils.constants import DEFAULT_GRAPH_TOKEN
from .utils.data_process import generate_edge_list, get_fix_shape_subgraph_sequence_fast

# Cora label order aligned with eval_pretrain.py (Planetoid)
CORA_LABEL_TEXTS = [
    "Case_Based",
    "Genetic_Algorithms",
    "Neural_Networks",
    "Probabilistic_Methods",
    "Reinforcement_Learning",
    "Rule_Learning",
    "Theory",
]


def unwrap_pt_root(obj: Any) -> Any:
    if isinstance(obj, tuple) and len(obj) >= 1:
        return obj[0]
    return obj


def is_pyg_like_dict(d: dict) -> bool:
    return "x" in d and "edge_index" in d and "y" in d


def is_llaga_processed(obj: Any) -> bool:
    if isinstance(obj, dict):
        return "label_texts" in obj
    return hasattr(obj, "label_texts")


class PyGCompatData:
    """Align with upstream processed_data.pt Cora fields for NC and graph indices."""

    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor,
        label_texts: List[str],
        train_mask: torch.Tensor | None = None,
    ):
        self.x = x
        self.edge_index = edge_index
        self.y = y.view(-1).long()
        self.num_nodes = int(x.shape[0])
        self.label_texts = label_texts
        self.title = [""] * self.num_nodes
        if train_mask is not None:
            self.train_mask = train_mask.view(-1).bool()
        else:
            self.train_mask = torch.ones(self.num_nodes, dtype=torch.bool)


def dict_to_pyg_compat(d: dict, dataset_name: str) -> PyGCompatData:
    x = d["x"]
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    ei = d["edge_index"]
    if not isinstance(ei, torch.Tensor):
        ei = torch.as_tensor(ei, dtype=torch.long)
    y = d["y"]
    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y)
    n_cls = int(y.view(-1).max().item()) + 1
    if dataset_name == "cora" and n_cls == 7:
        label_texts = list(CORA_LABEL_TEXTS)
    else:
        label_texts = [f"class_{i}" for i in range(n_cls)]
    tm = d.get("train_mask")
    if tm is not None and not isinstance(tm, torch.Tensor):
        tm = torch.as_tensor(tm)
    return PyGCompatData(x, ei, y, label_texts, tm)


def normalize_loaded_graph(raw: Any, dataset_name: str) -> Tuple[Any, bool]:
    """
    Returns:
        (data, is_pyg_compat): True if wrapped from PyG dict/Data,
        use pretrained_embedding_type=node_x when jsonl/simteg missing.
    """
    raw = unwrap_pt_root(raw)
    try:
        from torch_geometric.data import Data

        if isinstance(raw, Data):
            raw = raw.to_dict()
    except ImportError:
        pass
    if isinstance(raw, dict) and is_pyg_like_dict(raw) and not is_llaga_processed(raw):
        return dict_to_pyg_compat(raw, dataset_name), True
    return raw, False


def nd_structure_embedding_dim(use_hop: int, sample_neighbor_size: int) -> int:
    s = sample_neighbor_size
    if s <= 1:
        raise ValueError("ND template requires sample_neighbor_size > 1 when synthesizing structure.")
    return int((s ** (use_hop + 1) - 1) / (s - 1))


def load_nd_structure_or_zeros(
    laplacian_path: str,
    use_hop: int,
    sample_neighbor_size: int,
) -> torch.Tensor:
    """Same shape as build_laplacian_emb: (n_pos, n_pos) for subgraph sequence length."""
    if os.path.isfile(laplacian_path):
        return load_dataset_pt(laplacian_path)
    n_pos = nd_structure_embedding_dim(use_hop, sample_neighbor_size)
    return torch.zeros(n_pos, n_pos)


def build_cora_nc_user_prompt(center_text: str = "") -> str:
    classes = ", ".join(CORA_LABEL_TEXTS)
    return (
        f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent papers "
        f"and edges represent co-citations, the node feature of center node is {center_text}. "
        f"We need to classify the center node into 7 classes: {classes}, please tell me which class "
        f"the center node belongs to? Direct tell me the class name."
    )


def build_generic_nc_user_prompt(label_texts: List[str], center_text: str = "") -> str:
    classes = ", ".join(label_texts)
    n = len(label_texts)
    return (
        f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, we need to classify the center node "
        f"into {n} classes: {classes}, please tell me which class the center node belongs to? "
        f"Direct tell me the class name."
    )


def synthesize_nc_samples_from_pyg(
    data: PyGCompatData,
    dataset: str,
    use_hop: int,
    sample_neighbor_size: int,
    center_text: str = "",
) -> list:
    """Build NC samples like jsonl for train_mask nodes (graph node sequence)."""
    train_idx = torch.nonzero(data.train_mask, as_tuple=False).view(-1).tolist()
    edge_list = generate_edge_list(data)
    rows = []
    if dataset == "cora":
        human = build_cora_nc_user_prompt(center_text)
    else:
        human = build_generic_nc_user_prompt(data.label_texts, center_text)
    for idx in train_idx:
        seq = get_fix_shape_subgraph_sequence_fast(
            edge_list, idx, use_hop, sample_neighbor_size
        )
        lab = data.label_texts[int(data.y[idx].item())]
        rows.append(
            {
                "graph": seq,
                "conversations": [
                    {"from": "human", "value": human},
                    {"from": "gpt", "value": lab},
                ],
            }
        )
    return rows
