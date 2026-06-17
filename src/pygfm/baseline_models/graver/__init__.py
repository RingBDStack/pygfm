"""
GRAVER baseline: DisenGCN + per-source mask pretrain + generative graphon vocabulary + MoE/CoE routing for downstream classification.
Layout: preprompt.py / downprompt.py / downprompt_graph.py.
"""
from .preprompt import GRAVERPrePromptModel
from .downprompt import GRAVERDownPromptModel  # noqa: F401 — for_finetune on class
from .downprompt_graph import GRAVERDownPromptGraphModel
from .graph import edge_index_to_sparse_adj, inject_graphs_to_target
from .graphon import estimate_graphon
from .io import (
    load_cross_dataset_graphons,
    load_graver_node_features,
    load_graver_preprompt_checkpoint,
    make_forward_rng,
    set_seed_deterministic,
)

__all__ = [
    "edge_index_to_sparse_adj",
    "estimate_graphon",
    "GRAVERPrePromptModel",
    "GRAVERDownPromptModel",
    "GRAVERDownPromptGraphModel",
    "inject_graphs_to_target",
    "load_cross_dataset_graphons",
    "load_graver_node_features",
    "load_graver_preprompt_checkpoint",
    "make_forward_rng",
    "set_seed_deterministic",
]
