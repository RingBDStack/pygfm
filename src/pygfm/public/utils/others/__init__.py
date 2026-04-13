"""Misc: graph utils, paths/pickle, distributed, dict2xml, training helpers (seeds); re-exports from data_process / loss_func."""

from __future__ import annotations

from .dict2xml import Converter, Node, dict2xml
from .distributed_compat import (
    get_rank,
    get_world_size,
    master_process_only,
    process_on_master_and_sync_by_pickle,
    synchronize,
)
from .graph_utils import (
    edge_set_to_inds,
    get_edge_set,
    get_neighbors_within_k_hop,
    get_pairwise_topk_sim_mat_chunkdot,
    get_pairwise_topk_sim_mat_scipy,
    get_propagated_feature,
    get_spd_by_sp_matrix,
    get_spd_matrices,
    get_sparse_numpy_adj,
    k_hop_nb_graph,
    sample_nodes,
)
from .ppr import (
    calc_approximate_ppr_rank,
    calc_ppr,
    calc_ppr_topk_parallel,
    construct_sparse,
    find_top_k_neighbors_within_khop_ego_subgraph,
    find_top_k_neighbors_within_khop_ego_subgraph_iter,
    get_row_rank_from_sparse_matrix,
    ppr_topk,
    ppr_topk_batch,
    topk_approximate_ppr_matrix,
)
from .runtime import init_path, init_random_state, pickle_load, pickle_save, time_logger
from ..data_process import BertTextEncoder, encode_texts_with_bert
from ..loss_func import sample_negative_pairs

__all__ = [
    "BertTextEncoder",
    "Converter",
    "Node",
    "calc_approximate_ppr_rank",
    "calc_ppr",
    "calc_ppr_topk_parallel",
    "construct_sparse",
    "dict2xml",
    "edge_set_to_inds",
    "encode_texts_with_bert",
    "find_top_k_neighbors_within_khop_ego_subgraph",
    "find_top_k_neighbors_within_khop_ego_subgraph_iter",
    "get_edge_set",
    "get_neighbors_within_k_hop",
    "get_pairwise_topk_sim_mat_chunkdot",
    "get_pairwise_topk_sim_mat_scipy",
    "get_propagated_feature",
    "get_rank",
    "get_row_rank_from_sparse_matrix",
    "get_spd_by_sp_matrix",
    "get_spd_matrices",
    "get_sparse_numpy_adj",
    "get_world_size",
    "init_path",
    "init_random_state",
    "k_hop_nb_graph",
    "master_process_only",
    "pickle_load",
    "pickle_save",
    "ppr_topk",
    "ppr_topk_batch",
    "process_on_master_and_sync_by_pickle",
    "sample_negative_pairs",
    "sample_nodes",
    "synchronize",
    "time_logger",
    "topk_approximate_ppr_matrix",
]
