import numpy as np
import torch as th
import scipy.sparse as sp
from tqdm import tqdm

from .distributed_compat import process_on_master_and_sync_by_pickle
from .runtime import init_random_state, logger, pickle_save, time_logger
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


def _edge_index_and_n(g):
    """Unified (edge_index, num_nodes) from PyGGraph or object with .edge_index / .num_nodes()."""
    ei = g.edge_index
    n = g.num_nodes() if callable(getattr(g, "num_nodes", None)) else getattr(g, "num_nodes", lambda: ei.max().item() + 1)()
    return ei, n


def _to_scipy_adj(g):
    ei, n = _edge_index_and_n(g)
    if th.is_tensor(ei):
        row, col = ei[0].cpu().numpy(), ei[1].cpu().numpy()
    else:
        row, col = ei[0], ei[1]
    return sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(n, n))


def k_hop_nb_graph(g, k):
    """Return graph with edges (u,v) where dist(u,v) <= k, no self-loops. Returns object with .edges() and .num_nodes()."""
    adj = _to_scipy_adj(g).tocsr()
    n = adj.shape[0]
    pow_adj = adj.copy()
    total = adj.copy()
    for _ in range(k - 1):
        pow_adj = pow_adj @ adj
        total = total + pow_adj
    total.setdiag(0)
    total.eliminate_zeros()
    row, col = total.nonzero()
    edge_index = th.from_numpy(np.vstack([row, col]).astype(np.int64))
    try:
        from torch_geometric.data import Data

        from ..data_process.pyg_graph import PyGGraph
        data = Data(edge_index=edge_index, num_nodes=n)
        return PyGGraph(data)
    except Exception:
        class _KHop:
            def __init__(self, ei, n):
                self.edge_index = ei
                self._n = n
            def edges(self):
                return (self.edge_index[0], self.edge_index[1])
            def num_nodes(self):
                return self._n
        return _KHop(edge_index, n)


def sample_nodes(g, subset_nodes, fanout_list, to_numpy=True):
    subset_nodes = th.tensor(subset_nodes).long().view(-1)
    ei, n = _edge_index_and_n(g)
    src, dst = ei[0], ei[1]
    induced_nodes = {0: subset_nodes.clone()}
    init_random_state(0)
    for layer, fanout in enumerate(fanout_list):
        cur = induced_nodes[layer]
        neighbors = []
        for node in cur.tolist():
            mask = src == node
            idx = th.where(mask)[0]
            if idx.numel() == 0:
                continue
            if idx.numel() <= fanout:
                neighbors.append(dst[mask])
            else:
                perm = th.randperm(idx.numel(), device=ei.device)[:fanout]
                neighbors.append(dst[mask][perm])
        if not neighbors:
            induced_nodes[layer + 1] = cur
            break
        next_nodes = th.cat(neighbors).unique()
        induced_nodes[layer + 1] = next_nodes
    sampled_nodes = th.cat(list(induced_nodes.values())).unique()
    if to_numpy:
        sampled_nodes = sampled_nodes.cpu().numpy()
        induced_nodes = {h: v.cpu().numpy() for h, v in induced_nodes.items()}
    return sampled_nodes, induced_nodes


def get_neighbors_within_k_hop(graph, node_id, k, remove_center_node=False):
    ei, n = _edge_index_and_n(graph)
    src, dst = ei[0].cpu().numpy(), ei[1].cpu().numpy()
    seen = {node_id}
    frontier = {node_id}
    for _ in range(k):
        next_f = set()
        for u in frontier:
            for i in np.where(src == u)[0]:
                v = int(dst[i])
                next_f.add(v)
                seen.add(v)
        frontier = next_f
    if remove_center_node:
        seen.discard(node_id)
    return np.array(list(seen))


def get_edge_set(g):
    row, col = g.edges()
    if th.is_tensor(row):
        row, col = row.cpu().numpy(), col.cpu().numpy()
    return set(map(tuple, np.column_stack([row, col]).tolist()))


def edge_set_to_inds(edge_set):
    """ Unpack edge set to row_ids, col_ids"""
    return list(map(list, zip(*edge_set)))


def get_spd_by_sp_matrix(spd_sp_mat, i, j):
    # ! Note that the default value of a sp matrix is always zero
    # ! which is conflict with the self-loop spd (0)
    if i == j:  # Self loop
        return 0
    elif spd_sp_mat[i, j] == 0:  # Out of max hop
        return - 1
    else:
        return spd_sp_mat[i, j]


@time_logger()
@process_on_master_and_sync_by_pickle(cache_kwarg="cache_file")
def get_spd_matrices(g, max_hops, cache_file=None):
    n = g.num_nodes() if callable(getattr(g, "num_nodes", None)) else g.number_of_nodes()
    sp_mat_shape = (n, n)
    residue_mat = sp.csr_matrix(([], ([], [])), shape=sp_mat_shape, dtype=np.int64)

    for hop in tqdm(range(max_hops, 0, -1), 'building SPD matrices'):
        k_g = k_hop_nb_graph(g, hop)
        new_src, new_dst = k_g.edges()
        new_src = new_src.cpu().numpy() if th.is_tensor(new_src) else new_src
        new_dst = new_dst.cpu().numpy() if th.is_tensor(new_dst) else new_dst
        new_indices = np.vstack((new_src, new_dst))
        n_edges = new_src.numel() if th.is_tensor(new_src) else new_src.size
        new_residue = sp.csr_matrix((np.ones(n_edges, dtype=np.int64), new_indices), shape=sp_mat_shape)
        new_residue.data.fill(max_hops + 1 - hop)
        residue_mat = residue_mat.maximum(new_residue)
    spd_mat = residue_mat.copy()
    spd_mat.data = max_hops + 1 - residue_mat.data

    spd_nb_list = defaultdict(list)
    spd_nb_list[0] = [[i] for i in range(n)]
    for row in tqdm(range(spd_mat.shape[0]), 'building SPD neighbors'):
        start_idx = spd_mat.indptr[row]
        end_idx = spd_mat.indptr[row + 1]
        row_cols = spd_mat.indices[start_idx:end_idx]
        row_data = spd_mat.data[start_idx:end_idx]
        row_dict = {k: [] for k in range(1, max_hops + 1)}
        for col, value in zip(row_cols, row_data):
            row_dict[value].append(col)
        for value, positions in row_dict.items():
            spd_nb_list[value].append(positions)
    pickle_save((spd_mat, spd_nb_list), cache_file)


def get_sparse_numpy_adj(g):
    return _to_scipy_adj(g)


def get_propagated_feature(g, x, k):
    if isinstance(x, th.Tensor):
        x = x.cpu().numpy()
    adj = get_sparse_numpy_adj(g).tocsr()
    for _ in range(1, k + 1):
        x = adj @ x
    return x


@process_on_master_and_sync_by_pickle(cache_kwarg="cache_file")
@time_logger()
def get_pairwise_topk_sim_mat_scipy(x, k=20, cache_file=None):  # Preserve at most 20 neighbors
    # Set diagonal and zero-values to a very negative number
    sim_mat = cosine_similarity(x)
    np.fill_diagonal(sim_mat, -float('inf'))
    # Find the top-k similar graph
    nb_list = []
    for i in tqdm(range(sim_mat.shape[0]), desc=f'Building top-{k} similarity graph'):
        nonzero_indices = np.where(sim_mat[i] > 0)[0]
        nonzero_values = sim_mat[i][nonzero_indices]
        # Sort the non-zero values in descending order and get the top-k
        sorted_nonzero_indices = np.argsort(-nonzero_values)[:k]

        # Map it back to the original indices
        selected = nonzero_indices[sorted_nonzero_indices].tolist()
        nb_list.append(selected)
    pickle_save(nb_list, cache_file)


def _cosine_topk_fallback(x, k, batch_size=500):
    """Fallback when chunkdot is not available (e.g. Python 3.12). Slower, same result."""
    n = x.shape[0]
    nb_list = []
    for start in tqdm(range(0, n, batch_size), desc="cosine top-k fallback"):
        end = min(start + batch_size, n)
        sim_block = cosine_similarity(x[start:end], x)  # (batch_size, n)
        np.fill_diagonal(sim_block, -np.inf)  # exclude self
        for i in range(end - start):
            row = sim_block[i]
            top = np.argsort(-row)[:k]
            nb_list.append(top.tolist())
    return nb_list


@process_on_master_and_sync_by_pickle(cache_kwarg="cache_file")
@time_logger()
def get_pairwise_topk_sim_mat_chunkdot(x, k=20, max_mem_in_gb=5, cache_file=None):  # Preserve at most 20 neighbors
    try:
        from chunkdot import cosine_similarity_top_k
    except ImportError:
        # chunkdot lacks Python 3.12+ support; fall back to sklearn (slower, same result)
        logger.warning("chunkdot not installed (e.g. Python 3.12); using sklearn fallback for cosine top-k.")
        nb_list = _cosine_topk_fallback(np.asarray(x), k=k)
        pickle_save(nb_list, cache_file)
        return

    # Set diagonal and zero-values to a very negative number
    sim_mat = cosine_similarity_top_k(x, top_k=k + 1, max_memory=max_mem_in_gb * 1e9, show_progress=True)
    sim_mat.setdiag(-float('inf'))
    nb_list = []
    for row in tqdm(range(sim_mat.shape[0]), f'building similarity to {cache_file}'):
        start_idx = sim_mat.indptr[row]
        end_idx = sim_mat.indptr[row + 1]

        row_cols = sim_mat.indices[start_idx:end_idx]
        row_data = sim_mat.data[start_idx:end_idx]

        # Sort the non-zero values in descending order and get the top-k
        sorted_nonzero_indices = np.argsort(-row_data)[:k + 1].tolist()
        # Map it back to the original indices
        selected = row_cols[sorted_nonzero_indices[:k]]
        nb_list.append(selected.tolist())

    pickle_save(nb_list, cache_file)
