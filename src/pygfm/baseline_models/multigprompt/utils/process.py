from __future__ import annotations

import os

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys
import torch
import torch.nn as nn
from pathlib import Path

from ..paths import get_datasets_root, get_upstream_data_dir

def parse_skipgram(fname):
    with open(fname) as f:
        toks = list(f.read().split())
    nb_nodes = int(toks[0])
    nb_features = int(toks[1])
    ret = np.empty((nb_nodes, nb_features))
    it = 2
    for i in range(nb_nodes):
        cur_nd = int(toks[it]) - 1
        it += 1
        for j in range(nb_features):
            cur_ft = float(toks[it])
            ret[cur_nd][j] = cur_ft
            it += 1
    return ret

# Process a (subset of) a TU dataset into standard form
def process_tu(data, nb_nodes):
    nb_graphs = len(data)
    ft_size = data.num_features

    features = np.zeros((nb_graphs, nb_nodes, ft_size))
    adjacency = np.zeros((nb_graphs, nb_nodes, nb_nodes))
    labels = np.zeros(nb_graphs)
    sizes = np.zeros(nb_graphs, dtype=np.int32)
    masks = np.zeros((nb_graphs, nb_nodes))
       
    for g in range(nb_graphs):
        sizes[g] = data[g].x.shape[0]
        features[g, :sizes[g]] = data[g].x
        labels[g] = data[g].y[0]
        masks[g, :sizes[g]] = 1.0
        e_ind = data[g].edge_index
        coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])), shape=(nb_nodes, nb_nodes))
        adjacency[g] = coo.todense()

    return features, adjacency, labels, sizes, masks

def micro_f1(logits, labels):
    # Compute predictions
    preds = torch.round(nn.Sigmoid()(logits))
    
    # Cast to avoid trouble
    preds = preds.long()
    labels = labels.long()

    # Count true positives, true negatives, false positives, false negatives
    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0

    # Compute micro-f1 score
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def _find_pyg_graph_pt(data_dir: Path, dataset_str: str) -> Path | None:
    """
    Find single-graph ``.pt`` (PyG ``Data`` or ``dict`` with ``x`` / ``edge_index`` / ``y``), before Planetoid ``ind.*``.

    Search order: ``<data_dir>/data.pt``, ``<dataset>.pt`` (case-insensitive),
    ``<datasets/multigprompt>/*.pt`` when graphs are flat under the baseline folder.
    """
    ds_cf = dataset_str.strip().casefold()
    if not ds_cf:
        return None
    candidates = [data_dir / "data.pt", data_dir / f"{dataset_str}.pt"]
    for c in candidates:
        if c.is_file():
            return c
    if data_dir.is_dir():
        for p in data_dir.glob("*.pt"):
            if p.name.endswith("_enhanced_x64.pt") or p.name.endswith("_enhanced_x32.pt"):
                continue
            if p.stem.casefold() == ds_cf:
                return p
    root = get_datasets_root()
    if root.is_dir() and root != data_dir:
        for p in root.glob("*.pt"):
            if p.stem.casefold() == ds_cf:
                return p
    return None


def _torch_load_pt(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _extract_xy_ei(obj):
    """Extract x, edge_index, y and optional masks from ``Data`` / ``dict`` / InMemory ``(data_dict, slices)``."""
    if isinstance(obj, tuple) and len(obj) >= 1 and isinstance(obj[0], dict):
        obj = obj[0]
    if isinstance(obj, dict):
        if "x" not in obj or "edge_index" not in obj:
            raise ValueError(f"Expected dict with x, edge_index; keys={list(obj.keys())[:20]}")
        x = obj["x"]
        edge_index = obj["edge_index"]
        y = obj.get("y")
        train_mask = obj.get("train_mask")
        val_mask = obj.get("val_mask")
        test_mask = obj.get("test_mask")
    else:
        if not hasattr(obj, "x") or not hasattr(obj, "edge_index"):
            raise ValueError(f"Unsupported graph object type: {type(obj)}")
        x = obj.x
        edge_index = obj.edge_index
        y = getattr(obj, "y", None)
        train_mask = getattr(obj, "train_mask", None)
        val_mask = getattr(obj, "val_mask", None)
        test_mask = getattr(obj, "test_mask", None)
    return x, edge_index, y, train_mask, val_mask, test_mask


def _masks_to_idx_lists(train_mask, val_mask, test_mask, num_nodes: int):
    def _as_bool_idx(m):
        if m is None:
            return None
        if isinstance(m, torch.Tensor):
            t = m.view(-1).bool()
            return torch.where(t)[0].cpu().numpy().astype(np.int64).tolist()
        m = np.asarray(m).reshape(-1).astype(bool)
        return np.where(m)[0].astype(np.int64).tolist()

    tr = _as_bool_idx(train_mask)
    va = _as_bool_idx(val_mask)
    te = _as_bool_idx(test_mask)
    if tr is not None and va is not None and te is not None and (len(tr) + len(va) + len(te)) > 0:
        return tr, va, te
    return None


def _random_split_indices(num_nodes: int, seed: int = 39):
    """Use 140/500/rest when size is Cora-like; otherwise split proportionally."""
    rng = np.random.RandomState(seed)
    if num_nodes >= 1500:
        n_train, n_val = 140, 500
    else:
        n_train = max(1, num_nodes // 15)
        n_val = max(1, num_nodes // 5)
    if n_train + n_val >= num_nodes:
        n_train = max(1, num_nodes // 10)
        n_val = max(1, min(num_nodes - n_train - 1, num_nodes // 4))
    perm = rng.permutation(num_nodes)
    idx_train = sorted(perm[:n_train].tolist())
    idx_val = sorted(perm[n_train : n_train + n_val].tolist())
    idx_test = sorted(perm[n_train + n_val :].tolist())
    return idx_train, idx_val, idx_test


def _load_data_from_pyg_pt(pt_path: Path, dataset_str: str):
    """Single-graph ``.pt`` → same return type as ``load_data`` Planetoid branch."""
    raw = _torch_load_pt(pt_path)
    x, edge_index, y, train_mask, val_mask, test_mask = _extract_xy_ei(raw)

    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    else:
        x = x.detach().float().cpu()
    num_nodes = int(x.shape[0])

    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    else:
        edge_index = edge_index.long().cpu()
    ei = edge_index.numpy()
    row, col = ei[0], ei[1]
    data = np.ones(len(row), dtype=np.float32)
    adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    adj = adj.maximum(adj.T).tocsr()

    features = sp.lil_matrix(x.numpy())

    if y is None:
        raise ValueError(f"{pt_path}: missing labels y")
    y_t = y if isinstance(y, torch.Tensor) else torch.as_tensor(y)
    if y_t.dim() == 2 and y_t.size(1) > 1:
        labels_oh = y_t.detach().float().cpu().numpy()
        if labels_oh.shape[0] != num_nodes:
            raise ValueError(
                f"{pt_path}: y one-hot rows {labels_oh.shape[0]} != num_nodes {num_nodes}"
            )
    else:
        y_flat = y_t.view(-1).long().cpu().numpy()
        if y_flat.shape[0] != num_nodes:
            raise ValueError(
                f"{pt_path}: y length {y_flat.shape[0]} != num_nodes {num_nodes}"
            )
        valid = y_flat >= 0
        if not valid.any():
            raise ValueError(f"{pt_path}: no valid labels (y >= 0)")
        num_classes = int(y_flat[valid].max()) + 1
        labels_oh = np.zeros((num_nodes, num_classes), dtype=np.float32)
        labels_oh[np.arange(num_nodes, dtype=np.int64)[valid], y_flat[valid]] = 1.0

    split = _masks_to_idx_lists(train_mask, val_mask, test_mask, num_nodes)
    if split is None:
        idx_train, idx_val, idx_test = _random_split_indices(num_nodes, seed=39)
    else:
        idx_train, idx_val, idx_test = split

    return adj, features, labels_oh, idx_train, idx_val, idx_test


def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    data_dir = Path(get_upstream_data_dir(dataset_str))
    pt = _find_pyg_graph_pt(data_dir, dataset_str)
    if pt is not None:
        return _load_data_from_pyg_pt(pt, dataset_str)

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(data_dir / "ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(data_dir / "ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    return adj, features, labels, idx_train, idx_val, idx_test

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)




