"""
RAG-GFM motif library: train a subgraph encoder + build motif nano-vectordb.

Per dataset:
  1) CSE, Top-K, dual-view subgraph dataset, train MotifContrastiveModel, save encoder.pth + config.pth
  2) Encode all Top-K subgraphs with the trained encoder → motif_vectordb.json

Requires: nano-vectordb, torch, torch_geometric. Optional: pip install -e ".[rag_gfm]"
"""

from __future__ import annotations

import os
import pickle
import hashlib
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

try:
    from torch_geometric.loader import DataLoader as PyGDataLoader
except ImportError:  # PyG < 2.3
    from torch_geometric.data import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes
from tqdm import tqdm

from .corpus_builder import _parse_pt_to_data, _safe_torch_load, resolve_rag_gfm_graph_pt_path


def _edge_index_cache_fingerprint(edge_index: torch.Tensor) -> str:
    """Stable id for cache keys (shape-only keys collided across graphs / bad old caches)."""
    b = edge_index.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(b).hexdigest()[:32]


def _ensure_edge_index_2_by_e(edge_index: torch.Tensor) -> torch.Tensor:
    """
    PyG expects ``edge_index`` of shape ``[2, num_edges]``.

    Some ``.pt`` exports store ``[num_edges, 2]``; ``to_dense_adj`` then builds a tiny matrix
    (e.g. 2×2), walk centralities collapse to one row, and CSE has ``size(0)==1`` while ``x`` has 2708 rows.
    """
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.as_tensor(edge_index)
    ei = edge_index.long().contiguous()
    if ei.dim() != 2:
        raise ValueError(f"edge_index must be 2D, got shape {tuple(ei.shape)}")
    r, c = int(ei.size(0)), int(ei.size(1))
    if r == 2:
        return ei
    if c == 2:
        return ei.t().contiguous()
    raise ValueError(
        f"edge_index must be [2, E] or [E, 2] (PyG uses [2, E]); got shape {(r, c)}"
    )


# ---------------------------------------------------------------------------
# Data loading (same paths as corpus; raw_texts not required)
# ---------------------------------------------------------------------------

def _infer_num_nodes_conservative(data: Data) -> int:
    """Lower bound on node count: never trust a too-small ``data.num_nodes`` alone (bad exports break CSE)."""
    ei = data.edge_index
    n_ei = int(ei.max().item()) + 1 if ei is not None and ei.numel() > 0 else 0
    n_x = int(data.x.size(0)) if getattr(data, "x", None) is not None else 0
    n_stored = int(data.num_nodes) if getattr(data, "num_nodes", None) is not None else 0
    return max(n_ei, n_x, n_stored)


def load_node_data_for_motif(data_root: str, dataset_name: str) -> Optional[Data]:
    """Load graph (x, edge_index required) for Motif CSE / encoding."""
    pt_path = resolve_rag_gfm_graph_pt_path(data_root, dataset_name)
    if pt_path is None:
        return None
    data = _parse_pt_to_data(_safe_torch_load(pt_path))
    if not hasattr(data, "edge_index") or data.edge_index is None:
        raise ValueError(f"Dataset {dataset_name} has no edge_index")
    data.edge_index = _ensure_edge_index_2_by_e(data.edge_index)
    n_ref = _infer_num_nodes_conservative(data)
    if n_ref <= 0:
        raise ValueError(f"Dataset {dataset_name}: cannot infer num_nodes from edge_index / x")
    if int(getattr(data, "num_nodes", 0) or 0) != n_ref or (
        getattr(data, "x", None) is not None and int(data.x.size(0)) != n_ref
    ):
        warnings.warn(
            f"{dataset_name}: aligning num_nodes/features (inferred n={n_ref}; "
            f"stored num_nodes={getattr(data, 'num_nodes', None)}, x.rows="
            f"{data.x.size(0) if getattr(data, 'x', None) is not None else 0}).",
            UserWarning,
        )
    data.num_nodes = n_ref
    if not hasattr(data, "x") or data.x is None:
        data.x = torch.randn(n_ref, 64)
    elif int(data.x.size(0)) != n_ref:
        xd = int(data.x.size(1)) if data.x.dim() == 2 else 64
        if int(data.x.size(0)) == 1 and n_ref > 1:
            data.x = data.x.expand(n_ref, -1).contiguous()
        else:
            data.x = torch.randn(n_ref, xd, device=data.x.device, dtype=data.x.dtype)
    return data


# ---------------------------------------------------------------------------
# CSE: walk centrality + Top-K + CSE features (from centrality_utils)
# ---------------------------------------------------------------------------

def _to_dense_adj_pyg(edge_index, num_nodes, edge_weight=None):
    from torch_geometric.utils import to_dense_adj

    out = to_dense_adj(edge_index, max_num_nodes=num_nodes, edge_attr=edge_weight)
    adj = out[0] if isinstance(out, (list, tuple)) else out
    # PyG >=2.3 returns ``[batch, N, N]`` even for a single graph (often ``batch=1``). Downstream
    # walk / CSE code assumes a 2D adjacency; leaving the batch dim makes ``centralities.size(0)==1``
    # and ``select_top_k_nodes`` uses ``min(1, k)==1`` → one-row encodings.
    if isinstance(adj, torch.Tensor) and adj.dim() == 3:
        if int(adj.size(0)) != 1:
            raise ValueError(
                f"Motif CSE expects a single graph; got batched dense adjacency with batch={int(adj.size(0))}"
            )
        adj = adj.squeeze(0)
    return adj


def compute_walk_based_centrality(
    ksteps: List[int],
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor] = None,
    num_nodes: Optional[int] = None,
) -> torch.Tensor:
    """Random-walk landing probabilities (num_nodes, len(ksteps))."""
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    P = _to_dense_adj_pyg(edge_index, num_nodes, edge_weight)
    deg_inv = P.sum(dim=1).pow(-1.0)
    deg_inv.masked_fill_(deg_inv == float("inf"), 0)
    P = P * deg_inv.unsqueeze(1)
    rws = []
    Pk = P.clone()
    for i, k in enumerate(sorted(ksteps)):
        if i > 0:
            Pk = Pk @ P
        rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1).unsqueeze(1))
    return torch.cat(rws, dim=1)


def select_top_k_nodes(centralities: torch.Tensor, k: int):
    total = centralities.sum(dim=1)
    order = torch.argsort(total, descending=True)
    num_to_select = min(centralities.size(0), k)
    top_k_indices = order[:num_to_select]
    return order, top_k_indices


def _validate_cse_cache_payload(results: dict[str, Any], num_nodes: int, k: int) -> None:
    """
    Ensure a pickled CSE payload matches ``num_nodes`` (rows of encodings / centralities).

    Stale or corrupt caches (e.g. wrong ``cse_encodings`` shape while ``top_k_indices`` is valid)
    caused ``IndexError`` in ``SubgraphViewDataset.__getitem__``.
    """
    if not isinstance(results, dict):
        raise ValueError("CSE cache root must be a dict")
    cse = results.get("cse_encodings")
    if not isinstance(cse, torch.Tensor):
        raise ValueError("CSE cache missing cse_encodings tensor")
    if int(cse.size(0)) != int(num_nodes):
        raise ValueError(
            f"CSE cache cse_encodings rows {int(cse.size(0))} != num_nodes {int(num_nodes)}"
        )
    cent = results.get("centralities")
    if isinstance(cent, torch.Tensor) and int(cent.size(0)) != int(num_nodes):
        raise ValueError(
            f"CSE cache centralities rows {int(cent.size(0))} != num_nodes {int(num_nodes)}"
        )
    if "top_k_indices" not in results:
        raise ValueError("CSE cache missing top_k_indices")
    tki = results["top_k_indices"]
    tk = _normalize_top_k_indices(tki, k, num_nodes)
    results["top_k_indices"] = tk
    if tk.numel() == 0:
        raise ValueError("CSE cache top_k_indices empty after normalize")
    if int(tk.max()) >= int(num_nodes) or int(tk.min()) < 0:
        raise ValueError("CSE cache top_k_indices out of range for num_nodes")


def _normalize_top_k_indices(top_k_indices: torch.Tensor, k: int, num_nodes: int) -> torch.Tensor:
    """
    Ensure a 1-D long tensor of length at most ``min(k, num_nodes)``.

    Older CSE caches (or bad shapes) sometimes stored the **full** ``argsort`` vector
    (length ``num_nodes``); ``SubgraphViewDataset`` then breaks on ``.item()`` per batch index.
    """
    t = torch.as_tensor(top_k_indices, dtype=torch.long).reshape(-1)
    cap = min(int(k), int(num_nodes), int(t.numel()))
    return t[:cap].contiguous()


def extract_cse_encodings(centralities: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    if normalize:
        mean = centralities.mean(dim=0, keepdim=True)
        std = centralities.std(dim=0, keepdim=True, unbiased=False)
        std = torch.where(std < 1e-8, torch.ones_like(std), std)
        return (centralities - mean) / std
    return centralities


def compute_centrality_and_cse(
    edge_index: torch.Tensor,
    num_nodes: Optional[int] = None,
    ksteps: Optional[List[int]] = None,
    k: Optional[int] = None,
    normalize_cse: bool = True,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    edge_index = _ensure_edge_index_2_by_e(edge_index)
    if ksteps is None:
        ksteps = list(range(1, 9))
    if k is None and num_nodes:
        k = max(200, int(0.1 * num_nodes))
    elif k is None:
        k = 200
    # PyG ``maybe_num_nodes(edge_index, num_nodes)`` returns ``num_nodes`` verbatim when set; a wrong
    # stored count (e.g. 1) yields a 1×1 adjacency while ``edge_index`` still references real ids →
    # ``cse_encodings`` with one row and ``top_k_indices`` pointing at 2633, etc.
    n_ei = int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else 0
    if num_nodes is None:
        num_nodes = maybe_num_nodes(edge_index, None)
    else:
        num_nodes = max(int(num_nodes), n_ei)
    if num_nodes is None or int(num_nodes) <= 0:
        num_nodes = n_ei or maybe_num_nodes(edge_index, None) or 1

    if use_cache and cache_dir:
        ei_fp = _edge_index_cache_fingerprint(edge_index)
        key = hashlib.md5(
            (ei_fp + str(edge_index.shape) + str(num_nodes) + str(sorted(ksteps)) + str(k)).encode()
        ).hexdigest()
        path = os.path.join(cache_dir, f"cse_{key}.pkl")
        if os.path.isfile(path):
            try:
                with open(path, "rb") as f:
                    results = pickle.load(f)
                _validate_cse_cache_payload(results, num_nodes, k)
                return results
            except Exception:
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                except OSError:
                    pass
    P = _to_dense_adj_pyg(edge_index, num_nodes, None)
    deg_inv = P.sum(dim=1).pow(-1.0)
    deg_inv.masked_fill_(deg_inv == float("inf"), 0)
    P = P * deg_inv.unsqueeze(1)
    rws = []
    Pk = P.clone()
    for i, step in enumerate(sorted(ksteps)):
        if i > 0:
            Pk = Pk @ P
        rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1).unsqueeze(1))
    centralities = torch.cat(rws, dim=1)
    order, top_k_indices = select_top_k_nodes(centralities, k)
    top_k_indices = _normalize_top_k_indices(top_k_indices, k, num_nodes)
    cse_encodings = extract_cse_encodings(centralities, normalize=normalize_cse)
    results = {
        "centralities": centralities,
        "top_k_indices": top_k_indices,
        "cse_encodings": cse_encodings,
    }
    if use_cache and cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        try:
            with open(path, "wb") as f:
                pickle.dump(results, f)
        except Exception:
            pass
    return results


# ---------------------------------------------------------------------------
# Dual-view subgraph dataset (from SubgraphViewDataset)
# ---------------------------------------------------------------------------

class SubgraphViewDataset(torch.utils.data.Dataset):
    """Dual view: structure = CSE features, semantic = data.x (match semantic_dim)."""

    def __init__(
        self,
        full_graph_data: Data,
        cse_features: torch.Tensor,
        top_k_node_indices: Union[torch.Tensor, List[int]],
        semantic_features: Optional[torch.Tensor] = None,
    ):
        self.full_graph_data = full_graph_data
        self.cse_features = cse_features
        tk = (
            torch.tensor(top_k_node_indices, dtype=torch.long)
            if not isinstance(top_k_node_indices, torch.Tensor)
            else top_k_node_indices
        )
        self.top_k_indices = torch.as_tensor(tk, dtype=torch.long).reshape(-1).contiguous()
        self.semantic_features = semantic_features if semantic_features is not None else full_graph_data.x

    def __len__(self) -> int:
        return len(self.top_k_indices)

    def __getitem__(self, idx: int):
        center = self.top_k_indices[idx].item()
        ei = self.full_graph_data.edge_index
        n_ei = int(ei.max().item()) + 1 if ei.numel() > 0 else 0
        n_bound = max(
            n_ei,
            int(self.cse_features.size(0)),
            int(self.semantic_features.size(0)),
            int(self.full_graph_data.num_nodes or 0),
        )
        nodes, edge_index, _, _ = k_hop_subgraph(
            center, 1, ei, relabel_nodes=True, num_nodes=n_bound,
        )
        struct_x = self.cse_features[nodes]
        sem_x = self.semantic_features[nodes]
        struct_view = Data(x=struct_x, edge_index=edge_index, num_nodes=len(nodes))
        semantic_view = Data(x=sem_x, edge_index=edge_index, num_nodes=len(nodes))
        return struct_view, semantic_view


# ---------------------------------------------------------------------------
# Subgraph encoder + contrastive model (from train_motif_finder)
# ---------------------------------------------------------------------------

class SubgraphEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)


class MotifContrastiveModel(nn.Module):
    def __init__(
        self,
        struct_input_dim: int,
        semantic_input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.struct_encoder = SubgraphEncoder(struct_input_dim, hidden_dim, output_dim)
        self.semantic_encoder = SubgraphEncoder(semantic_input_dim, hidden_dim, output_dim)

    def forward(self, struct_batch: Data, semantic_batch: Data, temperature: float = 0.1) -> torch.Tensor:
        z1 = self.struct_encoder(struct_batch)
        z2 = self.semantic_encoder(semantic_batch)
        z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
        sim = torch.matmul(z1, z2.T) / temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(sim, labels)


def _project_to_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Project node features x [N, D] to target_dim (PCA or pad/slice)."""
    n, d = x.shape
    if d == target_dim:
        return x
    if d > target_dim:
        x_np = x.cpu().numpy()
        mean = x_np.mean(axis=0)
        x_c = x_np - mean
        u, s, vh = np.linalg.svd(x_c, full_matrices=False)
        proj = (x_c @ vh[:target_dim].T).astype(np.float32)
        return torch.from_numpy(proj).to(x.device)
    pad = torch.zeros(n, target_dim - d, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)


# ---------------------------------------------------------------------------
# One dataset: train encoder + build vectordb
# ---------------------------------------------------------------------------

def train_motif_encoder_one(
    dataset_name: str,
    data: Data,
    cse_encodings: torch.Tensor,
    top_k_indices: torch.Tensor,
    semantic_dim: int = 64,
    output_dir: str = "",
    hidden_dim: int = 64,
    output_dim: int = 32,
    batch_size: int = 64,
    lr: float = 1e-4,
    epochs: int = 200,
    device: torch.device = torch.device("cuda"),
    seed: int = 42,
) -> str:
    """Train motif structure encoder for one dataset; save encoder.pth + config.pth under output_dir."""
    torch.manual_seed(seed)
    n_g = _infer_num_nodes_conservative(data)
    if int(cse_encodings.size(0)) != n_g:
        raise RuntimeError(
            f"CSE encodings rows {int(cse_encodings.size(0))} != inferred graph nodes {n_g} "
            f"(dataset {dataset_name}). ``build_motif_lib`` should have recomputed CSE; "
            "report this as a bug if it persists after ``pip install -e .`` from a clean checkout."
        )
    if data.num_nodes is not None and int(data.num_nodes) != n_g:
        data.num_nodes = n_g
    if data.x.shape[1] != semantic_dim:
        data_x = _project_to_dim(data.x, semantic_dim)
    else:
        data_x = data.x
    dataset = SubgraphViewDataset(data, cse_encodings, top_k_indices, semantic_features=data_x)
    loader = PyGDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: (Batch.from_data_list([x[0] for x in b]), Batch.from_data_list([x[1] for x in b])),
    )
    struct_dim = cse_encodings.shape[1]
    model = MotifContrastiveModel(
        struct_input_dim=struct_dim,
        semantic_input_dim=semantic_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best_state, config_save = float("inf"), None, None

    for epoch in range(epochs):
        model.train()
        total, n_b = 0.0, 0
        for struct_batch, sem_batch in loader:
            struct_batch = struct_batch.to(device)
            sem_batch = sem_batch.to(device)
            loss = model(struct_batch, sem_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n_b += 1
        avg = total / max(n_b, 1)
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.cpu().clone() for k, v in model.struct_encoder.state_dict().items()}
            config_save = {"struct_input_dim": struct_dim, "hidden_dim": hidden_dim, "output_dim": output_dim}
    os.makedirs(output_dir, exist_ok=True)
    encoder_path = os.path.join(output_dir, "encoder.pth")
    config_path = os.path.join(output_dir, "config.pth")
    if best_state is not None:
        torch.save(best_state, encoder_path)
    if config_save is not None:
        torch.save(config_save, config_path)
    return encoder_path


def build_motif_vectordb_one(
    dataset_name: str,
    data: Data,
    cse_encodings: torch.Tensor,
    top_k_indices: torch.Tensor,
    motif_lib_path: str,
    device: torch.device = torch.device("cuda"),
) -> str:
    """Encode all subgraphs with existing encoder+config; write motif_vectordb.json."""
    try:
        from nano_vectordb import NanoVectorDB
    except ImportError:
        raise ImportError("Install nano-vectordb: pip install nano-vectordb")
    out_dir = os.path.join(motif_lib_path, dataset_name)
    config_path = os.path.join(out_dir, "config.pth")
    encoder_path = os.path.join(out_dir, "encoder.pth")
    db_path = os.path.join(out_dir, "motif_vectordb.json")
    if not os.path.isfile(config_path) or not os.path.isfile(encoder_path):
        raise FileNotFoundError(f"Train the encoder for this dataset first: {encoder_path} / {config_path}")
    try:
        config = torch.load(config_path, map_location=device, weights_only=False)
    except TypeError:
        config = torch.load(config_path, map_location=device)
    encoder = SubgraphEncoder(
        config["struct_input_dim"],
        config["hidden_dim"],
        config["output_dim"],
    )
    try:
        encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    except TypeError:
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.to(device)
    encoder.eval()
    dataset = SubgraphViewDataset(data, cse_encodings, top_k_indices)
    documents = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f"Encode {dataset_name}"):
            struct_view, _ = dataset[i]
            struct_view.batch = torch.zeros(struct_view.num_nodes, dtype=torch.long, device=device)
            struct_view = struct_view.to(device)
            # nano-vectordb hashes vectors via ``ndarray.tobytes()``; Python ``list`` raises AttributeError.
            emb = encoder(struct_view).detach().cpu().float().numpy().astype(np.float32).reshape(-1)
            documents.append({
                "__id__": str(i),
                "__vector__": emb,
                "metadata": {"domain": dataset_name, "center_node_original_idx": int(top_k_indices[i].item())},
            })
    if not documents:
        raise RuntimeError("No subgraphs to encode")
    dim = int(documents[0]["__vector__"].shape[0])
    if os.path.isfile(db_path):
        os.remove(db_path)
    db = NanoVectorDB(dim, storage_file=db_path)
    db.upsert(documents)
    db.save()
    return db_path


# ---------------------------------------------------------------------------
# Config and entrypoint
# ---------------------------------------------------------------------------

@dataclass
class MotifLibBuilderConfig:
    """Motif library build config"""

    data_root: str = "datasets/rag_gfm"
    dataset_names: List[str] = field(default_factory=lambda: ["Cora", "Citeseer", "Pubmed"])
    motif_lib_path: str = "downstream_data/rag_gfm/motif_lib"
    cache_dir: Optional[str] = None  # CSE cache; default motif_lib_path/cse_cache
    ksteps: List[int] = field(default_factory=lambda: list(range(1, 9)))
    top_k: int = 200
    semantic_dim: int = 64
    hidden_dim: int = 64
    output_dim: int = 32
    batch_size: int = 64
    lr: float = 1e-4
    epochs: int = 200
    device: str = "cuda"
    seed: int = 42
    use_cse_cache: bool = True


def build_motif_lib(config: MotifLibBuilderConfig) -> List[str]:
    """
    For each name in config.dataset_names: train encoder and build motif_vectordb.
    Returns list of written motif_vectordb.json paths.
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    cache_dir = config.cache_dir or os.path.join(config.motif_lib_path, "cse_cache")
    os.makedirs(config.motif_lib_path, exist_ok=True)
    done = []
    for name in config.dataset_names:
        try:
            data = load_node_data_for_motif(config.data_root, name)
        except Exception as e:
            warnings.warn(f"Skipping dataset {name}: {e}", UserWarning)
            continue
        if data is None:
            warnings.warn(f"Dataset not found: {name}", UserWarning)
            continue
        num_nodes = _infer_num_nodes_conservative(data)
        data.num_nodes = num_nodes
        k = max(config.top_k, int(0.1 * num_nodes))
        res = compute_centrality_and_cse(
            data.edge_index,
            num_nodes=num_nodes,
            ksteps=config.ksteps,
            k=k,
            use_cache=config.use_cse_cache,
            cache_dir=cache_dir,
        )
        cse_encodings = res["cse_encodings"]
        top_k_indices = res["top_k_indices"]
        if int(cse_encodings.size(0)) != int(num_nodes):
            warnings.warn(
                f"{name}: CSE encodings rows {int(cse_encodings.size(0))} != num_nodes {int(num_nodes)}; "
                "recomputing without disk cache (stale ``cse_*.pkl`` or mixed package versions).",
                UserWarning,
            )
            res = compute_centrality_and_cse(
                data.edge_index,
                num_nodes=num_nodes,
                ksteps=config.ksteps,
                k=k,
                use_cache=False,
                cache_dir=cache_dir,
            )
            cse_encodings = res["cse_encodings"]
            top_k_indices = res["top_k_indices"]
        if int(cse_encodings.size(0)) != int(num_nodes):
            ei = data.edge_index
            raise RuntimeError(
                f"{name}: CSE encodings rows {int(cse_encodings.size(0))} still != num_nodes {int(num_nodes)} "
                f"after no-cache recompute. edge_index shape={tuple(ei.shape)} "
                f"(expect [2, num_edges]). data_root={config.data_root!r}."
            )
        top_k_indices = _normalize_top_k_indices(top_k_indices, k, num_nodes)
        out_dir = os.path.join(config.motif_lib_path, name)
        train_motif_encoder_one(
            name,
            data,
            cse_encodings,
            top_k_indices,
            semantic_dim=config.semantic_dim,
            output_dir=out_dir,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            batch_size=config.batch_size,
            lr=config.lr,
            epochs=config.epochs,
            device=device,
            seed=config.seed,
        )
        build_motif_vectordb_one(name, data, cse_encodings, top_k_indices, config.motif_lib_path, device)
        done.append(os.path.join(out_dir, "motif_vectordb.json"))
    return done
