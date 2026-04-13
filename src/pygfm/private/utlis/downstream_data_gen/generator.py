"""
Downstream split generator: few-shot splits and subgraph batches.

Reads from ``data_root`` (upstream graphs, default ``datasets/mdgpt``); **writes only to**
``downstream_root`` (default ``downstream_data/mdgpt``). Few-shot artifacts are never written under ``data_root``.

Supports flat ``*.pt`` / ``processed/data.pt`` (same as ``load_all_datasets``) or PyG loaders:
- few-shot splits: ``.pt`` with ``{indices, labels}``
- graph batch splits: ``.pt`` with ``{idx, batch, labels}`` (1-hop/2-hop subgraphs)
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Amazon, Reddit
import torch_geometric.transforms as T


# Align with load_all_datasets; Reddit is downstream-only; pretrain uses the first five names
SUPPORTED_DATASETS = ["Cora", "Citeseer", "Pubmed", "Photo", "Computers", "Reddit"]


# Default downstream root (per baseline, relative to project root)
DEFAULT_DOWNSTREAM_ROOT = "downstream_data/mdgpt"


@dataclass
class DownstreamGeneratorConfig:
    """Configuration for downstream split generation."""

    dataset: str = "Cora"
    data_root: str = "datasets/mdgpt"
    downstream_root: str = DEFAULT_DOWNSTREAM_ROOT
    output_dir: Optional[str] = None
    seed: int = 42
    n_way: int = 0
    k_shot: int = 5
    n_splits: int = 100
    test_reserve: int = 1000
    max_one_hop: int = 10
    max_two_hop: int = 4
    n_jobs: int = 4
    data_path: Optional[str] = None

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = os.path.join(self.downstream_root, self.dataset)


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_data_from_pyg(data_root: str, dataset: str) -> Data:
    """Load from PyG Planetoid/Amazon/Reddit."""
    transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])
    name_lower = dataset.lower()
    if name_lower in ["cora", "citeseer", "pubmed"]:
        ds = Planetoid(root=data_root, name=dataset, transform=transform)
    elif name_lower in ["photo", "computers"]:
        ds = Amazon(root=data_root, name=dataset, transform=transform)
    elif name_lower == "reddit":
        ds = Reddit(root=data_root, transform=transform)
        return ds[0]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Supported: {SUPPORTED_DATASETS}")
    return ds[0]


def _safe_torch_load(path: str):
    """PyTorch 2.6+: use weights_only=False for PyG Data and similar."""
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def _load_data(config: DownstreamGeneratorConfig) -> Data:
    """Load graph: prefer .pt (data_path or data_root/dataset/processed/data.pt), else PyG."""
    def _parse_pt(obj):
        """Parse .pt: Data or PyG InMemory (data_dict, slices)."""
        if isinstance(obj, Data):
            return obj
        if isinstance(obj, tuple) and len(obj) >= 1:
            d = obj[0]
            if isinstance(d, Data):
                return d
            if isinstance(d, dict) and "x" in d and "edge_index" in d and "y" in d:
                return Data(x=d["x"], edge_index=d["edge_index"], y=d["y"])
        raise TypeError(f"Cannot parse .pt: expected Data or (dict, slices), got {type(obj)}")

    # 1. Explicit data_path
    if config.data_path and os.path.isfile(config.data_path):
        return _parse_pt(_safe_torch_load(config.data_path))
    # 2. Local processed/data.pt (avoid PyG download)
    local_pt = os.path.join(config.data_root, config.dataset, "processed", "data.pt")
    if os.path.isfile(local_pt):
        return _parse_pt(_safe_torch_load(local_pt))
    # 3. Same discovery as pretrain/finetune (flat Cora.pt, subdir data.pt, mapping; see load_all_datasets)
    try:
        from pygfm.public.utils.runtime import load_single_graph_dataset_or_reddit

        data, _ = load_single_graph_dataset_or_reddit(config.data_root, config.dataset)
        return data
    except ValueError:
        pass
    # 4. PyG Planetoid/Amazon/Reddit (may download)
    return _load_data_from_pyg(config.data_root, config.dataset)


def _get_label_indices(
    labels: torch.Tensor,
    n_way: int = 0,
    exclude_last_n: int = 0,
) -> tuple[dict, list]:
    """Build label -> node index list."""
    valid_indices = list(range(len(labels) - exclude_last_n))
    label_to_indices = defaultdict(list)
    for idx in valid_indices:
        label = labels[idx].item()
        label_to_indices[label].append(idx)
    unique_labels = sorted(label_to_indices.keys())
    if n_way > 0 and n_way < len(unique_labels):
        selected_labels = unique_labels[:n_way]
        label_to_indices = {k: v for k, v in label_to_indices.items() if k in selected_labels}
    else:
        selected_labels = unique_labels
    return label_to_indices, selected_labels


def _generate_few_shot_split(
    label_to_indices: dict,
    selected_labels: List[int],
    k_shot: int,
) -> tuple[List[int], dict]:
    """Sample one few-shot split."""
    support_indices = []
    class_indices = {}
    for label in selected_labels:
        indices = label_to_indices[label]
        if len(indices) < k_shot:
            selected = indices
        else:
            selected = np.random.choice(indices, k_shot, replace=False).tolist()
        support_indices.extend(selected)
        class_indices[label] = selected
    return support_indices, class_indices


def _build_neighbors(edge_index: torch.Tensor) -> dict:
    """Adjacency lists from edge_index."""
    neighbors = defaultdict(list)
    for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        neighbors[int(src)].append(int(dst))
    return neighbors


def _get_neighbors(
    node_idx: int,
    neighbors: dict,
    max_one_hop: int = 10,
    max_two_hop: int = 4,
) -> List[int]:
    """Center + 1-hop + 2-hop neighbors (sampled)."""
    one_hop = neighbors.get(node_idx, [])
    if len(one_hop) > max_one_hop:
        one_hop = random.sample(one_hop, max_one_hop)

    two_hop = []
    for n in one_hop:
        cand = neighbors.get(n, [])
        cand = [c for c in cand if c != node_idx and c not in one_hop and c not in two_hop]
        if cand:
            size = min(len(cand), max(1, max_two_hop // len(one_hop)))
            two_hop.extend(random.sample(cand, size))
            if len(two_hop) >= max_two_hop:
                two_hop = two_hop[:max_two_hop]
                break
    return [node_idx] + one_hop + two_hop


def build_test_subgraphs(
    edge_index: torch.Tensor,
    test_indices: List[int],
    max_one_hop: int = 10,
    max_two_hop: int = 4,
    seed: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build subgraphs (center + 1-hop + 2-hop) for test nodes (graph-level downstream).

    :param edge_index: [2, E]
    :param test_indices: test center node ids
    :param max_one_hop: max 1-hop neighbors
    :param max_two_hop: max 2-hop neighbors
    :param seed: RNG seed for neighbor sampling
    :return: (testlist, testindex)
        - testlist: flat node indices across subgraphs
        - testindex: graph id per node (0..len(test_indices)-1)
    """
    if seed is not None:
        random.seed(seed)
    neighbors = _build_neighbors(edge_index)
    all_idx, all_batch = [], []
    for bi, center in enumerate(test_indices):
        nodes = _get_neighbors(center, neighbors, max_one_hop, max_two_hop)
        all_idx.extend(nodes)
        all_batch.extend([bi] * len(nodes))
    return (
        torch.tensor(all_idx, dtype=torch.long),
        torch.tensor(all_batch, dtype=torch.long),
    )


def generate_few_shot_splits(
    config: Optional[DownstreamGeneratorConfig] = None,
    dataset: Optional[str] = None,
    data_root: Optional[str] = None,
    downstream_root: Optional[str] = None,
    output_dir: Optional[str] = None,
    k_shot: int = 5,
    n_splits: int = 100,
    n_way: int = 0,
    test_reserve: int = 1000,
    seed: int = 42,
    verbose: bool = True,
) -> str:
    """
    Generate few-shot splits and save one ``.pt`` file.

    Output ``{k}shot/splits.pt``:
        - splits: list of dict with indices, labels
        - meta: dataset, k_shot, n_splits, n_way

    Load: ``data = torch.load(...); split_i = data["splits"][i]``
    """
    cfg = config or DownstreamGeneratorConfig()
    if dataset is not None:
        cfg.dataset = dataset
    if data_root is not None:
        cfg.data_root = data_root
    if downstream_root is not None:
        cfg.downstream_root = downstream_root
    if output_dir is not None:
        cfg.output_dir = output_dir
    elif downstream_root is not None:
        cfg.output_dir = os.path.join(cfg.downstream_root, cfg.dataset)
    cfg.k_shot = k_shot
    cfg.n_splits = n_splits
    cfg.n_way = n_way
    cfg.test_reserve = test_reserve
    cfg.seed = seed

    _set_seed(cfg.seed)
    data = _load_data(cfg)
    labels = data.y

    label_to_indices, selected_labels = _get_label_indices(
        labels, n_way=cfg.n_way, exclude_last_n=cfg.test_reserve
    )

    out_dir = os.path.join(cfg.output_dir, f"{cfg.k_shot}shot")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "splits.pt")

    if verbose:
        print(f"dataset: {cfg.dataset}")
        print(f"num_nodes: {data.x.shape[0]}")
        print(f"train_pool_nodes: {data.x.shape[0] - cfg.test_reserve}")
        print(f"num_classes: {len(selected_labels)}")
        print(f"generating {cfg.n_splits} x {cfg.k_shot}-shot -> {out_file}")

    all_splits = []
    for split_idx in tqdm(range(cfg.n_splits), disable=not verbose):
        support_indices, _ = _generate_few_shot_split(
            label_to_indices, selected_labels, cfg.k_shot
        )
        support_labels = [int(labels[i].item()) for i in support_indices]
        all_splits.append({"indices": support_indices, "labels": support_labels})

    torch.save(
        {
            "splits": all_splits,
            "meta": {
                "dataset": cfg.dataset,
                "k_shot": cfg.k_shot,
                "n_splits": cfg.n_splits,
                "n_way": len(selected_labels),
            },
        },
        out_file,
    )

    if verbose:
        print(f"done -> {out_file}")
    return out_dir


def generate_graph_batch_splits(
    config: Optional[DownstreamGeneratorConfig] = None,
    dataset: Optional[str] = None,
    data_root: Optional[str] = None,
    downstream_root: Optional[str] = None,
    output_dir: Optional[str] = None,
    k_shot: int = 5,
    n_splits: int = 100,
    n_way: int = 0,
    test_reserve: int = 1000,
    max_one_hop: int = 10,
    max_two_hop: int = 4,
    n_jobs: int = 4,
    seed: int = 42,
    verbose: bool = True,
) -> str:
    """
    Few-shot subgraph batches saved to one ``.pt``.

    Each support node is a center; 1-hop/2-hop neighbors form the subgraph.
    Output ``{k}shot_graph_batch/splits.pt``:
        - splits: list of dict with idx, batch, labels
        - meta: dataset, k_shot, n_splits, n_way
    """
    cfg = config or DownstreamGeneratorConfig()
    if dataset is not None:
        cfg.dataset = dataset
    if data_root is not None:
        cfg.data_root = data_root
    if downstream_root is not None:
        cfg.downstream_root = downstream_root
    if output_dir is not None:
        cfg.output_dir = output_dir
    elif downstream_root is not None:
        cfg.output_dir = os.path.join(cfg.downstream_root, cfg.dataset)
    cfg.k_shot = k_shot
    cfg.n_splits = n_splits
    cfg.n_way = n_way
    cfg.test_reserve = test_reserve
    cfg.max_one_hop = max_one_hop
    cfg.max_two_hop = max_two_hop
    cfg.n_jobs = n_jobs
    cfg.seed = seed

    _set_seed(cfg.seed)
    data = _load_data(cfg)
    labels = data.y
    edge_index = data.edge_index

    label_to_indices, selected_labels = _get_label_indices(
        labels, n_way=cfg.n_way, exclude_last_n=cfg.test_reserve
    )
    neighbors = _build_neighbors(edge_index)

    out_dir = os.path.join(cfg.output_dir, f"{cfg.k_shot}shot_graph_batch")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "splits.pt")

    if verbose:
        print(f"dataset: {cfg.dataset}")
        print(f"num_nodes: {data.x.shape[0]}")
        print(f"generating {cfg.n_splits} x {cfg.k_shot}-shot subgraph batches -> {out_file}")

    def _gen_one(split_idx: int):
        support_indices, _ = _generate_few_shot_split(
            label_to_indices, selected_labels, cfg.k_shot
        )
        all_idx, all_batch, center_labels = [], [], []
        for bi, center in enumerate(support_indices):
            nodes = _get_neighbors(center, neighbors, cfg.max_one_hop, cfg.max_two_hop)
            all_idx.extend(nodes)
            all_batch.extend([bi] * len(nodes))
            center_labels.append(int(labels[center].item()))
        return {
            "idx": torch.tensor(all_idx, dtype=torch.long),
            "batch": torch.tensor(all_batch, dtype=torch.long),
            "labels": torch.tensor(center_labels, dtype=torch.long),
        }

    if _HAS_JOBLIB and cfg.n_jobs > 1:
        all_splits = Parallel(n_jobs=cfg.n_jobs)(
            delayed(_gen_one)(i) for i in tqdm(range(cfg.n_splits), disable=not verbose)
        )
    else:
        all_splits = [_gen_one(i) for i in tqdm(range(cfg.n_splits), disable=not verbose)]

    torch.save(
        {
            "splits": all_splits,
            "meta": {
                "dataset": cfg.dataset,
                "k_shot": cfg.k_shot,
                "n_splits": cfg.n_splits,
                "n_way": len(selected_labels),
            },
        },
        out_file,
    )

    if verbose:
        print(f"done -> {out_file}")
    return out_dir
