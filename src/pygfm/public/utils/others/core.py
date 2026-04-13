from __future__ import annotations

import torch
import random
import numpy as np
import os
from typing import Tuple
from torch_geometric.datasets import Planetoid, Amazon
import torch_geometric.transforms as T
from torch_geometric.utils import dropout_edge

def set_seed(seed=42):
    """Strictly maintain seed configuration."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def fast_aug(x, edge_index, drop_feat=0.2, drop_edge=0.2):
    """Original feature masking and edge dropping logic."""
    x_aug = x.clone()
    mask = torch.rand(x.size(1), device=x.device) < drop_feat
    x_aug[:, mask] = 0
    edge_index_aug, _ = dropout_edge(edge_index, p=drop_edge, training=True)
    return x_aug, edge_index_aug

def get_few_shot_split(labels, n_shot=5):
    """Original stratified few-shot split logic."""
    train_indices, test_indices = [], []
    labels_np = labels.cpu().numpy()
    num_classes = int(labels_np.max()) + 1
    for c in range(num_classes):
        idx = np.where(labels_np == c)[0]
        np.random.shuffle(idx)
        if len(idx) >= n_shot:
            train_indices.extend(idx[:n_shot])
            test_indices.extend(idx[n_shot:])
        else:
            train_indices.extend(idx)
    return torch.tensor(train_indices, dtype=torch.long), torch.tensor(test_indices, dtype=torch.long)

def load_all_datasets(data_root="../datasets"):
    """Original dataset loader with specific transforms."""
    datasets = []
    transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])
    # Planetoid datasets
    for name in ['Cora', 'Citeseer', 'Pubmed']:
        datasets.append({"name": name, "ds": Planetoid(root=data_root, name=name, transform=transform)})
    # Amazon datasets
    for name in ['Photo', 'Computers']:
        datasets.append({"name": name, "ds": Amazon(root=data_root, name=name, transform=transform)})
    return datasets


def early_stopping(
    loss: float,
    best: float,
    cnt_wait: int,
    patience: int,
) -> Tuple[bool, float, int]:
    """
    Early stopping helper. Returns (should_stop, new_best, new_cnt_wait).
    """
    if loss < best:
        return False, loss, 0
    cnt_wait += 1
    return cnt_wait >= patience, best, cnt_wait


def compute_prototypes(embeddings: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute per-class mean embeddings (prototypes). [N,D] + [N] -> [num_classes, D]."""
    device = embeddings.device
    prototypes = torch.zeros(num_classes, embeddings.size(1), device=device, dtype=embeddings.dtype)
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            prototypes[c] = embeddings[mask].mean(dim=0)
    return prototypes