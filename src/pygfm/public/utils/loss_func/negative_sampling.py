"""
``pygfm.public.utils.loss_func``: negative sampling (used with ``loss_support``).

Used by the **RAG-GFM** pretrain path (``scripts/rag_gfm/pretrain.py`` + PrePrompt model)
and shared with MDGPT, so MDGPT PrePrompt uses the same logic.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch


def sample_negative_pairs(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_neg: int = 50,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    **RAG-GFM** pretrain negative sampling: one positive neighbor + K non-neighbor negatives per node.

    Factored from shared RAG-GFM / MDGPT logic (matches MDGPT ``prompt_pretrain_sample``),
    same usage as ``NodeNodeContrastiveLoss`` and ``scripts/rag_gfm/pretrain.py``.
    Prefer importing via ``pygfm.public.utils.loss_func.loss_support``.

    :param edge_index: [2, E] PyG (src, dst).
    :param num_nodes: number of nodes.
    :param num_neg: negative samples per node.
    :param seed: optional RNG seed.
    :return: [N, 1+num_neg] int64; row i is [pos_i, neg_i1, ..., neg_iK]. If no neighbors, positive is self.
    """
    if seed is not None:
        np.random.seed(seed)
    ei = edge_index.cpu().numpy()
    neighbors: List[np.ndarray] = [np.array([], dtype=np.int64) for _ in range(num_nodes)]
    for k in range(ei.shape[1]):
        src, dst = int(ei[0, k]), int(ei[1, k])
        if src < num_nodes and dst < num_nodes:
            neighbors[src] = np.append(neighbors[src], dst)
    whole = np.arange(num_nodes, dtype=np.int64)
    res = np.zeros((num_nodes, 1 + num_neg), dtype=np.int64)
    for i in range(num_nodes):
        nb = np.unique(neighbors[i])
        non_nb = np.setdiff1d(whole, nb)
        np.random.shuffle(nb)
        np.random.shuffle(non_nb)
        if nb.size == 0:
            res[i, 0] = i
        else:
            res[i, 0] = nb[0]
        n_take = min(num_neg, len(non_nb))
        res[i, 1 : 1 + n_take] = non_nb[:n_take]
        if n_take < num_neg:
            res[i, 1 + n_take :] = np.random.choice(num_nodes, num_neg - n_take, replace=True)
    return torch.from_numpy(res).long()


__all__ = ["sample_negative_pairs"]
