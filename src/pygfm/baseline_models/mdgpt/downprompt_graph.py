"""
MDGPT-aligned DownPrompt for graph-level few-shot classification.

Support/query are subgraphs (center + 1/2-hop neighbors). Graph embeddings via
scatter_mean over nodes; then prototype matching (same as node DownPrompt).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ...private.utlis.domain_alignment import NodeLevelPrompt
from pygfm.public.utils.loss_func import TaskHead
from ...public.utils import compute_prototypes


def _scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Scatter mean: aggregate src by index. Fallback without torch_scatter.
    :param src: [M, D]
    :param index: [M] integer indices
    :return: [dim_size, D]
    """
    try:
        import torch_scatter
        return torch_scatter.scatter_mean(src, index, dim=0, dim_size=dim_size)
    except ImportError:
        if dim_size is None:
            dim_size = int(index.max().item()) + 1
        out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
        cnt = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
        index_exp = index.unsqueeze(1).expand(-1, src.size(1))
        out.scatter_add_(0, index_exp, src)
        cnt.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))
        return out / cnt.clamp(min=1).unsqueeze(1)


class DownPromptGraphModel(nn.Module):
    """
    DownPrompt for few-shot graph classification.

    Support/query are subgraphs. prefeature(x) -> frozen GCN -> node emb ->
    scatter_mean by batch -> graph emb -> prototypes -> cosine matching.
    """

    def __init__(
        self,
        gcn: nn.Module,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        prompt_mode: str = "mul",
        device: torch.device | None = None,
    ):
        super().__init__()
        self.gcn = gcn
        self.gcn.eval()
        for p in self.gcn.parameters():
            p.requires_grad = False
        self.prefeature = NodeLevelPrompt(input_dim, mode=prompt_mode)
        self.head = TaskHead(hidden_dim, task_type="matching")
        self.num_classes = num_classes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.register_buffer("_prototypes", torch.zeros(num_classes, hidden_dim))
        self.to(self.device)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        support_idx: torch.Tensor,
        support_batch: torch.Tensor,
        support_labels: torch.Tensor,
        query_idx: Optional[torch.Tensor] = None,
        query_batch: Optional[torch.Tensor] = None,
        train: bool = True,
    ) -> torch.Tensor:
        """
        :param x: [N, D] node features.
        :param edge_index: [2, E].
        :param support_idx: [M] flat node indices (all nodes in support subgraphs).
        :param support_batch: [M] graph id per node (0..num_support_graphs-1).
        :param support_labels: [num_support_graphs] label per support graph.
        :param query_idx: [Q] flat node indices for query subgraphs.
        :param query_batch: [Q] graph id per query node.
        :param train: if True, update prototypes from support.
        :return: logits [num_query_graphs, num_classes].
        """
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        support_idx = support_idx.to(self.device)
        support_batch = support_batch.to(self.device)
        support_labels = support_labels.to(self.device)

        x_prompted = self.prefeature(x)
        h = self.gcn(x_prompted, edge_index)

        support_node_emb = h[support_idx]
        support_graph_emb = _scatter_mean(
            support_node_emb,
            support_batch,
            dim_size=int(support_batch.max().item()) + 1,
        )

        if train:
            self._prototypes.copy_(
                compute_prototypes(
                    support_graph_emb, support_labels, self.num_classes
                ).detach()
            )

        if query_idx is not None and query_batch is not None:
            query_idx = query_idx.to(self.device)
            query_batch = query_batch.to(self.device)
            query_node_emb = h[query_idx]
            query_graph_emb = _scatter_mean(
                query_node_emb,
                query_batch,
                dim_size=int(query_batch.max().item()) + 1,
            )
        else:
            query_graph_emb = support_graph_emb

        logits = self.head(query_graph_emb, self._prototypes)
        return logits
