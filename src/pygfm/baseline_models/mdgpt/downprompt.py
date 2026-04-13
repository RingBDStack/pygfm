"""
MDGPT-aligned DownPrompt model: prefeature (NodeLevelPrompt) + frozen GCN + prototype matching.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ...private.utlis.domain_alignment import NodeLevelPrompt
from pygfm.public.utils.loss_func import TaskHead
from ...public.utils import compute_prototypes


class DownPromptModel(nn.Module):
    """
    DownPrompt-style model for few-shot node classification.
    prefeature(x) -> frozen GCN -> prototypes -> cosine matching.
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
        support_labels: torch.Tensor,
        query_idx: Optional[torch.Tensor] = None,
        train: bool = True,
    ) -> torch.Tensor:
        """
        :param x: [N, D] raw features.
        :param edge_index: [2, E].
        :param support_idx: indices of support nodes.
        :param support_labels: labels of support nodes.
        :param query_idx: indices of query nodes (if None, use all).
        :param train: if True, update prototypes from support.
        :return: logits [num_query, num_classes].
        """
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        support_idx = support_idx.to(self.device)
        support_labels = support_labels.to(self.device)

        x_prompted = self.prefeature(x)
        # GCN is frozen in __init__ (requires_grad=False);
        # gradients still flow into prefeature.
        h = self.gcn(x_prompted, edge_index)

        support_emb = h[support_idx]
        if train:
            # detach() avoids prototype graph entanglement with query path on backward
            self._prototypes.copy_(
                compute_prototypes(support_emb, support_labels, self.num_classes).detach()
            )

        if query_idx is not None:
            query_idx = query_idx.to(self.device)
            query_emb = h[query_idx]
        else:
            query_emb = h

        logits = self.head(query_emb, self._prototypes)
        return logits  # raw logits; cross_entropy applies softmax; argmax at eval is equivalent
