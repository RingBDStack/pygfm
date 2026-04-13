from __future__ import annotations

from typing import ClassVar, Optional

import torch
import torch.nn as nn

from pygfm.private.utlis.loss_calculation import TaskHead
from pygfm.public.model_bases import GFMDownPromptNodeModelBase
from pygfm.public.utils import compute_prototypes

from .prompt_layers import HGPromptEdgeTypePrompt


class HGPromptDownPromptModel(GFMDownPromptNodeModelBase):
    gfm_family: ClassVar[str] = "hgprompt"

    def __init__(
        self,
        gcn: nn.Module,
        hidden_dim: int,
        num_classes: int,
        num_edge_types: int,
        device: torch.device | None = None,
    ):
        super().__init__(device=device)
        self.gcn = gcn
        self.gcn.eval()
        for p in self.gcn.parameters():
            p.requires_grad = False
        self.hprompt = HGPromptEdgeTypePrompt(hidden_dim, num_edge_types=num_edge_types)
        self.head = TaskHead(hidden_dim, task_type="matching")
        self.num_classes = num_classes
        self.register_buffer("_prototypes", torch.zeros(num_classes, hidden_dim))
        self.to(self.device)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        support_idx: torch.Tensor,
        support_labels: torch.Tensor,
        query_idx: Optional[torch.Tensor] = None,
        train: bool = True,
    ) -> torch.Tensor:
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)
        support_idx = support_idx.to(self.device)
        support_labels = support_labels.to(self.device)
        h = self.gcn(self.hprompt(x, edge_index, edge_type), edge_index)
        support_emb = h[support_idx]
        if train:
            self._prototypes.copy_(
                compute_prototypes(support_emb, support_labels, self.num_classes).detach()
            )
        query_emb = h[query_idx.to(self.device)] if query_idx is not None else h
        return self.head(query_emb, self._prototypes)

