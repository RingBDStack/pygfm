from __future__ import annotations

from typing import ClassVar, Optional

import torch
import torch.nn as nn

from pygfm.private.utlis.loss_calculation import TaskHead
from pygfm.public.model_bases import GFMDownPromptGraphModelBase
from pygfm.public.utils import compute_prototypes

from .prompt_layers import GraphPromptWeightedSum, NodePromptFeatureWeighted, scatter_mean


class GraphPromptDownPromptGraphModel(GFMDownPromptGraphModelBase):
    gfm_family: ClassVar[str] = "graphprompt"

    def __init__(
        self,
        gcn: nn.Module,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        prompt_mode: str = "mul",
        device: torch.device | None = None,
    ):
        super().__init__(device=device)
        self.gcn = gcn
        self.gcn.eval()
        for p in self.gcn.parameters():
            p.requires_grad = False
        self.prompt_mode = prompt_mode
        self.prefeature = NodePromptFeatureWeighted(input_dim)
        self.readout = GraphPromptWeightedSum(hidden_dim)
        self.head = TaskHead(hidden_dim, task_type="matching")
        self.num_classes = num_classes
        self.register_buffer("_prototypes", torch.zeros(num_classes, hidden_dim))
        self.to(self.device)

    def _apply_prompt(self, x: torch.Tensor) -> torch.Tensor:
        px = self.prefeature(x)
        if self.prompt_mode == "add":
            return x + px
        return px

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
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        support_idx = support_idx.to(self.device)
        support_batch = support_batch.to(self.device)
        support_labels = support_labels.to(self.device)
        h = self.gcn(self._apply_prompt(x), edge_index)
        support_graph = self.readout(h[support_idx], support_batch)
        if train:
            self._prototypes.copy_(
                compute_prototypes(support_graph, support_labels, self.num_classes).detach()
            )
        if query_idx is not None and query_batch is not None:
            qh = h[query_idx.to(self.device)]
            qg = self.readout(qh, query_batch.to(self.device))
        else:
            qg = support_graph
        return self.head(qg, self._prototypes)

