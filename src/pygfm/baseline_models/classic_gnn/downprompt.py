from __future__ import annotations

from typing import ClassVar, Optional

import torch
import torch.nn as nn

from pygfm.private.utlis.loss_calculation import TaskHead
from pygfm.public.model_bases import GFMDownPromptNodeModelBase
from pygfm.public.utils import compute_prototypes

from ..graphprompt.prompt_layers import NodePromptFeatureWeighted


class ClassicGNNDownPromptModel(GFMDownPromptNodeModelBase):
    """Few-shot node classification with frozen classic GNN encoder and a learnable feature prompt."""

    gfm_family: ClassVar[str] = "classic_gnn"

    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        prompt_mode: str = "mul",
        device: torch.device | None = None,
    ):
        super().__init__(device=device)
        self.encoder = encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.prompt_mode = prompt_mode
        self.prefeature = NodePromptFeatureWeighted(input_dim)
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
        support_labels: torch.Tensor,
        query_idx: Optional[torch.Tensor] = None,
        train: bool = True,
    ) -> torch.Tensor:
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        support_idx = support_idx.to(self.device)
        support_labels = support_labels.to(self.device)
        h = self.encoder(self._apply_prompt(x), edge_index)
        support_emb = h[support_idx]
        if train:
            self._prototypes.copy_(
                compute_prototypes(support_emb, support_labels, self.num_classes).detach()
            )
        query_emb = h[query_idx.to(self.device)] if query_idx is not None else h
        return self.head(query_emb, self._prototypes)
