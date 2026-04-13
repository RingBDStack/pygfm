from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from pygfm.private.core.gnn_encoder import GCNEncoderSparse
from pygfm.private.utlis.loss_calculation import NodeNodeContrastiveLoss
from pygfm.public.model_bases import GFMPrePromptModelBase

from .prompt_layers import HGPromptEdgeTypePrompt


class HGPromptPrePromptModel(GFMPrePromptModelBase):
    """HGPrompt-inspired hetero preprompt (edge-type aware)."""

    gfm_family: ClassVar[str] = "hgprompt"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_edge_types: int,
        num_layers: int = 3,
        temperature: float = 1.0,
        device: torch.device | None = None,
    ):
        super().__init__(device=device)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hprompt = HGPromptEdgeTypePrompt(hidden_dim, num_edge_types=num_edge_types)
        self.gcn = GCNEncoderSparse(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation="relu",
            dropout=0.1,
            use_batch_norm=True,
        )
        self.contrastive = NodeNodeContrastiveLoss(temperature=temperature)
        self.to(self.device)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        tuples: torch.Tensor,
    ) -> torch.Tensor:
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)
        tuples = tuples.to(self.device)
        x = self.input_proj(x)
        x = self.hprompt(x, edge_index, edge_type)
        h = F.elu(self.gcn(x, edge_index))
        return self.contrastive(h, tuples)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x.to(self.device))
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)
        x = self.hprompt(x, edge_index, edge_type)
        return self.gcn(x, edge_index)

