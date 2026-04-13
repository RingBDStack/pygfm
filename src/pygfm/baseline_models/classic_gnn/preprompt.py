from __future__ import annotations

from typing import ClassVar, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from pygfm.private.core.gnn_encoder import (
    GCNEncoderSparse,
    GATEncoderSparse,
    GraphSAGEEncoderSparse,
)
from pygfm.private.utlis.loss_calculation import NodeNodeContrastiveLoss
from pygfm.public.model_bases import GFMPrePromptModelBase

from ..graphprompt.prompt_layers import NodePromptFeatureWeighted

BackboneName = Literal["gcn", "graphsage", "gat"]


def build_sparse_encoder(
    backbone: BackboneName,
    *,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    activation: str = "relu",
    dropout: float = 0.1,
    use_batch_norm: bool = True,
    gat_heads: int = 4,
) -> nn.Module:
    """Shared encoder factory for classic GNN GFM (GCN / GraphSAGE / GAT)."""
    common = dict(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=activation,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
    )
    if backbone == "gcn":
        return GCNEncoderSparse(**common)
    if backbone == "graphsage":
        return GraphSAGEEncoderSparse(**common)
    if backbone == "gat":
        return GATEncoderSparse(**common, heads=gat_heads)
    raise ValueError(f"Unknown backbone: {backbone}")


class ClassicGNNPrePromptModel(GFMPrePromptModelBase):
    """
    GraphPrompt-style GFM with a swappable backbone: GCN, GraphSAGE, or GAT.
    Per-domain feature prompt + shared encoder + node-node contrastive loss.
    """

    gfm_family: ClassVar[str] = "classic_gnn"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_domains: int,
        backbone: BackboneName = "gcn",
        num_layers: int = 3,
        prompt_mode: Literal["add", "mul"] = "mul",
        temperature: float = 1.0,
        gat_heads: int = 4,
        device: torch.device | None = None,
    ):
        super().__init__(device=device)
        self.backbone = backbone
        self.num_domains = num_domains
        self.prompt_mode = prompt_mode
        self.gat_heads = gat_heads
        self.pretexts = nn.ModuleList([NodePromptFeatureWeighted(input_dim) for _ in range(num_domains)])
        self.encoder = build_sparse_encoder(
            backbone,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation="relu",
            dropout=0.1,
            use_batch_norm=True,
            gat_heads=gat_heads,
        )
        self.contrastive = NodeNodeContrastiveLoss(temperature=temperature)
        self.to(self.device)

    def _apply_prompt(self, x: torch.Tensor, prompt: nn.Module) -> torch.Tensor:
        px = prompt(x)
        if self.prompt_mode == "add":
            return x + px
        return px

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        tuples: torch.Tensor,
    ) -> torch.Tensor:
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        batch = batch.to(self.device)
        tuples = tuples.to(self.device)
        x_prompted = x.clone()
        for d in range(self.num_domains):
            mask = batch == d
            if mask.any():
                x_prompted[mask] = self._apply_prompt(x[mask], self.pretexts[d])
        h = F.elu(self.encoder(x_prompted, edge_index))
        return self.contrastive(h, tuples)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.to(self.device), edge_index.to(self.device))
