"""
GCoT PrePrompt: GCN only + NodeNodeContrastiveLoss (no prompts).
Reuses GFM: GCNEncoderSparse, NodeNodeContrastiveLoss, sample_negative_pairs.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    _repo_root = Path(__file__).resolve().parents[3]
    _rp = str(_repo_root)
    if _rp not in sys.path:
        sys.path.insert(0, _rp)

from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from pygfm.public.model_bases import GFMPrePromptModelBase
from pygfm.private.core.gnn_encoder import GCNEncoderSparse
from pygfm.private.utlis.loss_calculation import NodeNodeContrastiveLoss


class GCoTPrePromptModel(GFMPrePromptModelBase):
    """
    GCoT PrePrompt: shared GCN -> ELU -> NodeNodeContrastiveLoss (LP). No input prompts.
    """

    gfm_family: ClassVar[str] = "gcot"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        temperature: float = 1.0,
        device: torch.device | None = None,
    ):
        super().__init__(device=device)

        self.gcn = GCNEncoderSparse(
            input_dim=input_dim,
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
        tuples: torch.Tensor,
    ) -> torch.Tensor:
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        tuples = tuples.to(self.device)
        h = self.gcn(x, edge_index)
        h = F.elu(h)
        return self.contrastive(h, tuples)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        return self.gcn(x, edge_index)
