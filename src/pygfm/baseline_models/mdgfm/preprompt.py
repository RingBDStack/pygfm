"""
MDGFM-aligned PrePrompt: per-domain pretext + shared sumtext -> GCN -> ELU -> NodeNodeContrastiveLoss.
Reuses GFM: NodeLevelPrompt, GCNEncoderSparse, NodeNodeContrastiveLoss, sample_negative_pairs.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    _repo_root = Path(__file__).resolve().parents[3]
    _rp = str(_repo_root)
    if _rp not in sys.path:
        sys.path.insert(0, _rp)

from typing import ClassVar, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from pygfm.public.model_bases import GFMPrePromptModelBase
from pygfm.private.utlis.domain_alignment import NodeLevelPrompt
from pygfm.private.core.gnn_encoder import GCNEncoderSparse
from pygfm.private.utlis.loss_calculation import NodeNodeContrastiveLoss


class MDGFMPrePromptModel(GFMPrePromptModelBase):
    """
    MDGFM PrePrompt: per-domain pretext (NodeLevelPrompt) -> ReLU -> shared sumtext (NodeLevelPrompt)
    -> shared GCN -> ELU -> NodeNodeContrastiveLoss.
    """

    gfm_family: ClassVar[str] = "mdgfm"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_domains: int,
        num_layers: int = 3,
        prompt_mode: Literal["add", "mul"] = "mul",
        temperature: float = 1.0,
        device: torch.device | None = None,
    ):
        super().__init__(device=device)
        self.num_domains = num_domains

        self.pretexts = nn.ModuleList(
            [NodeLevelPrompt(input_dim, mode=prompt_mode) for _ in range(num_domains)]
        )
        self.sumtext = NodeLevelPrompt(input_dim, mode=prompt_mode)

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
                x_prompted[mask] = self.pretexts[d](x[mask])
        x_prompted = F.relu(x_prompted)
        x_prompted = self.sumtext(x_prompted)

        h = self.gcn(x_prompted, edge_index)
        h = F.elu(h)
        return self.contrastive(h, tuples)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Embed without prompt (for downstream)."""
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        return self.gcn(x, edge_index)

    def get_weights(self) -> tuple:
        """Return (pretext_weights, sumtext_weight) for DownPrompt."""
        pretext_weights = [m.weight.detach().clone() for m in self.pretexts]
        sumtext_weight = self.sumtext.weight.detach().clone()
        return pretext_weights, sumtext_weight
