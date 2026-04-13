"""
BRIDGE PrePrompt: per-domain learnable feature mask + shared GCN + contrastive loss + mask-dependent noise variance regularizer.
Matches BRIDGE pretrain (per-domain subgraphs on leave-one-out source concat).
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
from pygfm.private.core.gnn_encoder import GNNBackboneEncoder


class BridgePrePromptModel(GFMPrePromptModelBase):
    """
    BRIDGE pretrain model (no MoE routing / graph_prompt / downstream prototypes).
    Forward: GCN on domain-masked features; contrastive loss + variance_weight * variance term.
    """

    gfm_family: ClassVar[str] = "bridge"

    def __init__(
        self,
        aligned_dim: int,
        hidden_dim: int,
        num_sources: int,
        dropout: float = 0.0,
        variance_weight: float = 0.1,
        n_samples: int = 3,
        device: torch.device | None = None,
    ):
        super().__init__(device=device)
        self.aligned_dim = aligned_dim
        self.hidden_dim = hidden_dim
        self.num_sources = num_sources
        self.variance_weight = variance_weight
        self.n_samples = n_samples

        self.masks_logits = nn.Parameter(torch.randn(num_sources, aligned_dim))
        self.input_proj = nn.Linear(aligned_dim, hidden_dim)
        self.backbone = GNNBackboneEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=3,
            gnn_type="gcn",
            dropout=dropout,
            use_batch_norm=True,
            trainable=True,
        ).model
        self.to(self.device)

    def _compare_loss(self, feature: torch.Tensor, tuples: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        feature = F.normalize(feature, p=2, dim=-1)
        h_all = feature[tuples]
        anchor = feature.unsqueeze(1)
        logits = torch.sum(anchor * h_all, dim=-1) / temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=feature.device)
        return F.cross_entropy(logits, labels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        domain_id: int,
        negative_samples: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: [N, aligned_dim]
        :param edge_index: [2, E] subgraph for current domain
        :param domain_id: 0 .. num_sources-1
        :param negative_samples: [N, 1+K] node indices (self in column 0)
        :return: scalar loss
        """
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        negative_samples = negative_samples.to(self.device)

        mask_prob = torch.sigmoid(self.masks_logits[domain_id]).unsqueeze(0)
        x_masked = x * mask_prob
        h_orig = self.backbone(self.input_proj(x_masked), edge_index)
        lploss = self._compare_loss(h_orig, negative_samples)

        if self.n_samples > 0:
            losses = []
            for _ in range(self.n_samples):
                noise = torch.randn_like(x) * (1 - mask_prob.detach())
                x_noisy = x_masked + noise
                h_n = self.backbone(self.input_proj(x_noisy), edge_index)
                losses.append(self._compare_loss(h_n, negative_samples))
            stack = torch.stack(losses)
            v_loss = torch.var(stack) if len(losses) > 1 else stack.mean()
        else:
            v_loss = torch.tensor(0.0, device=self.device)

        return lploss + self.variance_weight * v_loss

    def backbone_forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Backbone pass for downstream prototype init: full features, no domain mask."""
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        return self.backbone(self.input_proj(x), edge_index)
