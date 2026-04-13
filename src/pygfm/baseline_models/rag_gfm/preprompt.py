"""
RAG-GFM PrePrompt: per-domain NodeLevelPrompt + shared GCN + NodeNodeContrastiveLoss.

Similar to MDGPT pretrain but kept baseline-specific; loss from ``pygfm.public.utils.loss_func``.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...private.utlis.domain_alignment import NodeLevelPrompt
from pygfm.private.core import GCNEncoderSparse
from pygfm.public.utils.loss_func import NodeNodeContrastiveLoss


class PrePromptModel(nn.Module):
    """
    RAG-GFM PrePrompt pretraining model.
    Per-domain NodeLevelPrompt (PCA features) -> shared GCN -> ELU -> NodeNodeContrastiveLoss.
    Structure aligned with model_node_rag PrePrompt; encoder pieces from toolbox GCN/NodeLevelPrompt/Loss.
    """

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
        super().__init__()
        self.num_domains = num_domains
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pretexts = nn.ModuleList(
            [NodeLevelPrompt(input_dim, mode=prompt_mode) for _ in range(num_domains)]
        )
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
        """
        :param x: [N, D] aligned features (e.g. PCA unify_dim).
        :param edge_index: [2, E] merged graph.
        :param batch: [N] domain id per node (0..num_domains-1).
        :param tuples: [N, 1+K] from sample_negative_pairs.
        :return: scalar contrastive loss.
        """
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        batch = batch.to(self.device)
        tuples = tuples.to(self.device)

        x_prompted = x.clone()
        for d in range(self.num_domains):
            mask = batch == d
            if mask.any():
                x_prompted[mask] = self.pretexts[d](x[mask])

        h = self.gcn(x_prompted, edge_index)
        h = F.elu(h)
        return self.contrastive(h, tuples)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Downstream: GCN only, no prompts."""
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        return self.gcn(x, edge_index)
