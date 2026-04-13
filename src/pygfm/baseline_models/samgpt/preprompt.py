"""
SAMGPT-aligned PrePrompt: per-domain feature + per-layer structure prompts,
shared GCN (with optional structure prompts), LP contrastive loss.
Reuses GFM: NodeLevelPrompt, GCNEncoderSparseWithPrompts, NodeNodeContrastiveLoss, sample_negative_pairs.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow ``python preprompt.py`` from ``models/samgpt/`` without package context
if __package__ in (None, ""):
    _repo_root = Path(__file__).resolve().parents[3]
    _rp = str(_repo_root)
    if _rp not in sys.path:
        sys.path.insert(0, _rp)

from typing import ClassVar, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from pygfm.public.model_bases import GFMPrePromptModelBase
from pygfm.private.utlis.domain_alignment import NodeLevelPrompt
from pygfm.private.core.gnn_encoder import GCNEncoderSparseWithPrompts
from pygfm.private.utlis.loss_calculation import NodeNodeContrastiveLoss


class SAMGPTPrePromptModel(GFMPrePromptModelBase):
    """
    SAMGPT PrePrompt: feature prompt (per-domain) + structure prompt (per-domain per-layer),
    shared GCN, combine with alpha, ELU, then NodeNodeContrastiveLoss.
    """

    gfm_family: ClassVar[str] = "samgpt"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_domains: int,
        num_layers: int = 3,
        prompt_mode: Literal["add", "mul"] = "mul",
        temperature: float = 1.0,
        alpha: float = 1.0,
        device: torch.device | None = None,
    ):
        super().__init__(device=device)
        self.num_domains = num_domains
        self.num_layers = num_layers
        self.alpha = alpha

        self.feature_prompt_layers = nn.ModuleList(
            [NodeLevelPrompt(input_dim, mode=prompt_mode) for _ in range(num_domains)]
        )
        self.structure_prompt_layers = nn.ModuleList([
            nn.ModuleList([
                NodeLevelPrompt(hidden_dim, mode=prompt_mode) for _ in range(num_layers)
            ])
            for _ in range(num_domains)
        ])

        self.gcn = GCNEncoderSparseWithPrompts(
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

        # Feature branch: per-domain prompt on input -> GCN (no structure prompt)
        x_fea = x.clone()
        for d in range(self.num_domains):
            mask = batch == d
            if mask.any():
                x_fea[mask] = self.feature_prompt_layers[d](x[mask])
        h_fea = self.gcn(x_fea, edge_index, batch=None, structure_prompt_layers_per_domain=None, structure_prompt_layers=None)
        h_fea = F.elu(h_fea)

        # Structure branch: raw x -> GCN with per-domain per-layer structure prompts
        str_layers_per_domain = [
            list(self.structure_prompt_layers[d]) for d in range(self.num_domains)
        ]
        h_str = self.gcn(
            x,
            edge_index,
            batch=batch,
            structure_prompt_layers_per_domain=str_layers_per_domain,
            structure_prompt_layers=None,
        )
        h_str = F.elu(h_str)

        # Combine
        h = h_fea + self.alpha * h_str
        return self.contrastive(h, tuples)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Embed without any prompt (for downstream)."""
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        return self.gcn(x, edge_index, batch=None, structure_prompt_layers_per_domain=None, structure_prompt_layers=None)

    def get_weights(self) -> tuple:
        """Return (fea_weights, str_weights, [alpha]) for DownPrompt composed token."""
        fea_weights = [m.weight.detach().clone() for m in self.feature_prompt_layers]
        str_weights = [
            [m.weight.detach().clone() for m in layer_list]
            for layer_list in self.structure_prompt_layers
        ]
        return fea_weights, str_weights, [self.alpha]
