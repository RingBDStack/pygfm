"""
GraphKeeper-style PrePrompt (PyG, runnable subset aligned with NeurIPS'25-style ideas):

- **Link-prediction pretrain** (like upstream `base_training`): dot-product node embeddings + pos/neg edge BCE.
- **Optional** MDGPT-style **NodeNodeContrastiveLoss** (multi-domain; pairs with `sample_negative_pairs`).
- Same shape as MDGPT PrePrompt: per-domain `NodeLevelPrompt` + shared `GCNEncoderSparse`.

Note: full GraphKeeper uses DGL + incremental multi-expert + parseable classifier; this module matches other
gfm_new baselines with **PrePrompt -> DownPrompt(LoRA)** for shared scripts and data paths.
"""
from __future__ import annotations

import math
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
from torch_geometric.utils import negative_sampling

from pygfm.public.model_bases import GFMPrePromptModelBase
from pygfm.private.utlis.domain_alignment import NodeLevelPrompt
from pygfm.private.core.gnn_encoder import GCNEncoderSparse
from pygfm.private.utlis.loss_calculation import NodeNodeContrastiveLoss

def set_incremental_optimizer(model, current_domain_idx, lr):
    """
    Freeze prompt modules for domains before current_domain_idx; train current only.
    """
    # Freeze older-domain prompt layers
    for i, prompt_layer in enumerate(model.pretexts):
        if i < current_domain_idx:
            for param in prompt_layer.parameters():
                param.requires_grad = False
        elif i == current_domain_idx:
            for param in prompt_layer.parameters():
                param.requires_grad = True
        else:
            # Future-domain prompts stay frozen
            for param in prompt_layer.parameters():
                param.requires_grad = False
    
    return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)


class GraphKeeperPrePromptModel(GFMPrePromptModelBase):
    """
    Multi-domain NodeLevelPrompt -> shared GCN -> (link-pred loss + optional contrastive).
    """

    gfm_family: ClassVar[str] = "graphkeeper"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_domains: int,
        num_layers: int = 3,
        prompt_mode: Literal["add", "mul"] = "mul",
        temperature: float = 1.0,
        contrastive_weight: float = 0.0,
        lp_max_edges: int = 8192,
        device: torch.device | None = None,
    ):
        super().__init__(device=device)
        self.num_domains = num_domains
        self.contrastive_weight = contrastive_weight
        self.lp_max_edges = lp_max_edges

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
        self.contrastive = (
            NodeNodeContrastiveLoss(temperature=temperature) if contrastive_weight > 0 else None
        )
        self.to(self.device)

    def link_prediction_loss(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Dot-product link prediction BCE (same idea as GraphKeeper `base_training`, PyG)."""
        device = h.device
        num_nodes = h.size(0)
        row, col = edge_index[0], edge_index[1]
        E = row.size(0)
        if E > self.lp_max_edges:
            perm = torch.randperm(E, device=device)[: self.lp_max_edges]
            row, col = row[perm], col[perm]
        neg = negative_sampling(
            edge_index,
            num_nodes=num_nodes,
            num_neg_samples=row.size(0),
        )
        pos_score = (h[row] * h[col]).sum(dim=-1)
        neg_score = (h[neg[0]] * h[neg[1]]).sum(dim=-1)
        loss_pos = F.binary_cross_entropy_with_logits(
            pos_score, torch.ones_like(pos_score)
        )
        loss_neg = F.binary_cross_entropy_with_logits(
            neg_score, torch.zeros_like(neg_score)
        )
        return 0.5 * (loss_pos + loss_neg)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        tuples: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        batch = batch.to(self.device)

        x_prompted = x.clone()
        for d in range(self.num_domains):
            mask = batch == d
            if mask.any():
                x_prompted[mask] = self.pretexts[d](x[mask])

        h = self.gcn(x_prompted, edge_index)
        loss = self.link_prediction_loss(h, edge_index)

        if self.contrastive_weight > 0 and self.contrastive is not None and tuples is not None:
            tuples = tuples.to(self.device)
            h_c = F.elu(h)
            loss = loss + self.contrastive_weight * self.contrastive(h_c, tuples)

        return loss

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        return self.gcn(x, edge_index)
