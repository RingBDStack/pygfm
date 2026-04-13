"""
BRIDGE DownPrompt: load frozen PrePrompt backbone + MoE mask assembly + prototype matching + spectral regularizer + routing entropy.
For few-shot node classification; pairs with scripts/bridge/finetune.py.
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

from pygfm.public.model_bases import GFMDownPromptNodeModelBase
from pygfm.private.core.gnn_encoder import GNNBackboneEncoder


class BridgeDownPromptModel(GFMDownPromptNodeModelBase):
    """
    Freeze masks_logits / input_proj / backbone (from BridgePrePromptModel);
    train routing_net, graph_prompt, prototypes.
    """

    gfm_family: ClassVar[str] = "bridge"

    def __init__(
        self,
        aligned_dim: int,
        hidden_dim: int,
        num_sources: int,
        num_classes: int,
        domain_name: str,
        dropout: float = 0.0,
        prototype_scale: float = 15.0,
        device: torch.device | None = None,
    ):
        super().__init__(device=device)
        self.aligned_dim = aligned_dim
        self.hidden_dim = hidden_dim
        self.num_sources = num_sources
        self.domain_name = domain_name
        self.prototype_scale = prototype_scale

        self.masks_logits = nn.Parameter(torch.randn(num_sources, aligned_dim))
        self.routing_net = nn.Sequential(
            nn.Linear(aligned_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_sources),
            nn.Softmax(dim=-1),
        )
        self.graph_prompt = nn.Parameter(torch.randn(1, aligned_dim))
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
        self.prototypes = nn.Parameter(torch.randn(num_classes, hidden_dim))
        self.to(self.device)

    def load_preprompt_checkpoint(self, ckpt: dict, strict: bool = True) -> None:
        """Load pretrained backbone from pretrain.py checkpoint."""
        sd = ckpt["model"]
        self.load_state_dict(sd, strict=strict)

    def freeze_preprompt_parts(self) -> None:
        self.masks_logits.requires_grad = False
        self.input_proj.requires_grad_(False)
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _prompted_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        graph_repr = x.mean(dim=0, keepdim=True)
        source_weights = self.routing_net(graph_repr)
        expert_masks = torch.sigmoid(self.masks_logits)
        combined = torch.mm(source_weights, expert_masks)
        final_mask = combined * torch.sigmoid(self.graph_prompt)
        return x * final_mask, source_weights

    @staticmethod
    def spectral_loss(
        h: torch.Tensor,
        x_prompted: torch.Tensor,
        eivec: torch.Tensor,
        eival: torch.Tensor,
        nu: float = 0.1,
    ) -> torch.Tensor:
        h_hat = torch.matmul(eivec.t(), h)
        x_hat = torch.matmul(eivec.t(), x_prompted)
        delta_h = torch.norm(h_hat, p=2, dim=-1)
        delta_x = torch.norm(x_hat, p=2, dim=-1)
        reg = F.relu(eival * delta_h - nu * eival * delta_x)
        return reg.mean()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        eivec: torch.Tensor | None = None,
        eival: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :return: h_nodes [N,H], logits [N,C], reg_loss, ent_loss
        """
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        x_p, source_weights = self._prompted_features(x)
        h_nodes = self.backbone(self.input_proj(x_p), edge_index)
        h_norm = F.normalize(h_nodes, p=2, dim=-1)
        p_norm = F.normalize(self.prototypes, p=2, dim=-1)
        logits = torch.mm(h_norm, p_norm.t()) * self.prototype_scale

        reg = torch.tensor(0.0, device=self.device)
        if eivec is not None and eival is not None:
            eivec = eivec.to(self.device)
            eival = eival.to(self.device)
            reg = self.spectral_loss(h_nodes, x_p, eivec, eival)

        ent = -torch.mean(torch.sum(source_weights * torch.log(source_weights + 1e-10), dim=1))
        return h_nodes, logits, reg, ent

    def embed_backbone_unmasked(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Backbone node embeddings for class-mean prototype init on the support set."""
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        return self.backbone(self.input_proj(x), edge_index)
