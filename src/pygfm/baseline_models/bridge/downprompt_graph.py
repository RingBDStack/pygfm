"""
BRIDGE graph-level DownPrompt: node embeddings on disjoint-union batch, mean-pool per graph batch vector, then prototype classification.
Shares MoE masks and spectral regularizer with the node variant; for multi-graph DataLoader (see scripts/bridge/finetune_graph.py).
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
from torch_geometric.utils import scatter

from pygfm.baseline_models.bridge.downprompt import BridgeDownPromptModel


class BridgeDownPromptGraphModel(BridgeDownPromptModel):
    """Readout for task_type=graph in forward_graph."""

    gfm_stage: ClassVar[str] = "downprompt_graph"

    def forward_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        eivec: torch.Tensor | None = None,
        eival: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :return: h_graph [B,H], logits [B,C], reg_loss, ent_loss
        """
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        batch = batch.to(self.device)

        graph_repr = scatter(x, batch, dim=0, reduce="mean")
        source_weights = self.routing_net(graph_repr)
        expert_masks = torch.sigmoid(self.masks_logits)
        combined = torch.mm(source_weights, expert_masks)
        gp = torch.sigmoid(self.graph_prompt).expand(combined.size(0), -1)
        final_mask = combined * gp
        x_p = x * final_mask[batch]

        h_nodes = self.backbone(self.input_proj(x_p), edge_index)
        h_graph = scatter(h_nodes, batch, dim=0, reduce="mean")

        h_norm = F.normalize(h_graph, p=2, dim=-1)
        p_norm = F.normalize(self.prototypes, p=2, dim=-1)
        logits = torch.mm(h_norm, p_norm.t()) * self.prototype_scale

        reg = torch.tensor(0.0, device=self.device)
        if eivec is not None and eival is not None:
            eivec = eivec.to(self.device)
            eival = eival.to(self.device)
            reg = self.spectral_loss(h_nodes, x_p, eivec, eival)

        ent = -torch.mean(torch.sum(source_weights * torch.log(source_weights + 1e-10), dim=1))
        return h_graph, logits, reg, ent

    def forward_pyg_fewshot(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        support_idx: torch.Tensor,
        support_batch: torch.Tensor,
        query_idx: torch.Tensor,
        query_batch: torch.Tensor,
        eivec: torch.Tensor | None = None,
        eival: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Same subgraph index contract as MDGPT graph DownPrompt: MoE+GCN on the full graph, then mean-pool support/query subgraph nodes.
        :return: logits_support [G_s,C], logits_query [G_q,C], reg_loss, ent_loss
        """
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        support_idx = support_idx.to(self.device)
        support_batch = support_batch.to(self.device)
        query_idx = query_idx.to(self.device)
        query_batch = query_batch.to(self.device)

        x_p, source_weights = self._prompted_features(x)
        h = self.backbone(self.input_proj(x_p), edge_index)

        reg = torch.tensor(0.0, device=self.device)
        if eivec is not None and eival is not None:
            reg = self.spectral_loss(
                h, x_p, eivec.to(self.device), eival.to(self.device)
            )
        ent = -torch.mean(
            torch.sum(source_weights * torch.log(source_weights + 1e-10), dim=1)
        )

        ns = int(support_batch.max().item()) + 1
        nq = int(query_batch.max().item()) + 1
        sg = scatter(h[support_idx], support_batch, dim=0, dim_size=ns, reduce="mean")
        qg = scatter(h[query_idx], query_batch, dim=0, dim_size=nq, reduce="mean")
        pn = F.normalize(self.prototypes, p=2, dim=-1)
        ls = torch.mm(F.normalize(sg, p=2, dim=-1), pn.t()) * self.prototype_scale
        lq = torch.mm(F.normalize(qg, p=2, dim=-1), pn.t()) * self.prototype_scale
        return ls, lq, reg, ent
