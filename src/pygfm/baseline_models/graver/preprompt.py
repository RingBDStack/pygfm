"""
GRAVER PrePrompt: per-source learnable feature masks + DisenGCN encoder + link-level contrastive loss.

Pretraining uses Disentangled GCN (DisenGCN) as the backbone: apply a sigmoid mask per source,
encode, then compute InfoNCE on the concatenated global embeddings.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DisenGCN building blocks
# ---------------------------------------------------------------------------

class InitDisenLayer(nn.Module):
    """Linear map to K-factor disentangled representation [N, K, D/K]."""

    def __init__(self, inp_dim: int, hid_dim: int, num_factors: int):
        super().__init__()
        self.num_factors = num_factors
        self.hid_dim = (hid_dim // num_factors) * num_factors
        self.factor_lins = nn.Linear(inp_dim, self.hid_dim)
        nn.init.xavier_uniform_(self.factor_lins.weight)
        nn.init.zeros_(self.factor_lins.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.factor_lins(x).view(-1, self.num_factors, self.hid_dim // self.num_factors)
        return F.normalize(F.relu(z), dim=2)


class RoutingLayer(nn.Module):
    """Iterative capsule-style routing: aggregate neighbors on edges with per-factor weights."""

    def __init__(self, num_factors: int, routit: int, tau: float):
        super().__init__()
        self.routit = routit
        self.tau = tau

    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """
        :param x: [N, K, D_factor] factorized node states
        :param edges: coalesced sparse COO adjacency [N, N]
        """
        edges = edges.coalesce()
        src = edges.indices()[0]
        trg = edges.indices()[1]
        z = x
        c = x
        for _ in range(self.routit):
            p = (z[trg] * c[src]).sum(dim=2, keepdim=True)
            p = F.softmax(p / self.tau, dim=1)
            weight_sum = p * z[trg]
            c = z + torch.zeros_like(z).index_add_(0, src, weight_sum)
            c = F.normalize(c, dim=2)
        return c


class DisenGCN(nn.Module):
    """
    Disentangled graph convolutional network (DisenGCN).

    Split node features into K independent factor channels, then aggregate neighbors via iterative routing.
    Output dim = (hid_dim // init_k) * init_k.
    """

    def __init__(
        self,
        inp_dim: int,
        hid_dim: int,
        init_k: int = 2,
        delta_k: int = 0,
        routit: int = 1,
        tau: float = 1.0,
        dropout: float = 0.2,
        num_layers: int = 1,
    ):
        super().__init__()
        self.init_disen = InitDisenLayer(inp_dim, hid_dim, init_k)
        self.conv_layers = nn.ModuleList()
        k = init_k
        for _ in range(num_layers):
            self.conv_layers.append(RoutingLayer(k, routit, tau))
            k = max(1, k - delta_k)
        self.dropout = dropout

    @property
    def output_dim(self) -> int:
        return self.init_disen.hid_dim

    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        from .graph import as_sparse_adj

        adj = as_sparse_adj(edges, x.size(0))
        z = self.init_disen(x)
        for conv in self.conv_layers:
            z = conv(z, adj)
            z = F.dropout(F.relu(z), p=self.dropout, training=self.training)
        return z.reshape(z.size(0), -1)


# ---------------------------------------------------------------------------
# GRAVER PrePrompt model
# ---------------------------------------------------------------------------

class GRAVERPrePromptModel(nn.Module):
    """
    GRAVER pretraining model.

    Steps:
    1. Per source, mask raw features with sigmoid(masks_logits[i])
    2. DisenGCN encode + ELU
    3. Concat embeddings from all sources
    4. InfoNCE on global positive/negative link tuples
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_sources: int,
        init_k: int = 2,
        delta_k: int = 0,
        routit: int = 1,
        tau: float = 1.0,
        dropout: float = 0.2,
        num_layers: int = 1,
        temperature: float = 1.0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_sources = num_sources
        self.temperature = temperature
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.masks_logits = nn.Parameter(torch.randn(num_sources, input_dim))
        self.disen_gcn = DisenGCN(
            inp_dim=input_dim,
            hid_dim=hidden_dim,
            init_k=init_k,
            delta_k=delta_k,
            routit=routit,
            tau=tau,
            dropout=dropout,
            num_layers=num_layers,
        )
        self.to(self.device)

    # ---- Contrastive loss ----

    @staticmethod
    def _gather(feature: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """index [A,B] → output [A,B,D]"""
        idx_flat = index.flatten().unsqueeze(1).expand(-1, feature.size(1))
        return torch.gather(feature, 0, idx_flat).reshape(
            index.size(0), index.size(1), feature.size(1)
        )

    def _compare_loss(self, feature: torch.Tensor, tuples: torch.Tensor) -> torch.Tensor:
        """
        Cosine InfoNCE: -log(exp(pos/T) / sum(exp(neg/T))).
        Each row of tuples: [positive_idx, neg_1, neg_2, ...]
        """
        h_tuples = self._gather(feature, tuples)
        anchors = torch.arange(tuples.size(0), device=feature.device)
        anchors = anchors.unsqueeze(1).expand_as(tuples)
        h_anchors = self._gather(feature, anchors)

        sim = F.cosine_similarity(h_anchors, h_tuples, dim=2)
        exp = torch.exp(sim / self.temperature)
        numerator = exp[:, 0:1]
        denominator = exp[:, 1:].sum(dim=1, keepdim=True)
        loss = -torch.log(numerator / (denominator + 1e-8) + 1e-8)
        return loss.mean()

    # ---- Forward ----

    def forward(
        self,
        features_list: list[torch.Tensor],
        edge_index_list: list[torch.Tensor],
        negative_samples: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param features_list: per-source features [N_i, input_dim]
        :param edge_index_list: per-source edges [2, E_i]
        :param negative_samples: pos/neg tuple indices on concatenated graph [N_total, 1+K]
        :return: scalar contrastive loss
        """
        mask_prob = torch.sigmoid(self.masks_logits)
        embeddings = []
        for i, (feat, ei) in enumerate(zip(features_list, edge_index_list)):
            feat = feat.to(self.device)
            ei = ei.to(self.device)
            masked = feat * mask_prob[i].unsqueeze(0)
            h = F.elu(self.disen_gcn(masked, ei))
            embeddings.append(h)
        all_h = torch.cat(embeddings, dim=0)
        negative_samples = negative_samples.to(self.device)
        return self._compare_loss(all_h, negative_samples)

    # ---- Inference ----

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Node embeddings without mask + graph-level mean readout."""
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        h = self.disen_gcn(x, edge_index)
        return h.detach(), h.mean(dim=0, keepdim=True).detach()
