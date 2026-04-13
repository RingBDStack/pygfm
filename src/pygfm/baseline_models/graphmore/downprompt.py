"""
GraphMoRE DownPrompt: load pretrained Riemannian experts + gating → expert-weighted
embeddings concat with raw features → GNN classifier for node classification.

Node classification:
1. Load pretrained expert and gating weights from checkpoint
2. Experts encode nodes in tangent space → gating outputs expert weights
3. Element-wise weighted embeddings → concat with raw features [N, in_features + K*embed_dim]
4. GNN classifier (GCN/GAT/SAGE) on concatenated features
5. Jointly optimize classification + distortion (RiemannianAdam on manifold params)

Reference: GraphMoRE (AAAI 2025) — node-level downstream
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import dropout_edge

from .preprompt import RiemannianExperts, TopologyAwareGating


# ---------------------------------------------------------------------------
# GNN classifier (GCN / GAT / SAGE backbones)
# ---------------------------------------------------------------------------

class _GCNBackbone(nn.Module):
    def __init__(self, n_layers: int, in_dim: int, hid_dim: int, out_dim: int,
                 drop_edge: float = 0.0, drop_feat: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_dim, hid_dim))
        for _ in range(n_layers - 2):
            self.layers.append(GCNConv(hid_dim, hid_dim))
        self.layers.append(GCNConv(hid_dim, out_dim))
        self.drop_edge = drop_edge
        self.drop_feat = nn.Dropout(drop_feat)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge, _ = dropout_edge(edge_index, self.drop_edge, training=self.training)
        for layer in self.layers[:-1]:
            x = self.drop_feat(F.relu(layer(x, edge)))
        return self.layers[-1](x, edge)


class _GATBackbone(nn.Module):
    def __init__(self, n_layers: int, in_dim: int, hid_dim: int, out_dim: int,
                 heads: int = 8, drop_edge: float = 0.0, drop_feat: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_dim, hid_dim, heads, dropout=drop_feat, concat=False))
        for _ in range(n_layers - 2):
            self.layers.append(GATConv(hid_dim, hid_dim, heads, dropout=drop_feat, concat=False))
        self.layers.append(GATConv(hid_dim, out_dim, heads, dropout=drop_feat, concat=False))
        self.drop_edge = drop_edge

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge, _ = dropout_edge(edge_index, self.drop_edge, training=self.training)
        for layer in self.layers:
            x = layer(x, edge)
        return x


class _SAGEBackbone(nn.Module):
    def __init__(self, n_layers: int, in_dim: int, hid_dim: int, out_dim: int,
                 drop_edge: float = 0.0, drop_feat: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_dim, hid_dim))
        for _ in range(n_layers - 2):
            self.layers.append(SAGEConv(hid_dim, hid_dim))
        self.layers.append(SAGEConv(hid_dim, out_dim))
        self.drop_edge = drop_edge
        self.drop_feat = nn.Dropout(drop_feat)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge, _ = dropout_edge(edge_index, self.drop_edge, training=self.training)
        for layer in self.layers[:-1]:
            x = self.drop_feat(F.relu(layer(x, edge)))
        return self.layers[-1](x, edge)


def _build_classifier(
    backbone: str,
    n_layers: int,
    in_dim: int,
    hid_dim: int,
    out_dim: int,
    n_heads: int = 8,
    drop_edge: float = 0.0,
    drop_feat: float = 0.0,
) -> nn.Module:
    backbone = backbone.lower()
    if backbone == "gcn":
        return _GCNBackbone(n_layers, in_dim, hid_dim, out_dim, drop_edge, drop_feat)
    elif backbone == "gat":
        return _GATBackbone(n_layers, in_dim, hid_dim, out_dim, n_heads, drop_edge, drop_feat)
    elif backbone == "sage":
        return _SAGEBackbone(n_layers, in_dim, hid_dim, out_dim, drop_edge, drop_feat)
    raise ValueError(f"Unsupported backbone: {backbone!r}. Choose from 'gcn', 'gat', 'sage'.")


# ---------------------------------------------------------------------------
# GraphMoRE DownPrompt node classification model
# ---------------------------------------------------------------------------

class GraphMoREDownPromptModel(nn.Module):
    """
    GraphMoRE node classification downstream.

    Loads pretrained Riemannian experts and gating, adds a GNN classifier.
    Joint finetuning:
    - Expert params → RiemannianAdam (manifold constraints)
    - Gating + classifier → Adam
    """

    def __init__(
        self,
        in_features: int,
        embed_features: int,
        init_curvs: list[float] | None = None,
        learnable_curv: bool = True,
        sample_hops: list[int] | None = None,
        hidden_features_expert: int = 64,
        # GNN classifier hyperparameters
        backbone: str = "gcn",
        hidden_features_cls: int = 32,
        num_classes: int = 7,
        n_layers_cls: int = 2,
        n_heads: int = 8,
        drop_edge_cls: float = 0.0,
        drop_feat_cls: float = 0.0,
        coef_dis: float = 1e-4,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.init_curvs = init_curvs or [-3.0, -1.0, 0.0, 1.0, 3.0]
        self.sample_hops = sample_hops or [2, 3]
        self.embed_features = embed_features
        self.num_experts = len(self.init_curvs)
        self.coef_dis = coef_dis

        self.experts = RiemannianExperts(
            self.init_curvs,
            in_features,
            hidden_features_expert,
            embed_features,
            learnable_curv,
        )
        self.gating = TopologyAwareGating(
            in_features,
            hidden_features_expert,
            embed_features,
            self.num_experts,
            len(self.sample_hops),
        )

        cls_in_dim = in_features + self.num_experts * embed_features
        self.classifier = _build_classifier(
            backbone,
            n_layers_cls,
            cls_in_dim,
            hidden_features_cls,
            num_classes,
            n_heads,
            drop_edge_cls,
            drop_feat_cls,
        )
        self.to(self.device)

    # ---- Weight loading ----

    def load_preprompt_checkpoint(self, ckpt: dict, strict: bool = False) -> None:
        """Load pretrained experts + gating from pretrain.py checkpoint."""
        pretrained_sd = ckpt["model"]
        model_sd = self.state_dict()
        loaded_keys = []
        for k, v in pretrained_sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k] = v
                loaded_keys.append(k)
        self.load_state_dict(model_sd, strict=False)
        return loaded_keys

    def freeze_preprompt_parts(self) -> None:
        """Freeze experts + gating; train classifier only."""
        for p in self.experts.parameters():
            p.requires_grad = False
        for p in self.gating.parameters():
            p.requires_grad = False

    def get_riemannian_params(self) -> list[nn.Parameter]:
        """Parameters on Riemannian manifolds (optimize with RiemannianAdam)."""
        return list(self.experts.parameters())

    def get_euclidean_params(self) -> list[nn.Parameter]:
        """Euclidean params for Adam (gating + classifier)."""
        return list(self.gating.parameters()) + list(self.classifier.parameters())

    # ---- Forward ----

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        subgraph_features: list[torch.Tensor],
        subgraph_edge_indices: list[torch.Tensor],
        subgraph_batches: list[torch.Tensor],
        dis_shortest: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param features: [N, in_features] raw node features
        :param edge_index: [2, E] full-graph edges
        :param subgraph_features: multi-res ego subgraph features
        :param subgraph_edge_indices: multi-res ego subgraph edges
        :param subgraph_batches: multi-res ego subgraph batch vectors
        :param dis_shortest: shortest-path distances per edge (distortion loss)
        :return: (logits [N, num_classes], loss_distortion)
        """
        features = features.to(self.device)
        edge_index = edge_index.to(self.device)

        embeddings = self.experts.encode(features, edge_index)

        if dis_shortest is not None:
            expert_weights, loss_distortion = self.gating(
                subgraph_features,
                subgraph_edge_indices,
                subgraph_batches,
                embeddings,
                dis_shortest,
                self.embed_features,
                edge_index,
            )
        else:
            expert_weights = self.gating(
                subgraph_features, subgraph_edge_indices, subgraph_batches
            )
            loss_distortion = torch.tensor(0.0, device=self.device)

        weighted_emb = embeddings * expert_weights.repeat_interleave(
            self.embed_features, dim=1
        )
        features_concat = torch.cat([features, weighted_emb], dim=-1)
        logits = self.classifier(features_concat, edge_index)
        return logits, loss_distortion

    @torch.no_grad()
    def predict(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        subgraph_features: list[torch.Tensor],
        subgraph_edge_indices: list[torch.Tensor],
        subgraph_batches: list[torch.Tensor],
    ) -> torch.Tensor:
        """Inference without distortion loss."""
        self.eval()
        features = features.to(self.device)
        edge_index = edge_index.to(self.device)
        embeddings = self.experts.encode(features, edge_index)
        expert_weights = self.gating(
            subgraph_features, subgraph_edge_indices, subgraph_batches
        )
        weighted_emb = embeddings * expert_weights.repeat_interleave(
            self.embed_features, dim=1
        )
        features_concat = torch.cat([features, weighted_emb], dim=-1)
        return self.classifier(features_concat, edge_index)
