"""
GraphMoRE graph-level DownPrompt: node-level DownPrompt plus graph pooling for graph classification.

Graph classification:
1. Reuse node-level GraphMoRE (experts + gating + weighted embeddings + concat)
2. GNN classifier yields node embeddings
3. Scatter mean over batch → graph-level vectors
4. Linear head → graph classification logits

Reference: GraphMoRE (AAAI 2025)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .downprompt import GraphMoREDownPromptModel, _build_classifier

try:
    from torch_geometric.utils import scatter
except ImportError:
    from torch_scatter import scatter


# ---------------------------------------------------------------------------
# GraphMoRE graph-level DownPrompt model
# ---------------------------------------------------------------------------

class GraphMoREDownPromptGraphModel(GraphMoREDownPromptModel):
    """
    GraphMoRE graph classification downstream.

    Inherits experts + gating + classifier backbone from GraphMoREDownPromptModel,
    adds graph pooling and a separate graph head.

    vs node-level:
    - classifier outputs node embeddings (dim = hidden_features_cls)
    - graph_head maps pooled graph states to classes
    """

    def __init__(
        self,
        in_features: int,
        embed_features: int,
        init_curvs: list[float] | None = None,
        learnable_curv: bool = True,
        sample_hops: list[int] | None = None,
        hidden_features_expert: int = 64,
        backbone: str = "gcn",
        hidden_features_cls: int = 32,
        num_classes: int = 2,
        n_layers_cls: int = 2,
        n_heads: int = 8,
        drop_edge_cls: float = 0.0,
        drop_feat_cls: float = 0.0,
        coef_dis: float = 1e-4,
        device: torch.device | None = None,
    ):
        super().__init__(
            in_features=in_features,
            embed_features=embed_features,
            init_curvs=init_curvs,
            learnable_curv=learnable_curv,
            sample_hops=sample_hops,
            hidden_features_expert=hidden_features_expert,
            backbone=backbone,
            hidden_features_cls=hidden_features_cls,
            num_classes=hidden_features_cls,
            n_layers_cls=n_layers_cls,
            n_heads=n_heads,
            drop_edge_cls=drop_edge_cls,
            drop_feat_cls=drop_feat_cls,
            coef_dis=coef_dis,
            device=device,
        )
        self.graph_head = nn.Linear(hidden_features_cls, num_classes)
        self.graph_head.to(self.device)

    def forward_graph(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        subgraph_features: list[torch.Tensor],
        subgraph_edge_indices: list[torch.Tensor],
        subgraph_batches: list[torch.Tensor],
        dis_shortest: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Graph classification forward.

        :param features: [N_total, in_features] node features (disjoint union)
        :param edge_index: [2, E_total] edges (disjoint union)
        :param batch: [N_total] graph id per node
        :param subgraph_features: ego subgraph features on the batched graph
        :param subgraph_edge_indices: ego subgraph edges
        :param subgraph_batches: ego subgraph batch vectors
        :param dis_shortest: shortest-path dict for distortion
        :return: (logits [B, num_classes], loss_distortion)
        """
        node_emb, loss_distortion = super().forward(
            features,
            edge_index,
            subgraph_features,
            subgraph_edge_indices,
            subgraph_batches,
            dis_shortest,
        )

        batch = batch.to(self.device)
        num_graphs = int(batch.max().item()) + 1
        graph_emb = scatter(node_emb, batch, dim=0, dim_size=num_graphs, reduce="mean")
        logits = self.graph_head(graph_emb)
        return logits, loss_distortion

    @torch.no_grad()
    def predict_graph(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        subgraph_features: list[torch.Tensor],
        subgraph_edge_indices: list[torch.Tensor],
        subgraph_batches: list[torch.Tensor],
    ) -> torch.Tensor:
        """Graph classification inference."""
        self.eval()
        node_emb = super().predict(
            features, edge_index, subgraph_features, subgraph_edge_indices, subgraph_batches
        )
        batch = batch.to(self.device)
        num_graphs = int(batch.max().item()) + 1
        graph_emb = scatter(node_emb, batch, dim=0, dim_size=num_graphs, reduce="mean")
        return self.graph_head(graph_emb)

    def get_euclidean_params(self) -> list[nn.Parameter]:
        """Include graph_head parameters."""
        return super().get_euclidean_params() + list(self.graph_head.parameters())
