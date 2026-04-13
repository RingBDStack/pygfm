from __future__ import annotations

import torch
import torch.nn as nn


class HGPromptEdgeTypePrompt(nn.Module):
    """
    HGPrompt-style edge-type aware message scaling.
    Works on PyG edge_index / edge_type (no DGL dependency).
    """

    def __init__(self, hidden_dim: int, num_edge_types: int):
        super().__init__()
        self.node_weight = nn.Parameter(torch.empty(1, hidden_dim))
        self.edge_weight = nn.Embedding(num_edge_types, hidden_dim)
        nn.init.xavier_uniform_(self.node_weight)
        nn.init.xavier_uniform_(self.edge_weight.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        row, col = edge_index
        src = x[row] * self.node_weight
        et = self.edge_weight(edge_type)
        msg = src + et
        out = torch.zeros_like(x)
        out.index_add_(0, col, msg)
        return out

