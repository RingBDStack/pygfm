from __future__ import annotations

import torch
import torch.nn as nn


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int | None = None) -> torch.Tensor:
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    cnt = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    out.scatter_add_(0, index.unsqueeze(1).expand(-1, src.size(1)), src)
    cnt.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / cnt.clamp(min=1).unsqueeze(1)


class NodePromptFeatureWeighted(nn.Module):
    """GraphPrompt-style node prompt: element-wise feature scaling."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, input_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class GraphPromptWeightedSum(nn.Module):
    """GraphPrompt-style weighted graph readout over nodes."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, hidden_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return scatter_mean(h * self.weight, batch)

