"""Downstream heads: linear / mlp / matching / reconstruction."""
from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        task_type: Literal["linear", "mlp", "matching", "reconstruction"],
        output_dim: Optional[int] = None,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.task_type = task_type

        if task_type == "linear":
            self.head = nn.Linear(input_dim, output_dim if output_dim else input_dim)
        elif task_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim if output_dim else input_dim),
            )
        elif task_type == "reconstruction":
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )
        elif task_type == "matching":
            self.head = nn.Identity()

    def forward(self, x: torch.Tensor, support_set: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.task_type == "matching":
            if support_set is None:
                raise ValueError("Matching head requires a support_set.")
            x_norm = F.normalize(x, p=2, dim=1)
            s_norm = F.normalize(support_set, p=2, dim=1)
            return torch.mm(x_norm, s_norm.t())

        return self.head(x)


__all__ = ["TaskHead"]
