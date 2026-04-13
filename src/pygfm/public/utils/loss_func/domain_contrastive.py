"""
Matrix-style contrastive loss on multi-domain graphs (in-domain positive, cross-domain negative, InfoNCE-like).

Canonical implementation in this package (``data.loss_calculation`` only re-exports for compatibility).
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLossModule(nn.Module):
    """Matrix-based Contrastive Loss (InfoNCE style) for multi-domain graph learning."""

    def __init__(
        self,
        temperature: float = 0.07,
        metric: Literal["euclidean", "cosine"] = "cosine",
    ) -> None:
        """
        :param temperature: Softmax temperature for similarity scaling.
        :param metric: ``cosine`` recommended for GNN embeddings.
        """
        super().__init__()
        self.temperature = temperature
        self.metric = metric

    def forward(self, h: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        :param h: ``[N, D]`` node embeddings.
        :param batch_idx: ``[N]`` domain ids (e.g. ``big_batch.batch``).
        """
        N = h.size(0)

        if self.metric == "cosine":
            h = F.normalize(h, dim=1)
            logits = torch.matmul(h, h.t()) / self.temperature
        elif self.metric == "euclidean":
            dist = torch.cdist(h, h, p=2)
            logits = -dist / self.temperature
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        pos_mask = (batch_idx.unsqueeze(0) == batch_idx.unsqueeze(1)).float()
        diag_mask = torch.eye(N, device=h.device)
        pos_mask = pos_mask - diag_mask

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * (1 - diag_mask)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        pos_counts = pos_mask.sum(1)
        valid_indices = pos_counts > 0

        if not valid_indices.any():
            return torch.tensor(0.0, device=h.device, requires_grad=True)

        mean_log_prob_pos = (pos_mask * log_prob).sum(1)[valid_indices] / pos_counts[valid_indices]
        return -mean_log_prob_pos.mean()


__all__ = ["ContrastiveLossModule"]
