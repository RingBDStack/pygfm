"""Node-level contrastive loss (cosine + InfoNCE)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss_support import gather_rows


class NodeNodeContrastiveLoss(nn.Module):
    """
    Node-node contrastive (MDGPT compareloss): cosine + InfoNCE,
    ``-log(exp(pos/T) / sum_k exp(sim_k/T))``.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, h: torch.Tensor, tuples: torch.Tensor) -> torch.Tensor:
        """
        :param h: ``[N, D]`` node embeddings.
        :param tuples: ``[N, 1+K]`` int64; row i is ``[pos_i, neg_i1, ...]``.
        """
        h_tuples = gather_rows(h, tuples)
        anchors = torch.arange(tuples.size(0), device=h.device, dtype=torch.long)
        anchors = anchors.reshape(-1, 1).expand_as(tuples)
        h_anchors = gather_rows(h, anchors)
        sim = F.cosine_similarity(h_anchors, h_tuples, dim=2)
        exp = torch.exp(sim / self.temperature)
        exp = exp.permute(1, 0)
        numerator = exp[0:1].permute(1, 0)
        denominator = exp[1:].sum(dim=0, keepdim=True).t()
        loss = -torch.log(numerator / (denominator + 1e-8) + 1e-8)
        return loss.mean()


__all__ = ["NodeNodeContrastiveLoss"]
