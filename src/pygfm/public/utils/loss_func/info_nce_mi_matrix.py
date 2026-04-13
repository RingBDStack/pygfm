"""
InfoNCE-style loss from a mutual-information matrix (legacy OFA / GP stack).

**Canonical** implementation here; ``pygfm.baseline_models.oneforall.gp.nn.loss`` re-exports only.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class InfoNCEMIMatrixLoss(nn.Module):
    """
    Square ``MI_MAT``: diagonal positives, off-diagonal negatives, InfoNCE-style objective.

    Different API from ``NodeNodeContrastiveLoss`` (node embeddings + tuple indices).
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, mi_mat: torch.Tensor) -> torch.Tensor:
        n = len(mi_mat)
        e_neg_mat = (
            mi_mat.view(-1)[1:].view(n - 1, n + 1)[:, :-1].reshape(n, n - 1)
        )
        e_pos = torch.diagonal(mi_mat)
        loss = -torch.mean(
            torch.log(torch.exp(e_pos) / torch.exp(e_neg_mat).sum(dim=-1))
        )
        return loss


# Alias matches upstream name for ``from pygfm.public.utils.loss_func import InfoNCEloss``
class InfoNCEloss(InfoNCEMIMatrixLoss):
    """Alias for the original OneForAll class name."""
