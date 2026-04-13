"""
1-pos-many-neg scoring, BCE, Margin Ranking, etc. (legacy OFA / GP stack).

Canonical implementations live here; ``pygfm.baseline_models.oneforall.gp.nn.loss`` re-exports only.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class IDLoss(nn.Module):
    """Identity: ``forward(res) -> res``; placeholder or passthrough."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, res: torch.Tensor) -> torch.Tensor:
        return res


class NegLogLoss(nn.Module):
    """Reshape to ``[*, neg_sample+1]``; neg-log on first column (positive), mean neg-log(1-sigmoid) on negatives."""

    def __init__(self, num_neg_samples: int) -> None:
        super().__init__()
        self.neg_sample = num_neg_samples

    def forward(self, res: torch.Tensor) -> torch.Tensor:
        score_mat = res.view(-1, self.neg_sample + 1)
        score_mat = torch.sigmoid(score_mat)
        loss = torch.mean(
            -torch.log(score_mat[:, 0])
            - (torch.log(1 - score_mat[:, 1:])).mean(dim=-1)
        )
        return loss


class FirstPosNegLoss(nn.Module):
    """1 pos many neg, BCEWithLogits on flat logits (targets: first column 1, rest 0)."""

    def __init__(self, num_neg_samples: int) -> None:
        super().__init__()
        self.neg_sample = num_neg_samples
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, res: torch.Tensor) -> torch.Tensor:
        score_mat = res.view(-1, self.neg_sample + 1)
        target = torch.zeros_like(score_mat)
        target[:, 0] = 1
        return self.loss(score_mat.flatten(), target.flatten())


class MRRLoss(nn.Module):
    """Positive score should exceed each negative: ``MarginRankingLoss``."""

    def __init__(self, num_neg_samples: int, margin: float = 15.0) -> None:
        super().__init__()
        self.loss = nn.MarginRankingLoss(margin, reduction="sum")
        self.num_neg_samples = num_neg_samples

    def forward(self, res: torch.Tensor) -> torch.Tensor:
        scores_mat = res.view(-1, self.num_neg_samples + 1)
        score_pos = (
            scores_mat[:, 0]
            .unsqueeze(1)
            .repeat_interleave(self.num_neg_samples, dim=-1)
        )
        score_neg = scores_mat[:, 1:]
        return self.loss(
            score_pos,
            score_neg,
            torch.ones_like(score_pos, device=res.device),
        )
