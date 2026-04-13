"""Domain-alignment regularizers: MMD / CORAL / adversarial (GRL) / prototype."""
from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gradient_reversal import GradientReversal


class DomainRegularizer(nn.Module):
    def __init__(
        self,
        method: Optional[Literal["mmd", "coral", "adversarial", "prototype"]] = None,
        feature_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.method = method

        if method == "adversarial":
            if feature_dim is None:
                raise ValueError("Adversarial method requires feature_dim.")
            self.discriminator = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, 1),
                nn.Sigmoid(),
            )

    def forward(
        self,
        source_feat: torch.Tensor,
        target_feat: torch.Tensor,
        source_labels: Optional[torch.Tensor] = None,
        target_labels: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        if self.method is None:
            return torch.tensor(0.0, device=source_feat.device)

        if self.method == "mmd":
            return self._mmd_loss(source_feat, target_feat)
        if self.method == "coral":
            return self._coral_loss(source_feat, target_feat)
        if self.method == "adversarial":
            return self._adversarial_loss(source_feat, target_feat, alpha)
        if self.method == "prototype":
            return self._prototype_loss(source_feat, target_feat, source_labels, target_labels)
        return torch.tensor(0.0)

    def _mmd_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        delta = torch.mean(x, dim=0) - torch.mean(y, dim=0)
        return torch.norm(delta, p=2)

    def _coral_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d = x.size(1)
        cov_x = self._compute_covariance(x)
        cov_y = self._compute_covariance(y)
        return torch.norm(cov_x - cov_y, p="fro") / (4 * d * d)

    def _compute_covariance(self, data: torch.Tensor) -> torch.Tensor:
        n = data.size(0)
        ones = torch.ones(n, 1, device=data.device)
        centered = data - (torch.mm(ones, torch.mm(ones.t(), data)) / n)
        return torch.mm(centered.t(), centered) / (n - 1)

    def _adversarial_loss(self, x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
        x_rev = GradientReversal.apply(x, alpha)
        y_rev = GradientReversal.apply(y, alpha)

        x_pred = self.discriminator(x_rev)
        y_pred = self.discriminator(y_rev)

        loss_x = F.binary_cross_entropy(x_pred, torch.ones_like(x_pred))
        loss_y = F.binary_cross_entropy(y_pred, torch.zeros_like(y_pred))
        return (loss_x + loss_y) / 2

    def _prototype_loss(self, x, y, lx, ly) -> torch.Tensor:
        if lx is None or ly is None:
            raise ValueError("Labels required for prototype alignment.")
        unique_labels = torch.unique(lx)
        loss, count = torch.tensor(0.0, device=x.device), 0
        for label in unique_labels:
            mask_x, mask_y = (lx == label), (ly == label)
            if mask_x.any() and mask_y.any():
                loss += torch.norm(x[mask_x].mean(0) - y[mask_y].mean(0), p=2)
                count += 1
        return loss / count if count > 0 else loss


__all__ = ["DomainRegularizer"]
