"""
Spectral regularizer on graph Laplacian eigenbasis (e.g. BRIDGE DownPrompt).

Compare energy of embeddings ``h`` vs prompted features ``x_prompted`` in the eigenvector basis;
penalize excess of ``eival * ||proj h||`` over ``nu * eival * ||proj x||``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def spectral_loss(
    h: torch.Tensor,
    x_prompted: torch.Tensor,
    eivec: torch.Tensor,
    eival: torch.Tensor,
    nu: float = 0.1,
) -> torch.Tensor:
    """
    :param h: node embeddings ``[N, H]``
    :param x_prompted: prompted node features ``[N, D]`` (same as BRIDGE ``_prompted_features``)
    :param eivec: Laplacian eigenvector matrix ``[N, N]`` (layout matches caller / BRIDGE)
    :param eival: eigenvalues ``[N]`` or broadcastable with ``delta_*``
    :param nu: relative energy scale between features and embeddings
    :return: scalar ``mean(relu(eival * ||proj h|| - nu * eival * ||proj x||))``
    """
    h_hat = torch.matmul(eivec.t(), h)
    x_hat = torch.matmul(eivec.t(), x_prompted)
    delta_h = torch.norm(h_hat, p=2, dim=-1)
    delta_x = torch.norm(x_hat, p=2, dim=-1)
    reg = F.relu(eival * delta_h - nu * eival * delta_x)
    return reg.mean()


class SpectralRegularizationLoss(torch.nn.Module):
    """Same formula as :func:`spectral_loss` with configurable ``nu``."""

    def __init__(self, nu: float = 0.1) -> None:
        super().__init__()
        self.nu = nu

    def forward(
        self,
        h: torch.Tensor,
        x_prompted: torch.Tensor,
        eivec: torch.Tensor,
        eival: torch.Tensor,
    ) -> torch.Tensor:
        return spectral_loss(h, x_prompted, eivec, eival, nu=self.nu)


__all__ = ["spectral_loss", "SpectralRegularizationLoss"]
