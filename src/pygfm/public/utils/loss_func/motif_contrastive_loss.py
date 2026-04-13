"""CLIP-style contrastive loss for dual-view motif subgraphs (functional API)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def motif_subgraph_contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    CLIP-style contrastive on dual-view subgraph embeddings (same as ``MotifContrastiveModel.forward``).

    :param z1: structure view ``[B, D]``
    :param z2: semantic view ``[B, D]`` (row-aligned positives with z1)
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.T) / temperature
    targets = torch.arange(z1.size(0), device=z1.device, dtype=torch.long)
    return F.cross_entropy(logits, targets)


__all__ = ["motif_subgraph_contrastive_loss"]
