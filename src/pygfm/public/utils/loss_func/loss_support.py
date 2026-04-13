"""
Shared helpers for losses (not loss modules).

Negative sampling, gather, few-shot CE helpers for scripts/models to
``import``; **``nn.Module`` losses** live in sibling modules under ``loss_func``.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .negative_sampling import sample_negative_pairs


def gather_rows(feature: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Gather rows by 2D index for 1-pos-many-neg contrastive tuples.

    :param feature: ``[N, D]`` embeddings.
    :param index: ``[A, B]`` int64 node indices per row.
    :return: ``[A, B, D]``.

    Equivalent to legacy mygather.

    Temperature scaling (vs NodeNodeContrastiveLoss):

    - ``NodeNodeContrastiveLoss`` uses ``exp(cosine_sim / temperature)`` (standard InfoNCE).
    - MultigPrompt ``compareloss`` uses ``exp(cosine_sim) / temperature`` (differs from above).
    - SA2GFM ``compareloss`` uses ``exp(sim / temperature)``, same scaling as ``NodeNodeContrastiveLoss``.
    """
    idx = index.flatten().reshape(-1, 1).expand(-1, feature.size(1))
    res = torch.gather(feature, dim=0, index=idx)
    return res.reshape(index.size(0), index.size(1), feature.size(1))


def few_shot_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Thin wrapper around ``F.cross_entropy`` for ``[N,C]`` logits and ``[N]`` labels in few-shot setups
    (same as ``torch.nn.functional.cross_entropy``).
    """
    return F.cross_entropy(logits, labels, weight=weight, ignore_index=ignore_index)


__all__ = [
    "gather_rows",
    "sample_negative_pairs",
    "few_shot_cross_entropy",
]
