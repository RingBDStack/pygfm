"""
Data-side loss re-exports (used with other ``pygfm.private.utlis`` modules).

Losses and negative sampling live in ``pygfm.public.utils.loss_func``.
"""
from pygfm.public.utils.loss_func import (
    ContrastiveLossModule,
    DomainRegularizer,
    GradientReversal,
    NodeNodeContrastiveLoss,
    TaskHead,
)
from pygfm.public.utils.loss_func import sample_negative_pairs

__all__ = [
    "ContrastiveLossModule",
    "DomainRegularizer",
    "GradientReversal",
    "NodeNodeContrastiveLoss",
    "TaskHead",
    "sample_negative_pairs",
]
