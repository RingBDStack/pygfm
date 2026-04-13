"""Differentiable losses, task heads, and helpers (gather, negative sampling, few-shot CE)."""

from .cca_loss import CCALoss
from .domain_contrastive import ContrastiveLossModule
from .domain_regularizer import DomainRegularizer
from .gradient_reversal import GradientReversal
from .info_nce_mi_matrix import InfoNCEMIMatrixLoss, InfoNCEloss
from .loss_support import (
    few_shot_cross_entropy,
    gather_rows,
    sample_negative_pairs,
)
from .motif_contrastive_loss import motif_subgraph_contrastive_loss
from .node_node_contrastive import NodeNodeContrastiveLoss
from .pairwise_ranking import (
    FirstPosNegLoss,
    IDLoss,
    MRRLoss,
    NegLogLoss,
)
from .spectral_loss import SpectralRegularizationLoss, spectral_loss
from .task_head import TaskHead

__all__ = [
    "CCALoss",
    "ContrastiveLossModule",
    "DomainRegularizer",
    "FirstPosNegLoss",
    "GradientReversal",
    "IDLoss",
    "InfoNCEMIMatrixLoss",
    "InfoNCEloss",
    "MRRLoss",
    "NegLogLoss",
    "NodeNodeContrastiveLoss",
    "SpectralRegularizationLoss",
    "TaskHead",
    "few_shot_cross_entropy",
    "gather_rows",
    "sample_negative_pairs",
    "spectral_loss",
    "motif_subgraph_contrastive_loss",
]
