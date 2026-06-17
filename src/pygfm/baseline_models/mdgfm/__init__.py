"""
MDGFM models: faithful reproduction of the original MDGFM architecture.

PrePrompt: per-domain pretext + shared sumtext + balance token +
           ATT_learner refined adjacency + GCN LP + Calbound loss.
DownPrompt: prefeatureprompt + ATT_learner + alpha blending +
            prototype cosine similarity + softmax.
"""
from .preprompt import MDGFMPrePromptModel, ATTLearner, CombinePrompt
from .downprompt import (
    MDGFMDownPromptModel,
    WeightedPrompt,
    ComposedToken,
    DownstreamPrompt,
    PreFeaturePrompt,
    averageemb,
)
from .downprompt_graph import MDGFMDownPromptGraphModel, _scatter_mean
from ._tools import knn_fast, apply_non_linearity, sim_con, calc_lower_bound
from ._gcn import GCN, GcnLayers
from ._lp import Lp
from ._attentive import Attentive

__all__ = [
    "MDGFMPrePromptModel",
    "MDGFMDownPromptModel",
    "MDGFMDownPromptGraphModel",
    "ATTLearner",
    "CombinePrompt",
    "WeightedPrompt",
    "ComposedToken",
    "DownstreamPrompt",
    "PreFeaturePrompt",
    "averageemb",
    "_scatter_mean",
    "knn_fast",
    "apply_non_linearity",
    "sim_con",
    "calc_lower_bound",
    "GCN",
    "GcnLayers",
    "Lp",
    "Attentive",
]
