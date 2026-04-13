"""
MDGFM models: PrePrompt (pretext+sumtext+GCN+contrastive), DownPrompt (node/graph).
Reuses GFM: NodeLevelPrompt, ComposedNodeLevelPrompt, GCNEncoderSparse, NodeNodeContrastiveLoss,
TaskHead(matching), compute_prototypes.
"""
from .preprompt import MDGFMPrePromptModel
from .downprompt import MDGFMDownPromptModel
from .downprompt_graph import MDGFMDownPromptGraphModel

__all__ = [
    "MDGFMPrePromptModel",
    "MDGFMDownPromptModel",
    "MDGFMDownPromptGraphModel",
]
