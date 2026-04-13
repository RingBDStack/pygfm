"""
SAMGPT models: PrePrompt (feature + structure prompts, LP loss), DownPrompt (node/graph).
Reuses GFM interfaces: NodeLevelPrompt, ComposedNodeLevelPrompt, GCNEncoderSparseWithPrompts,
NodeNodeContrastiveLoss, TaskHead(matching), compute_prototypes.
"""
from .preprompt import SAMGPTPrePromptModel
from .downprompt import SAMGPTDownPromptModel
from .downprompt_graph import SAMGPTDownPromptGraphModel

__all__ = [
    "SAMGPTPrePromptModel",
    "SAMGPTDownPromptModel",
    "SAMGPTDownPromptGraphModel",
]
