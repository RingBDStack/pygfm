"""
GCoT (Chain-of-Thought for Graph): PrePrompt (GCN+contrastive), DownPrompt (ConditionNet think + prototype matching).
"""
from .preprompt import GCoTPrePromptModel
from .downprompt import GCoTDownPromptModel
from .downprompt_graph import GCoTDownPromptGraphModel

__all__ = [
    "GCoTPrePromptModel",
    "GCoTDownPromptModel",
    "GCoTDownPromptGraphModel",
]
