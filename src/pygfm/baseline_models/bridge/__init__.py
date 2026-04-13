"""
BRIDGE baseline: PrePrompt (domain mask + contrastive/variance pretrain) + DownPrompt (MoE + prototypes + spectral reg).
Layout matches MDGPT: preprompt.py / downprompt.py / downprompt_graph.py.
"""
from .preprompt import BridgePrePromptModel
from .downprompt import BridgeDownPromptModel
from .downprompt_graph import BridgeDownPromptGraphModel

__all__ = [
    "BridgePrePromptModel",
    "BridgeDownPromptModel",
    "BridgeDownPromptGraphModel",
]
