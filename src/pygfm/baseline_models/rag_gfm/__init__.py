"""
RAG-GFM baseline: PrePrompt pretraining (baseline-specific from MDGPT; shares GCN/NodeLevelPrompt/loss patterns).
"""
from .preprompt import PrePromptModel

__all__ = ["PrePromptModel"]
