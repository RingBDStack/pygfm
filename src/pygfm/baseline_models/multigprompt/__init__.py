"""
MultiGPrompt baseline (WWW 2024): multi-task pre-training + prompting on graphs.

Experiment scripts live under ``scripts/multigprompt/`` (e.g. ``execute.py``).
"""

from .downprompt import downprompt, featureprompt
from .preprompt import PrePrompt

__all__ = ["PrePrompt", "downprompt", "featureprompt"]
