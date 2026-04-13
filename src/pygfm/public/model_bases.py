"""
Shared GFM model bases (import via ``pygfm.public.model_bases``).

Concrete classes under ``pygfm.baseline_models.<baseline>`` should subclass the matching
PrePrompt / DownPrompt base and set ``gfm_family`` for unified dispatch in wrappers and ckpt code.
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import torch
import torch.nn as nn

GFMStage = Literal["preprompt", "downprompt_node", "downprompt_graph", "legacy"]


@dataclass(frozen=True)
class GFMModelDescriptor:
    family: str
    stage: GFMStage | str
    class_name: str
    module: str


class GFMModelBase(nn.Module, ABC):
    gfm_family: ClassVar[str] = "unknown"
    gfm_stage: ClassVar[GFMStage | str] = "unknown"

    def __init__(self, *, device: torch.device | None = None) -> None:
        nn.Module.__init__(self)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def describe(self) -> GFMModelDescriptor:
        cls = self.__class__
        return GFMModelDescriptor(
            family=cls.gfm_family,
            stage=cls.gfm_stage,  # type: ignore[arg-type]
            class_name=cls.__name__,
            module=cls.__module__,
        )

    def gfm_extra_checkpoint_keys(self) -> dict[str, Any]:
        return {}


class GFMPrePromptModelBase(GFMModelBase):
    gfm_stage: ClassVar[GFMStage | str] = "preprompt"


class GFMDownPromptNodeModelBase(GFMModelBase):
    gfm_stage: ClassVar[GFMStage | str] = "downprompt_node"


class GFMDownPromptGraphModelBase(GFMModelBase):
    gfm_stage: ClassVar[GFMStage | str] = "downprompt_graph"


class GFMLegacyModelBase(GFMModelBase):
    gfm_stage: ClassVar[GFMStage | str] = "legacy"


__all__ = [
    "GFMStage",
    "GFMModelDescriptor",
    "GFMModelBase",
    "GFMPrePromptModelBase",
    "GFMDownPromptNodeModelBase",
    "GFMDownPromptGraphModelBase",
    "GFMLegacyModelBase",
]
