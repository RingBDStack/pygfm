"""Random LLM stub for OneForAll-style code paths when full weights are unavailable.

- Set env ``GFM_RANDOM_LLM=1`` to route any supported ``llm_name`` through the stub forward; or
- Set ``llm_name: stub`` in config (see ``LLM_DIM_DICT``) without the env var.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Union

import torch


def is_random_llm_enabled() -> bool:
    v = os.environ.get("GFM_RANDOM_LLM", "").strip().lower()
    return v in ("1", "true", "yes", "on")


class RandomLlmStub(torch.nn.Module):
    def __init__(self, indim: int):
        super().__init__()
        self.indim = indim

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        **kwargs: Any,
    ):
        if input_ids is None:
            raise ValueError("RandomLlmStub expects input_ids")
        b, s = input_ids.shape
        device = input_ids.device
        h = torch.randn(b, s, self.indim, device=device, dtype=torch.float32) * 0.02
        if return_dict:
            return {"hidden_states": [h], "last_hidden_state": h}
        return (h,)


class _StubTokenizer:
    padding_side = "right"
    truncation_side = "right"
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 2

    def __init__(self, max_length: int = 500) -> None:
        self.model_max_length = max_length

    def __call__(
        self,
        texts: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str | None = "pt",
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        b = max(1, len(texts))
        s = min(8, self.model_max_length)
        return {
            "input_ids": torch.zeros(b, s, dtype=torch.long),
            "attention_mask": torch.ones(b, s, dtype=torch.long),
        }


def build_stub_tokenizer(max_length: int = 500) -> _StubTokenizer:
    return _StubTokenizer(max_length=max_length)


__all__ = ["RandomLlmStub", "build_stub_tokenizer", "is_random_llm_enabled"]
