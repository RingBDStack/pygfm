"""
Sentence embeddings with HuggingFace **BERT** (tokenizer + ``AutoModel``).

Matches the **RAG-GFM** ``data/rag_gfm/corpus_builder.py`` ``text_encoder="bert"`` path
([CLS] pooling, ``bert-base-uncased``) for reuse outside baselines.

Requires: ``pip install transformers torch`` (optional ``accelerate``).
"""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch


class BertTextEncoder:
    """
    Encode a list of strings to dense vectors with BERT (default: **[CLS]** from last_hidden_state).

    Weights load on first ``encode``; reuse one instance to avoid repeated download/load.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: Optional[torch.device] = None,
        batch_size: int = 32,
    ):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "BertTextEncoder requires transformers: pip install transformers"
            ) from e

        self.model_name = model_name
        self.batch_size = batch_size
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(device)
        self._model.eval()
        self.embedding_dim: int = self._model.config.hidden_size

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        *,
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        :param texts: list of strings.
        :param return_numpy: if True, ``float32`` ``numpy.ndarray`` [N, D]; else ``torch.Tensor``.
        """
        if not texts:
            return (
                np.zeros((0, self.embedding_dim), dtype=np.float32)
                if return_numpy
                else torch.zeros(0, self.embedding_dim, device=self.device)
            )

        outs: List[torch.Tensor] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).to(self.device)
            h = self._model(**inputs).last_hidden_state[:, 0, :]
            outs.append(h.cpu())

        cat = torch.cat(outs, dim=0)
        if return_numpy:
            return cat.numpy().astype(np.float32)
        return cat.to(self.device)


def encode_texts_with_bert(
    texts: List[str],
    *,
    model_name: str = "bert-base-uncased",
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    return_numpy: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """
    One-shot helper: build ``BertTextEncoder`` and encode. **Best for small text batches**;
    for large corpora, construct ``encoder = BertTextEncoder(...)`` and call ``encode`` repeatedly.
    """
    enc = BertTextEncoder(model_name=model_name, device=device, batch_size=batch_size)
    return enc.encode(texts, return_numpy=return_numpy)


__all__ = ["BertTextEncoder", "encode_texts_with_bert"]
