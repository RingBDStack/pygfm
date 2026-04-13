"""GraphText offline smoke test: no HF download, no API calls.

- ``GraphTextOfflineTokenizer``: byte ids + extensible special tokens, matching ``GraphText`` tokenizer usage.
- ``build_offline_smoke_llm_and_tokenizer``: randomly initialized ``LlamaForCausalLM`` (transformers class only, no weights).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence, Union

import torch as th
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.tokenization_utils_base import BatchEncoding


BYTE_START = 10  # ids [BYTE_START, BYTE_START+255] map to UTF-8 single bytes


class GraphTextOfflineTokenizer:
    """Minimal tokenizer: no vocab file, no network. Compatible with ``smart_tokenizer_and_embedding_resize``."""

    model_input_names = ["input_ids", "attention_mask"]
    padding_side = "right"
    truncation_side = "right"

    def __init__(self, model_max_length: int = 2048) -> None:
        self.model_max_length = model_max_length
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.unk_token = "<unk>"
        self.unk_token_id = 1
        self.bos_token = "<s>"
        self.bos_token_id = 2
        self.eos_token = "</s>"
        self.eos_token_id = 3
        self._tok2id: Dict[str, int] = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }
        self._id2tok: Dict[int, str] = {v: k for k, v in self._tok2id.items()}
        self._next_id = BYTE_START + 256

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._tok2id)

    def __len__(self) -> int:
        return self._next_id

    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None:
        """Match HF Tokenizer API; smoke test writes minimal JSON placeholder."""
        del kwargs
        os.makedirs(save_directory, exist_ok=True)
        meta = {
            "type": "GraphTextOfflineTokenizer",
            "model_max_length": self.model_max_length,
            "vocab_size": self._next_id,
        }
        path = os.path.join(save_directory, "offline_tokenizer.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _id_for_byte(self, b: int) -> int:
        return BYTE_START + int(b)

    def _special_keys_longest_first(self) -> List[str]:
        return sorted(
            (k for k in self._tok2id if "<" in k and k not in (self.pad_token, self.unk_token)),
            key=len,
            reverse=True,
        )

    def _encode_mixed(self, text: str) -> List[int]:
        """Interleave ``<c0>``, ``<x emb>``, etc. with UTF-8 bytes as ids (same as ``convert_tokens_to_ids``)."""
        if text in self._tok2id:
            return [self._tok2id[text]]
        ids: List[int] = []
        specials = self._special_keys_longest_first()
        i = 0
        n = len(text)
        while i < n:
            hit = False
            for sp in specials:
                if text.startswith(sp, i):
                    ids.append(self._tok2id[sp])
                    i += len(sp)
                    hit = True
                    break
            if hit:
                continue
            ch = text[i]
            for b in ch.encode("utf-8"):
                ids.append(self._id_for_byte(b))
            i += 1
        return ids

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> List[int]:
        del kwargs
        core = self._encode_mixed(text)
        if add_special_tokens:
            return [self.bos_token_id] + core + [self.eos_token_id]
        return core

    def decode(
        self,
        token_ids: Union[int, List[int], th.Tensor],
        skip_special_tokens: bool = False,
        **kwargs: Any,
    ) -> str:
        del kwargs
        if isinstance(token_ids, th.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if skip_special_tokens:
            token_ids = [i for i in token_ids if i not in (self.pad_token_id, self.bos_token_id, self.eos_token_id)]
        parts: List[str] = []
        cur_bytes: List[int] = []
        for i in token_ids:
            if BYTE_START <= i < BYTE_START + 256:
                cur_bytes.append(i - BYTE_START)
            else:
                if cur_bytes:
                    parts.append(bytes(cur_bytes).decode("utf-8", errors="replace"))
                    cur_bytes = []
                tok = self._id2tok.get(i)
                if tok is not None and not skip_special_tokens:
                    parts.append(tok)
                elif tok is None and i >= BYTE_START + 256:
                    parts.append(f"<id{i}>")
        if cur_bytes:
            parts.append(bytes(cur_bytes).decode("utf-8", errors="replace"))
        return "".join(parts)

    def batch_decode(self, sequences: Sequence, skip_special_tokens: bool = False, **kwargs: Any) -> List[str]:
        del kwargs
        out = []
        for seq in sequences:
            if isinstance(seq, th.Tensor):
                seq = seq.tolist()
            out.append(self.decode(seq, skip_special_tokens=skip_special_tokens))
        return out

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            if tokens in self._tok2id:
                return self._tok2id[tokens]
            return self.unk_token_id
        return [self.convert_tokens_to_ids(t) for t in tokens]

    def convert_ids_to_tokens(self, ids: Union[int, List[int]], **kwargs: Any) -> Union[str, List[str]]:
        del kwargs
        if isinstance(ids, int):
            return self._id2tok.get(ids, f"<id{ids}>")
        return [self.convert_ids_to_tokens(i) for i in ids]

    def tokenize(self, text: str, **kwargs: Any) -> List[str]:
        del kwargs
        return [str(i) for i in self.encode(text, add_special_tokens=False)]

    def add_special_tokens(self, special_tokens_dict: Dict[str, Any]) -> int:
        added = 0
        extra = special_tokens_dict.get("additional_special_tokens") or []
        for t in extra:
            if t not in self._tok2id:
                self._tok2id[t] = self._next_id
                self._id2tok[self._next_id] = t
                self._next_id += 1
                added += 1
        return added

    def __call__(
        self,
        text: Union[str, List[str]],
        text_pair: Any = None,
        return_tensors: str | None = None,
        padding: bool | str = False,
        max_length: int | None = None,
        truncation: bool = False,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> BatchEncoding:
        del kwargs, text_pair
        if isinstance(text, str):
            seqs = [self.encode(text, add_special_tokens=add_special_tokens)]
        else:
            seqs = [self.encode(t, add_special_tokens=add_special_tokens) for t in text]

        if truncation and max_length:
            seqs = [s[:max_length] for s in seqs]

        if padding == "longest" or padding is True:
            m = max(len(s) for s in seqs) if seqs else 0
            attn = []
            for s in seqs:
                pad_n = m - len(s)
                attn.append([1] * len(s) + [0] * pad_n)
                s.extend([self.pad_token_id] * pad_n)
        else:
            attn = [[1] * len(s) for s in seqs]

        if return_tensors == "pt":
            return BatchEncoding(
                {
                    "input_ids": th.tensor(seqs, dtype=th.long),
                    "attention_mask": th.tensor(attn, dtype=th.long),
                }
            )
        # Match HF: batch (list[str]) → nested lists; single string → flat id list
        if isinstance(text, list):
            return BatchEncoding({"input_ids": seqs, "attention_mask": attn})
        return BatchEncoding({"input_ids": seqs[0], "attention_mask": attn[0]})


def build_offline_smoke_llm_and_tokenizer(cfg: Any, max_tgt_len: int) -> tuple[GraphTextOfflineTokenizer, LlamaForCausalLM]:
    hidden = int(cfg.llm.get("smoke_hidden_size", 256))
    n_layers = int(cfg.llm.get("smoke_num_hidden_layers", 2))
    n_heads = int(cfg.llm.get("smoke_num_attention_heads", 4))
    inter = int(cfg.llm.get("smoke_intermediate_size", max(hidden * 2, 512)))
    tok = GraphTextOfflineTokenizer(model_max_length=max_tgt_len)
    vocab = len(tok)
    max_pos = max(int(max_tgt_len), 512)
    lc = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        max_position_embeddings=max_pos,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(lc)
    model.config.use_cache = False
    return tok, model


__all__ = [
    "GraphTextOfflineTokenizer",
    "build_offline_smoke_llm_and_tokenizer",
]
