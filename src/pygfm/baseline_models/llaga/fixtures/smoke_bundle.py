"""
Offline smoke fake OPT: random LlagaOPTForCausalLM + local BPE tokenizer.

No Hub/API; first run generates smoke_opt_llm/ under this dir.
Folder name contains opt for train.py branch detection.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch


def smoke_opt_llm_dir() -> Path:
    return Path(__file__).resolve().parent / "smoke_opt_llm"


def _build_tokenizer_and_vocab_size(out_dir: Path) -> int:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    os.makedirs(out_dir, exist_ok=True)
    tokenizer_obj = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer_obj.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    specials = ["<s>", "<pad>", "</s>", "<unk>", "<graph>", "#", "\n", "Human", "Assistant", "USER"]
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=specials)
    corpus = [
        "### Human: Given a node-centered graph: <graph> classify.\n### Assistant: Theory\n" * 40,
        "Case_Based Neural_Networks Reinforcement_Learning Rule_Learning\n" * 30,
        "USER: hello ASSISTANT: world\n" * 50,
    ]
    tokenizer_obj.train_from_iterator(corpus, trainer)
    hf = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        model_max_length=1536,
        padding_side="right",
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    hf.save_pretrained(out_dir)
    return len(hf)


def _build_llaga_opt(out_dir: Path, vocab_size: int) -> None:
    from ..model.language_model.llaga_opt import LlagaOPTConfig, LlagaOPTForCausalLM

    config = LlagaOPTConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        ffn_dim=256,
        num_attention_heads=4,
        max_position_embeddings=512,
        word_embed_proj_dim=64,
        do_layer_norm_before=False,
        torch_dtype=torch.float32,
    )
    model = LlagaOPTForCausalLM(config)
    model.save_pretrained(out_dir, safe_serialization=True)


def ensure_smoke_llm_fixture() -> Path:
    """
    Create fake model/tokenizer if smoke_opt_llm/config.json missing.
    Racy if parallel first init; smoke is single-process.
    """
    out = smoke_opt_llm_dir()
    if (out / "config.json").is_file() and (out / "tokenizer.json").is_file():
        return out
    out.mkdir(parents=True, exist_ok=True)
    vs = _build_tokenizer_and_vocab_size(out)
    _build_llaga_opt(out, vs)
    return out
