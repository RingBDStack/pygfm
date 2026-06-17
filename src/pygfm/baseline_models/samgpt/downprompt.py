"""Original-style SAMGPT downstream node prompt head."""
from __future__ import annotations

from typing import ClassVar, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from pygfm.public.model_bases import GFMDownPromptNodeModelBase
from pygfm.baseline_models.samgpt._original import averageemb, composedtoken, textprompt, weighted_prompt


class _DownstreamPrompt(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        fea_pretext_weights: list[torch.Tensor],
        str_pretext_weights: list[list[torch.Tensor]],
        combines: list[float],
        prompt_mode: str = "mul",
        ablation: str = "all",
    ) -> None:
        super().__init__()
        self.composedprompt_fea = composedtoken(fea_pretext_weights, prompt_mode)
        self.composedprompt_str = nn.ModuleList(
            [
                composedtoken([pretext[i] for pretext in str_pretext_weights], prompt_mode)
                for i in range(num_layers)
            ]
        )
        self.open_prompt_fea = textprompt(feature_dim, prompt_mode)
        self.open_prompt_str = nn.ModuleList()
        for weight in str_pretext_weights[0]:
            self.open_prompt_str.append(textprompt(weight.size(1), prompt_mode))

        self.alpha = combines[0]
        self.beta = 1.0 if len(combines) <= 1 else combines[1]
        self.weighted_prompt = weighted_prompt(2)
        self.ablation_choice = ablation

    def forward(self, seq: torch.Tensor, gcn: nn.Module, adj: torch.Tensor, sparse: bool) -> torch.Tensor:
        if self.ablation_choice == "None":
            return gcn(seq, adj, sparse, None)

        composed_seq_fea = self.composedprompt_fea(seq)
        open_seq_fea = self.open_prompt_fea(seq)
        if self.beta < 0:
            seq_fea = self.weighted_prompt([composed_seq_fea, open_seq_fea])
        else:
            seq_fea = composed_seq_fea + self.beta * open_seq_fea
        if self.ablation_choice.endswith("fo"):
            seq_fea = open_seq_fea
        elif self.ablation_choice.endswith("fc"):
            seq_fea = composed_seq_fea
        embed_fea = gcn(seq_fea, adj, sparse, None)
        if self.ablation_choice == "ft":
            return embed_fea

        composed_embed_str = gcn(seq, adj, sparse, None, self.composedprompt_str)
        open_embed_str = gcn(seq, adj, sparse, None, self.open_prompt_str)
        if self.beta < 0:
            embed_str = self.weighted_prompt([composed_embed_str, open_embed_str])
        else:
            embed_str = composed_embed_str + self.beta * open_embed_str
        if self.ablation_choice.startswith("so"):
            embed_str = open_embed_str
        elif self.ablation_choice.startswith("sc"):
            embed_str = composed_embed_str
        if self.ablation_choice == "st":
            return embed_str
        return embed_fea + self.alpha * embed_str


class SAMGPTDownPromptModel(GFMDownPromptNodeModelBase):
    gfm_family: ClassVar[str] = "samgpt"

    def __init__(
        self,
        gcn,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int,
        fea_pretext_weights: list[torch.Tensor],
        str_pretext_weights: list[list[torch.Tensor]],
        combines: list[float],
        prompt_mode: Literal["add", "mul"] = "mul",
        ablation: str = "all",
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self.gcn = gcn
        self.downstreamPrompt = _DownstreamPrompt(
            input_dim,
            hidden_dim,
            num_layers,
            fea_pretext_weights,
            str_pretext_weights,
            combines,
            prompt_mode,
            ablation,
        )
        self.nb_classes = num_classes
        self.register_buffer("ave", torch.zeros(num_classes, hidden_dim))
        self.to(self.device)

    def forward(
        self,
        features: torch.Tensor,
        adj: torch.Tensor,
        sparse: bool,
        gcn=None,
        idx: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        train: int | bool = 0,
        *,
        support_idx: torch.Tensor | None = None,
        support_labels: torch.Tensor | None = None,
        query_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if support_idx is not None:
            idx = support_idx if query_idx is None or train else query_idx
            labels = support_labels
        if idx is None:
            raise ValueError("idx/support_idx is required")
        gcn = gcn if gcn is not None else self.gcn

        embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0)
        rawret = embeds[idx]
        num = rawret.shape[0]
        if train:
            if labels is None:
                raise ValueError("labels/support_labels is required during train=True")
            self.ave.copy_(averageemb(labels=labels, rawret=rawret).detach())

        ave = self.ave.to(rawret.device)
        ret = torch.cat((rawret, ave), dim=0)
        ret = torch.cosine_similarity(ret.unsqueeze(1), ret.unsqueeze(0), dim=-1)
        ret = ret[:num, num:]
        return F.softmax(ret, dim=1)
