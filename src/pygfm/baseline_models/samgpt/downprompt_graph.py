"""Original-style SAMGPT downstream graph prompt head."""
from __future__ import annotations

from typing import ClassVar, Literal

import torch
import torch.nn.functional as F

from pygfm.public.model_bases import GFMDownPromptGraphModelBase
from pygfm.baseline_models.samgpt._original import averageemb
from pygfm.baseline_models.samgpt.downprompt import _DownstreamPrompt


def _scatter_mean(src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    try:
        import torch_scatter

        return torch_scatter.scatter(src=src, index=index, dim=0, reduce="mean")
    except ImportError:
        dim_size = int(index.max().item()) + 1
        out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
        cnt = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
        out.index_add_(0, index, src)
        cnt.index_add_(0, index, torch.ones(index.size(0), 1, device=src.device, dtype=src.dtype))
        return out / cnt.clamp_min(1.0)


class SAMGPTDownPromptGraphModel(GFMDownPromptGraphModelBase):
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
        batch: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        train: int | bool = 0,
        *,
        support_idx: torch.Tensor | None = None,
        support_batch: torch.Tensor | None = None,
        support_labels: torch.Tensor | None = None,
        query_idx: torch.Tensor | None = None,
        query_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if support_idx is not None:
            if train:
                idx = support_idx
                batch = support_batch
            else:
                idx = query_idx
                batch = query_batch
            labels = support_labels
        if idx is None or batch is None:
            raise ValueError("idx/batch or support/query kwargs are required")
        gcn = gcn if gcn is not None else self.gcn

        embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0)
        rawret = _scatter_mean(embeds[idx], batch)
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
