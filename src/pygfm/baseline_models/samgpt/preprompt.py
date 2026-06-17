"""SAMGPT PrePrompt model aligned closely with the original SAMGPT-main implementation."""
from __future__ import annotations

from typing import ClassVar, Literal, Sequence

import torch
import torch.nn as nn

from pygfm.public.model_bases import GFMPrePromptModelBase
from pygfm.baseline_models.samgpt._original import GraphCL, GcnLayers, Lp, textprompt


class SAMGPTPrePromptModel(GFMPrePromptModelBase):
    """Original-style SAMGPT pretraining model with GraphCL and LP branches."""

    gfm_family: ClassVar[str] = "samgpt"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_domains: int,
        num_layers: int = 3,
        prompt_mode: Literal["add", "mul"] = "mul",
        temperature: float = 1.0,
        alpha: float = 1.0,
        dropout: float = 0.1,
        backbone: str = "gcn",
        ablation: str = "all",
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        del temperature
        if backbone != "gcn":
            raise ValueError("Only backbone='gcn' is supported in the aligned pygfm SAMGPT patch.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains
        self.num_layers = num_layers
        self.prompt_mode = prompt_mode
        self.combine = alpha
        self.ablation_choice = ablation

        self.lp = Lp(input_dim, hidden_dim)
        self.graphcledge = GraphCL(input_dim, hidden_dim, "prelu")
        self.graphclmask = GraphCL(input_dim, hidden_dim, "prelu")

        self.feature_prompt_layers = nn.ModuleList(
            [textprompt(input_dim, prompt_mode) for _ in range(num_domains)]
        )
        self.structure_prompt_layers = nn.ModuleList(
            [
                nn.ModuleList([textprompt(hidden_dim, prompt_mode) for _ in range(num_layers)])
                for _ in range(num_domains)
            ]
        )

        self.gcn = GcnLayers(input_dim, hidden_dim, num_layers, dropout)
        self.loss = nn.BCEWithLogitsLoss()
        self.to(self.device)

    def ablation(self, fea_prelogits: torch.Tensor, str_prelogits: torch.Tensor) -> torch.Tensor:
        if self.ablation_choice == "st":
            return str_prelogits
        if self.ablation_choice == "ft":
            return fea_prelogits
        return fea_prelogits + self.combine * str_prelogits

    def compute_prelogits_lp(
        self,
        feature_prompt_layers: Sequence[nn.Module],
        structure_prompt_layers: Sequence[Sequence[nn.Module]],
        seq_list: Sequence[torch.Tensor],
        adj_list: Sequence[torch.Tensor],
        sparse: bool = False,
    ):
        for fea_pretext, str_layers, seq, adj in zip(
            feature_prompt_layers,
            structure_prompt_layers,
            seq_list,
            adj_list,
        ):
            if self.ablation_choice == "None":
                yield self.lp(self.gcn, seq, adj, sparse)
                continue
            fea_prelogits = self.lp(self.gcn, fea_pretext(seq), adj, sparse)
            str_prelogits = self.lp(self.gcn, seq, adj, sparse, str_layers)
            yield self.ablation(fea_prelogits, str_prelogits)

    def compute_prelogits_graphcl(
        self,
        feature_prompt_layers: Sequence[nn.Module],
        structure_prompt_layers: Sequence[Sequence[nn.Module]],
        seq_list: Sequence[torch.Tensor],
        adj_list: Sequence[torch.Tensor],
        sparse: bool = False,
        msk: torch.Tensor | None = None,
        samp_bias1: torch.Tensor | None = None,
        samp_bias2: torch.Tensor | None = None,
    ):
        for fea_pretext, str_layers, seq, adj in zip(
            feature_prompt_layers,
            structure_prompt_layers,
            seq_list,
            adj_list,
        ):
            if self.ablation_choice == "None":
                yield self.graphcledge(
                    self.gcn,
                    seq[0],
                    seq[1],
                    seq[2],
                    seq[3],
                    adj[0],
                    adj[1],
                    adj[2],
                    sparse,
                    msk,
                    samp_bias1,
                    samp_bias2,
                    "edge",
                )
                continue

            preseq_list = [fea_pretext(seq[i]) for i in range(len(seq))]
            fea_prelogits = self.graphcledge(
                self.gcn,
                preseq_list[0],
                preseq_list[1],
                preseq_list[2],
                preseq_list[3],
                adj[0],
                adj[1],
                adj[2],
                sparse,
                msk,
                samp_bias1,
                samp_bias2,
                "edge",
            )

            str_prelogits = self.graphcledge(
                self.gcn,
                seq[0],
                seq[1],
                seq[2],
                seq[3],
                adj[0],
                adj[1],
                adj[2],
                sparse,
                msk,
                samp_bias1,
                samp_bias2,
                "edge",
                str_layers,
            )
            yield self.ablation(fea_prelogits, str_prelogits)

    def embed(
        self,
        seq: torch.Tensor,
        adj: torch.Tensor,
        sparse: bool,
        msk: torch.Tensor | None,
        lp: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_1 = self.gcn(seq, adj, sparse, lp)
        if msk is None:
            c = torch.mean(h_1, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            c = torch.sum(h_1 * msk, 1) / torch.sum(msk, 1)
        return h_1.detach(), c.detach()

    def get_weights(self) -> tuple[list[torch.Tensor], list[list[torch.Tensor]], list[float]]:
        fea_pretext_weights = [layer.weight.detach().clone() for layer in self.feature_prompt_layers]
        str_pretext_weights = [
            [layer.weight.detach().clone() for layer in structure_prompt_layer]
            for structure_prompt_layer in self.structure_prompt_layers
        ]
        combines = [self.combine]
        return fea_pretext_weights, str_pretext_weights, combines

    def forward(
        self,
        seq_list: Sequence[torch.Tensor],
        adj_list: Sequence[torch.Tensor],
        sparse: bool,
        msk: torch.Tensor | None,
        samp_bias1: torch.Tensor | None,
        samp_bias2: torch.Tensor | None,
        lbl,
        samples=None,
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        if samples is None:
            logits = list(
                self.compute_prelogits_graphcl(
                    self.feature_prompt_layers,
                    self.structure_prompt_layers,
                    seq_list,
                    adj_list,
                    sparse,
                    msk,
                    samp_bias1,
                    samp_bias2,
                )
            )
            for logit, label in zip(logits, lbl):
                total_loss = total_loss + self.loss(logit, label)
            return total_loss

        logits = list(
            self.compute_prelogits_lp(
                self.feature_prompt_layers,
                self.structure_prompt_layers,
                seq_list,
                adj_list,
                sparse,
            )
        )
        if isinstance(samples, list):
            samples = [torch.as_tensor(sample, dtype=torch.int64, device=self.device) for sample in samples]
            for logit, sample in zip(logits, samples):
                total_loss = total_loss + compareloss(logit, sample, temperature=1.0)
            return total_loss

        samples = torch.as_tensor(samples, dtype=torch.int64, device=self.device)
        logits_cat = torch.cat(logits, dim=0)
        return compareloss(logits_cat, samples, temperature=1.0)


def mygather(feature: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    input_size = index.size(0)
    index = index.flatten().reshape(-1, 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))
    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size, -1, feature.size(1))


def compareloss(feature: torch.Tensor, tuples: torch.Tensor, temperature: float) -> torch.Tensor:
    h_tuples = mygather(feature, tuples)
    temp = torch.arange(0, len(tuples), device=feature.device).reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    h_i = mygather(feature, temp)
    sim = torch.nn.functional.cosine_similarity(h_i, h_tuples, dim=2)
    exp = torch.exp(sim) / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1 : exp.size(0)].permute(1, 0).sum(dim=1, keepdim=True)
    return (-1 * torch.log(numerator / denominator)).mean()
