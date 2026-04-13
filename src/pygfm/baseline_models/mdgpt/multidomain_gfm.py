"""
Legacy MultiDomainGFM: adapters + unifying_prompt + heads (used by old pygfm/mdgpt.py run script).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Dict

from ...private.utlis.domain_alignment import TaskAdapter
from pygfm.private.core import GNNBackboneEncoder
from pygfm.public.utils.loss_func import TaskHead


class MultiDomainGFM(nn.Module):
    def __init__(self, aligned_dim: int, hidden_dim: int, domain_info: Dict[str, int], device: torch.device, num_sources: int):
        super().__init__()
        self.device = device
        self.domain_names = list(domain_info.keys())

        self.backbone = GNNBackboneEncoder(
            input_dim=aligned_dim,
            hidden_dim=hidden_dim,
            gnn_type="gcn"
        ).model.to(self.device)

        self.unifying_prompt = nn.Parameter(torch.randn(1, hidden_dim))
        self.adapters = nn.ModuleDict({
            d: TaskAdapter(input_dim=hidden_dim, task_type="injection", output_dim=hidden_dim).to(self.device)
            for d in self.domain_names
        })

        self.mixing_adapter = TaskAdapter(
            input_dim=hidden_dim,
            task_type="mixing",
            output_dim=hidden_dim,
            num_source_domains=num_sources
        ).to(self.device)

        self.heads = nn.ModuleDict({
            d: TaskHead(input_dim=hidden_dim, task_type="linear", output_dim=n).to(self.device)
            for d, n in domain_info.items()
        })

        self.register_buffer("static_weights", torch.zeros(1, num_sources))
        self.to(device)

    def forward(self, x, edge_index, domain_name: str, mode="train"):
        x, edge_index = x.to(self.device), edge_index.to(self.device)
        h = self.backbone(x, edge_index)

        if mode == "pretrain":
            h_adapt = self.adapters[domain_name](h) + self.unifying_prompt
            out = self.heads[domain_name](h_adapt)
            return h, out
        else:
            h_final = self.mixing_adapter(h) + self.unifying_prompt
            out = self.heads[domain_name](h_final)
            return h, (out, None)

    def sync_mixing_prompts(self, ordered_source_names: List[str], raw_prototypes: List[torch.Tensor]):
        self.eval()
        with torch.no_grad():
            for idx, name in enumerate(ordered_source_names):
                source_token = self.adapters[name].prompt_token.detach()
                self.mixing_adapter.prompt_bank[idx].copy_(source_token.squeeze())
                proto = raw_prototypes[idx].to(self.device)
                self.mixing_adapter.domain_prototypes[idx].copy_(proto)
