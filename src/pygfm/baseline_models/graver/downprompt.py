"""
GRAVER DownPrompt: MoE/CoE routing mixes tokens and graphon → generative graph vocabulary
injection → DisenGCN encoding → prototype cosine classification + entropy regularization.
For few-shot node classification; pairs with scripts/graver/finetune.py.

Pipeline:
1. Tokens from pretrained masks (sigmoid) → learnable weights + MoE softmax → final token
2. Per-class graphons per source via CoE + MoE → final graphon
3. Token applied to features (mul/add) + open prompt → combined features
4. Sample small graphs from final graphon (GraphonGenerator), inject into target graph
5. Frozen DisenGCN encodes expanded graph → embeddings at target nodes
6. Cosine similarity to class prototypes → softmax → class probs + prediction entropy
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprompt import DisenGCN
from ...public.utils import compute_prototypes


# ---------------------------------------------------------------------------
# MoE / CoE router
# ---------------------------------------------------------------------------

class MoECoERouter(nn.Module):
    """
    Mixture-of-experts + chain-of-experts routing.

    - MoE: softmax-weighted merge over num_tokens tokens (token level)
    - CoE: per source domain, softmax merge over per-class graphons in that domain
    - Final graphon = MoE weights × per-source CoE-merged graphons
    """

    def __init__(self, num_tokens: int, num_labels_list: List[int]):
        super().__init__()
        self.moe_weights = nn.Parameter(torch.randn(num_tokens))
        self.coe_weights = nn.ParameterList([
            nn.Parameter(torch.randn(nl)) for nl in num_labels_list
        ])

    def forward(
        self,
        tokens: torch.Tensor,
        graphons_list: List[List[torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param tokens: [num_tokens, dim] weighted mask tokens
        :param graphons_list: graphons_list[i][j] = [S,S] graphon for source i, class j
        :return: (final_token [1, dim], final_graphon [S, S])
        """
        device = tokens.device
        moe_w = F.softmax(self.moe_weights, dim=0)
        final_token = torch.matmul(moe_w, tokens)

        per_source_graphons = []
        for i, graphons in enumerate(graphons_list):
            coe_w = F.softmax(self.coe_weights[i], dim=0).to(device)
            stacked = torch.stack(
                [g.float().to(device) if isinstance(g, torch.Tensor)
                 else torch.from_numpy(g).float().to(device)
                 for g in graphons],
                dim=0,
            )
            per_source_graphons.append(torch.einsum("l,lxy->xy", coe_w, stacked))

        stacked_g = torch.stack(per_source_graphons, dim=0)
        final_graphon = torch.einsum("t,txy->xy", moe_w.to(device), stacked_g)
        return final_token.unsqueeze(0), final_graphon


# ---------------------------------------------------------------------------
# Graphon generator
# ---------------------------------------------------------------------------

class GraphonGenerator:
    """Sample a small graph of fixed size from a graphon probability matrix; node features repeat the token."""

    def __init__(self, graphon: torch.Tensor, num_nodes: int, token: torch.Tensor):
        self.graphon = graphon
        self.num_nodes = num_nodes
        self.token = token

    def generate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (x [num_nodes, dim], edge_index [2, E])
        """
        g = self.graphon.detach().float()
        if g.dim() == 2:
            g = g.unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(
            g, size=(self.num_nodes, self.num_nodes),
            mode="bilinear", align_corners=False,
        ).squeeze()

        prob_np = resized.clamp(0, 1).cpu().numpy()
        sampled = (np.random.rand(self.num_nodes, self.num_nodes) < prob_np).astype(np.int32)
        sampled = np.triu(sampled, k=1)
        sampled = sampled + sampled.T
        rows, cols = np.nonzero(sampled)
        ei_np = np.stack([rows, cols], axis=0).astype(np.int64)
        edge_index = torch.from_numpy(ei_np).to(self.token.device)
        x = self.token.detach().expand(self.num_nodes, -1).clone()
        return x, edge_index


# ---------------------------------------------------------------------------
# Graph injection (attach sampled graphon subgraphs to the target graph)
# ---------------------------------------------------------------------------

def inject_graphs_to_target(
    gen_x_list: List[torch.Tensor],
    gen_ei_list: List[torch.Tensor],
    target_x: torch.Tensor,
    target_edge_index: torch.Tensor,
    idx: List[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Inject len(idx) generated graphs into the target at the listed nodes.
    In each generated graph, the highest-degree node merges with the target node; others append.

    :return: (expanded_x [N', dim], expanded_edge_index [2, E'])
    """
    device = target_x.device
    new_x = target_x.clone()
    N = target_x.size(0)
    rows = target_edge_index[0].tolist()
    cols = target_edge_index[1].tolist()

    for (gx, gei), tgt_node in zip(zip(gen_x_list, gen_ei_list), idx):
        if gei.numel() == 0:
            continue
        gx, gei = gx.to(device), gei.to(device)
        num_g = gx.size(0)
        deg = torch.zeros(num_g, device=device)
        deg.scatter_add_(0, gei[0], torch.ones(gei.size(1), device=device))
        hub = deg.argmax().item()

        keep = [j for j in range(num_g) if j != hub]
        if not keep:
            continue
        old2new = {old: N + i for i, old in enumerate(keep)}
        new_x = torch.cat([new_x, gx[keep]], dim=0)

        for s, t in gei.t().tolist():
            s_new = tgt_node if s == hub else old2new.get(s)
            t_new = tgt_node if t == hub else old2new.get(t)
            if s_new is not None and t_new is not None:
                rows.extend([s_new, t_new])
                cols.extend([t_new, s_new])
        N += len(keep)

    new_ei = torch.tensor([rows, cols], dtype=torch.long, device=device)
    return new_x, new_ei


# ---------------------------------------------------------------------------
# GRAVER DownPrompt node classification model
# ---------------------------------------------------------------------------

class GRAVERDownPromptModel(nn.Module):
    """
    GRAVER downstream few-shot node classification.

    Freeze masks_logits and DisenGCN (from PrePrompt).
    Trainable: token_weights, MoECoERouter, open_prompt_weight, combine_weights.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_sources: int,
        num_classes: int,
        num_labels_list: List[int],
        init_k: int = 2,
        delta_k: int = 0,
        routit: int = 1,
        tau: float = 1.0,
        dropout: float = 0.2,
        num_layers: int = 1,
        gen_num_nodes: int = 10,
        combine_type: str = "mul",
        device: torch.device | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_sources = num_sources
        self.num_classes = num_classes
        self.gen_num_nodes = gen_num_nodes
        self.combine_type = combine_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Load from pretrain (then freeze) ----
        self.masks_logits = nn.Parameter(torch.randn(num_sources, input_dim))
        self.disen_gcn = DisenGCN(
            input_dim, hidden_dim, init_k, delta_k, routit, tau, dropout, num_layers,
        )

        # ---- Downstream trainable params ----
        self.token_weights = nn.Parameter(torch.empty(1, num_sources))
        nn.init.xavier_uniform_(self.token_weights)

        self.moe_coe_router = MoECoERouter(num_sources, num_labels_list)

        self.open_prompt_weight = nn.Parameter(torch.empty(1, input_dim))
        nn.init.xavier_uniform_(self.open_prompt_weight)

        self.combine_weights = nn.Parameter(torch.empty(1, 2))
        nn.init.xavier_uniform_(self.combine_weights)

        self.register_buffer("prototypes", torch.zeros(num_classes, self.disen_gcn.output_dim))
        self.to(self.device)

    # ---- Weight loading ----

    def load_preprompt_checkpoint(self, ckpt: dict, strict: bool = False) -> None:
        """Load shared weights from PrePrompt ckpt (masks_logits, disen_gcn.*)."""
        self.load_state_dict(ckpt["model"], strict=strict)

    def freeze_pretrain_parts(self) -> None:
        """Freeze pretrained parts: masks_logits + DisenGCN."""
        self.masks_logits.requires_grad = False
        for p in self.disen_gcn.parameters():
            p.requires_grad = False

    # ---- Feature prompting ----

    def _prompt_features(
        self,
        x: torch.Tensor,
        graphon_list: List[List[torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        1. sigmoid(masks_logits) → token matrix
        2. Learnable scalar weights → MoE/CoE routing → final token + graphon
        3. Token on x (composed branch) + open-prompt branch → ELU mix

        :return: (prompted_x [N,D], graphon [S,S], final_token [D])
        """
        soft_masks = torch.sigmoid(self.masks_logits)
        weighted_tokens = self.token_weights.T * soft_masks
        token, graphon = self.moe_coe_router(weighted_tokens, graphon_list)

        if self.combine_type == "add":
            composed = token.expand(x.size(0), -1) + x
        else:
            composed = token * x

        opened = self.open_prompt_weight * x
        alpha, beta = self.combine_weights[0, 0], self.combine_weights[0, 1]
        combined = F.elu(alpha * composed + beta * opened)
        return combined, graphon, token.squeeze(0)

    # ---- Forward ----

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        idx: torch.Tensor,
        graphon_list: List[List[torch.Tensor]],
        labels: torch.Tensor | None = None,
        train: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: [N, input_dim] full-graph features on target domain
        :param edge_index: [2, E] full-graph edges on target domain
        :param idx: [M] query/support node indices for this episode
        :param graphon_list: [num_sources][num_labels_i] graphon tensors
        :param labels: [M] support labels (used to refresh prototypes when train=True)
        :param train: if True, update class prototypes
        :return: (probs [M, C], entropy [M])
        """
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        idx = idx.to(self.device)

        x_prompted, graphon, token = self._prompt_features(x, graphon_list)

        gen = GraphonGenerator(graphon, self.gen_num_nodes, token)
        idx_list = idx.tolist()
        with torch.no_grad():
            graphs = [gen.generate() for _ in range(len(idx_list))]
        gen_x = [g[0] for g in graphs]
        gen_ei = [g[1] for g in graphs]
        x_exp, ei_exp = inject_graphs_to_target(gen_x, gen_ei, x_prompted, edge_index, idx_list)

        embeds = self.disen_gcn(x_exp, ei_exp)
        emb_at_idx = embeds[idx]

        if train and labels is not None:
            labels = labels.to(self.device)
            self.prototypes.copy_(
                compute_prototypes(emb_at_idx.detach(), labels, self.num_classes)
            )

        all_emb = torch.cat([emb_at_idx, self.prototypes], dim=0)
        cos_sim = F.cosine_similarity(all_emb.unsqueeze(1), all_emb.unsqueeze(0), dim=-1)
        M = emb_at_idx.size(0)
        logits = cos_sim[:M, M:]
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return probs, entropy

    # ---- Helpers ----

    def embed_backbone(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """DisenGCN embeddings without prompting (prototype init)."""
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        return self.disen_gcn(x, edge_index)
