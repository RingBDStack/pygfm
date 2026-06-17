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

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import as_sparse_adj, inject_graphs_return_sparse, inject_graphs_to_target
from .preprompt import DisenGCN
from ...public.utils import compute_prototypes


def averageemb(
    labels: torch.Tensor,
    rawret: torch.Tensor,
    nb_class: int,
) -> torch.Tensor:
    """Per-class mean embeddings via scatter (GRAVER ``averageemb``)."""
    import torch_scatter

    return torch_scatter.scatter(src=rawret, index=labels, dim=0, reduce="mean")


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

    def __init__(self, num_tokens: int, num_labels_list: List[int], *, defer_init: bool = False):
        super().__init__()
        if defer_init:
            self.moe_weights = nn.Parameter(torch.empty(num_tokens))
            self.coe_weights = nn.ParameterList([
                nn.Parameter(torch.empty(nl)) for nl in num_labels_list
            ])
        else:
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

    def generate(self, rng: np.random.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (x [num_nodes, dim], edge_index [2, E])
        """
        if isinstance(self.graphon, torch.Tensor):
            graphon_np = self.graphon.detach().cpu().numpy()
        else:
            graphon_np = np.asarray(self.graphon, dtype=np.float32)

        graphon_resized = cv2.resize(
            graphon_np,
            dsize=(self.num_nodes, self.num_nodes),
            interpolation=cv2.INTER_LINEAR,
        )

        if rng is None:
            rand_matrix = np.random.rand(self.num_nodes, self.num_nodes)
        else:
            rand_matrix = rng.random((self.num_nodes, self.num_nodes))
        sampled = (rand_matrix < graphon_resized).astype(np.int32)
        sampled = np.triu(sampled, k=1)
        sampled = sampled + sampled.T
        rows, cols = np.nonzero(sampled)
        ei_np = np.stack([rows, cols], axis=0).astype(np.int64)
        edge_index = torch.from_numpy(ei_np).to(self.token.device)
        x = self.token.detach().expand(self.num_nodes, -1).clone()
        return x, edge_index


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

        out_dim = self.disen_gcn.output_dim
        self.register_buffer(
            "prototypes",
            torch.zeros(num_classes, out_dim, device=self.device),
        )
        self.ave = torch.empty(num_classes, out_dim, device=self.device)
        self.to(self.device)

    # ---- Weight loading ----

    def load_preprompt_checkpoint(self, ckpt: dict, strict: bool = False) -> None:
        """Load shared weights from PrePrompt ckpt (masks_logits, disen_gcn.*)."""
        from .io import preprompt_state_dict_from_checkpoint

        self.load_state_dict(preprompt_state_dict_from_checkpoint(ckpt), strict=strict)

    def freeze_pretrain_parts(self) -> None:
        """Freeze pretrained parts: masks_logits + DisenGCN."""
        self.masks_logits.requires_grad = False
        for p in self.disen_gcn.parameters():
            p.requires_grad = False

    def init_downprompt_trainable(self) -> None:
        """
        Re-init downstream trainable weights on CPU before copying to device.

        GRAVER DownPrompt initializes trainable weights on CPU then copies to device;
        CUDA xavier/randn draws a different sequence, so cross-dataset repro mirrors CPU init.
        """
        hid = self.disen_gcn.output_dim

        token = torch.empty(1, self.num_sources)
        nn.init.xavier_uniform_(token)
        self.token_weights.data.copy_(token.to(self.token_weights.device))

        self.moe_coe_router.moe_weights.data.copy_(
            torch.randn(self.num_sources).to(self.moe_coe_router.moe_weights.device)
        )
        for i, coe in enumerate(self.moe_coe_router.coe_weights):
            coe.data.copy_(torch.randn(coe.numel()).to(coe.device))

        opened = torch.empty(1, self.input_dim)
        nn.init.xavier_uniform_(opened)
        self.open_prompt_weight.data.copy_(opened.to(self.open_prompt_weight.device))

        combined = torch.empty(1, 2)
        nn.init.xavier_uniform_(combined)
        self.combine_weights.data.copy_(combined.to(self.combine_weights.device))

        self.ave = torch.empty(self.num_classes, hid, device=self.device)

    @classmethod
    def for_finetune(
        cls,
        ckpt: dict,
        num_labels_list: List[int],
        num_classes: int,
        *,
        seed: int = 39,
        gen_num_nodes: int = 10,
        combine_type: str = "mul",
        device: torch.device | None = None,
    ) -> "GRAVERDownPromptModel":
        """
        Build a finetune model with GRAVER downprompt RNG (CPU init after seed).

        Frozen backbone is loaded from ``ckpt``; trainable weights are initialized only
        after ``set_seed_deterministic(seed)`` without consuming the seed beforehand.
        """
        from .io import set_seed_deterministic

        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = int(ckpt["input_dim"])
        hidden_dim = int(ckpt["hidden_dim"])
        num_sources = int(ckpt["num_sources"])

        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.input_dim = input_dim
        model.num_sources = num_sources
        model.num_classes = num_classes
        model.gen_num_nodes = gen_num_nodes
        model.combine_type = combine_type
        model.device = dev

        model.masks_logits = nn.Parameter(torch.zeros(num_sources, input_dim))
        model.disen_gcn = DisenGCN(
            input_dim,
            hidden_dim,
            init_k=ckpt.get("init_k", 2),
            delta_k=ckpt.get("delta_k", 0),
            routit=ckpt.get("routit", 1),
            tau=ckpt.get("tau", 1.0),
            dropout=ckpt.get("dropout", 0.2),
            num_layers=ckpt.get("num_layers", 1),
        )
        model.token_weights = nn.Parameter(torch.empty(1, num_sources))
        model.moe_coe_router = MoECoERouter(num_sources, num_labels_list, defer_init=True)
        model.open_prompt_weight = nn.Parameter(torch.empty(1, input_dim))
        model.combine_weights = nn.Parameter(torch.empty(1, 2))
        out_dim = hidden_dim  # DisenGCN output_dim equals hidden_dim when num_layers=1
        model.register_buffer("prototypes", torch.zeros(num_classes, out_dim))
        model.ave = torch.empty(num_classes, out_dim)

        model.load_preprompt_checkpoint(ckpt, strict=False)
        model.freeze_pretrain_parts()

        set_seed_deterministic(seed)
        model.init_downprompt_trainable()
        return model.to(dev)

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

    # ---- Standard finetune (on-the-fly graphon, edge_index, pygfm toolbox default) ----

    def forward_standard(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        idx: torch.Tensor,
        graphon_list: List[List[torch.Tensor]],
        labels: torch.Tensor | None = None,
        train: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard toolbox path: PCA-aligned features, target-graph graphon estimate."""
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
        m = emb_at_idx.size(0)
        logits = cos_sim[:m, m:]
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return probs, entropy

    # ---- Cross-dataset reproduction (external graphon, sparse adj, dual-trial eval) ----

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        idx: torch.Tensor,
        seq: torch.Tensor,
        graphon_list: List[List[torch.Tensor]],
        labels: torch.Tensor | None = None,
        train: bool = False,
        rng: np.random.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: [N, input_dim] full-graph features on target domain
        :param adj: coalesced sparse COO adjacency [N, N]
        :param idx: [M] query/support node indices for this episode
        :param seq: [M, D] pretrain backbone embs at ``idx`` (shape used for logits slice)
        :param graphon_list: [num_sources][num_labels_i] graphon tensors
        :param labels: [M] support labels (used to refresh prototypes when train=True)
        :param train: if True, update class prototypes via ``averageemb``
        :return: (probs [M, C], entropy [M])
        """
        x = x.to(self.device)
        adj = as_sparse_adj(adj, x.size(0)).to(self.device)
        idx = idx.to(self.device)
        seq = seq.to(self.device)

        x_prompted, graphon, token = self._prompt_features(x, graphon_list)

        gen = GraphonGenerator(graphon, self.gen_num_nodes, token)
        idx_list = idx.tolist()
        graphs = [gen.generate(rng=rng) for _ in range(len(idx_list))]
        x_exp, adj_exp = inject_graphs_return_sparse(graphs, x_prompted, adj, idx_list)

        embeds = self.disen_gcn(x_exp, adj_exp)
        rawret = embeds[idx]

        if train and labels is not None:
            labels = labels.to(self.device)
            self.ave = averageemb(labels, rawret, self.num_classes)

        rawret = torch.cat((rawret, self.ave), dim=0)
        cos_sim = F.cosine_similarity(rawret.unsqueeze(1), rawret.unsqueeze(0), dim=-1)
        M = seq.shape[0]
        logits = cos_sim[:M, M:]
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return probs, entropy

    # ---- Helpers ----

    def embed_backbone(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """DisenGCN embeddings without prompting (cross-dataset backbone precompute)."""
        x = x.to(self.device)
        adj = as_sparse_adj(edges, x.size(0)).to(self.device)
        return self.disen_gcn(x, adj)
