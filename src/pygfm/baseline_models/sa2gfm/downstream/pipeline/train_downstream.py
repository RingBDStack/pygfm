#!/usr/bin/env python3
"""
MoE downstream node classification — simplified protocol:
  For each few-shot split: train for --epochs using train nodes only (no test in the loop),
  then evaluate once on the fixed test band and print test accuracy.
  No Top-K selection, no checkpoint saving, no re-test pipeline.
"""
from __future__ import annotations

import os
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

# Without ``pip install -e .``, you can run: ``python .../baseline_models/sa2gfm/.../train_downstream.py``
_SA2GFM_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_SA2GFM_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_SA2GFM_REPO_ROOT))

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree

from pygfm.baseline_models.sa2gfm.downstream.lib.config import (
    get_args,
    get_pretrain_datasets,
    graph_attr,
    graph_edge_index_first,
    graph_feature_first,
    graph_label_first,
    normalize_sa2gfm_loaded_object,
)
from pygfm.baseline_models.sa2gfm.downstream.models.down_model import SparseLookup, downprompt


def _coerce_feature_tensor(feat):
    if sp.issparse(feat):
        feat = feat.toarray()
    if isinstance(feat, np.ndarray):
        return torch.from_numpy(np.asarray(feat, dtype=np.float32))
    if isinstance(feat, torch.Tensor):
        return feat.detach().float()
    return torch.as_tensor(feat, dtype=torch.float32)


def _coerce_edge_index_tensor(ei):
    t = torch.as_tensor(ei, dtype=torch.long)
    if t.dim() == 2 and t.size(0) != 2 and t.size(1) == 2:
        t = t.t().contiguous()
    return t


def _coerce_label_tensor(y):
    t = torch.as_tensor(y)
    if t.dim() > 1 and t.size(-1) > 1:
        t = t.argmax(dim=-1).long()
    else:
        t = t.view(-1).long()
    return t


def _resolve_ckpt_pt(save_dir: str, stem: str) -> str | None:
    """
    Find ``{stem}.pt`` under ``save_dir``; matching is **case-insensitive** (e.g. Cora.pt vs cora.pt on Linux).
    """
    root = Path(save_dir)
    if not root.is_dir():
        return None
    s = stem.strip()
    if not s:
        return None
    cf = s.casefold()
    for p in (root / f"{s}.pt", root / f"{cf}.pt"):
        if p.is_file():
            return str(p)
    try:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() == ".pt" and p.stem.casefold() == cf:
                return str(p)
    except OSError:
        pass
    return None


def _torch_load(path: str, map_location=None):
    kw = {}
    if map_location is not None:
        kw["map_location"] = map_location
    try:
        return torch.load(path, weights_only=False, **kw)
    except TypeError:
        return torch.load(path, **kw)


def build_node_to_cluster_map(all_communities):
    return {int(node): i for i, comm in enumerate(all_communities) for node in comm}


def precompute_appnp_matrix(edge_index, num_nodes, alpha=0.1, k=10, device="cpu", batch_size=512):
    with torch.no_grad():
        edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index_with_loops
        deg = degree(col, num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        T = torch.sparse_coo_tensor(edge_index_with_loops, norm, (num_nodes, num_nodes)).to(device).coalesce()

    all_S_cols = []
    for i in range(0, num_nodes, batch_size):
        batch_indices = torch.arange(i, min(i + batch_size, num_nodes), device=device)
        H_0_batch = torch.sparse_coo_tensor(
            torch.stack([batch_indices, batch_indices]),
            torch.ones(len(batch_indices), device=device),
            (num_nodes, num_nodes),
        ).coalesce()
        H_batch = H_0_batch.clone()
        for _ in range(k):
            H_batch = (torch.sparse.mm(T, H_batch).coalesce() * (1 - alpha) + H_0_batch * alpha).coalesce()
        all_S_cols.append(H_batch.cpu())

    indices = torch.cat([s.indices() for s in all_S_cols], dim=1)
    values = torch.cat([s.values() for s in all_S_cols], dim=0)
    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).to(device).coalesce()


def _synthetic_block_communities(num_nodes: int, chunk: int) -> list[list[int]]:
    chunk = max(2, min(int(chunk), num_nodes))
    out: list[list[int]] = []
    i = 0
    while i < num_nodes:
        j = min(i + chunk, num_nodes)
        block = list(range(i, j))
        if len(block) == 1 and out:
            out[-1].extend(block)
        else:
            out.append(block)
        i = j
    return out


def load_communities(file_path: str, *, num_nodes: int | None = None) -> list:
    p = Path(file_path)
    if p.is_file():
        return _torch_load(str(p))["communities"]
    if num_nodes is not None:
        chunk = max(5, min(20, num_nodes // 25 + 1))
        warnings.warn(
            f"Community file missing: {file_path}. Using synthetic block communities (~{chunk} nodes). "
            "For real communities run: python scripts/sa2gfm/detect.py --dataset <name>",
            UserWarning,
            stacklevel=2,
        )
        return _synthetic_block_communities(num_nodes, chunk)
    raise FileNotFoundError(file_path)


def _synthesize_few_shot_support(
    y: torch.Tensor,
    shot_num: int,
    test_reserve: int,
    split_id: int,
    base_seed: int,
) -> tuple[list[int], list[int]]:
    """Like toolbox generator: k-shot per class, pool [0, n - test_reserve)."""
    rng = random.Random(int(base_seed) + int(split_id) * 10007)
    n = int(y.numel())
    pool_end = max(0, n - int(test_reserve))
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx in range(pool_end):
        by_label[int(y[idx].item())].append(idx)
    support_idx: list[int] = []
    support_lab: list[int] = []
    for lab in sorted(by_label.keys()):
        pool = by_label[lab]
        if not pool:
            continue
        k = min(int(shot_num), len(pool))
        support_idx.extend(rng.sample(pool, k=k))
        support_lab.extend([lab] * k)
    return support_idx, support_lab


def load_few_shot_split(
    data_dir: str,
    split_id: int,
    *,
    y: torch.Tensor,
    shot_num: int,
    test_reserve: int,
    seed: int,
    ds_key: str,
    shot_dir_name: str,
) -> tuple[list[int], list[int]]:
    """
    Resolution order: SA2GFM ``split_{i}.pt`` → ``splits.pt`` in the same dir →
    ``downstream_data/{mdgpt|sa2gfm|multigprompt}/<ds>/<k>shot/splits.pt`` → in-memory synthetic split.
    """
    dd = Path(data_dir)
    split_pt = dd / f"split_{split_id}.pt"
    if split_pt.is_file():
        data = _torch_load(str(split_pt))
        return data["indices"], data["labels"]

    splits_pt = dd / "splits.pt"
    if splits_pt.is_file():
        blob = _torch_load(str(splits_pt))
        splits = blob.get("splits")
        if not isinstance(splits, list) or split_id < 0 or split_id >= len(splits):
            raise IndexError(
                f"{splits_pt}: splits list missing or split_id={split_id} out of range "
                f"(len={len(splits) if isinstance(splits, list) else 'n/a'})."
            )
        s = splits[split_id]
        return s["indices"], s["labels"]

    repo = Path(__file__).resolve().parents[4]
    for sub in ("mdgpt", "sa2gfm", "multigprompt"):
        alt = repo / "downstream_data" / sub / ds_key / shot_dir_name / "splits.pt"
        if alt.is_file():
            blob = _torch_load(str(alt))
            splits = blob.get("splits")
            if isinstance(splits, list) and 0 <= split_id < len(splits):
                s = splits[split_id]
                return s["indices"], s["labels"]

    idx, lab = _synthesize_few_shot_support(y, shot_num, test_reserve, split_id, seed)
    warnings.warn(
        f"No few-shot files under {str(data_dir)!r} and no shared downstream_data/*/.../splits.pt. "
        f"Synthesized {shot_num}-shot/class for split_id={split_id} (pool excludes last {test_reserve} nodes). "
        "Persist with: python scripts/sa2gfm/generate_fewshot.py --dataset <name>",
        UserWarning,
        stacklevel=2,
    )
    return idx, lab


def accuracy(pred, labels):
    return (torch.argmax(pred, dim=1) == labels).float().mean().item()


def train_one_split(
    args,
    features,
    edge_index,
    num_nodes,
    pretrained_models,
    pretrain_model_multi,
    train_idx,
    train_labels,
    test_idx,
    test_labels,
    all_communities,
    S_lookup,
    split_id: int,
):
    device = features.device
    train_labels = (
        torch.tensor(train_labels, dtype=torch.long).to(device)
        if not isinstance(train_labels, torch.Tensor)
        else train_labels.to(device)
    )
    test_labels = (
        torch.tensor(test_labels, dtype=torch.long).to(device)
        if not isinstance(test_labels, torch.Tensor)
        else test_labels.to(device)
    )
    for model_expert in pretrained_models:
        model_expert.eval()

    model = downprompt(args=args, pretrained_models=pretrained_models, pretrain_model_multi=pretrain_model_multi).to(
        device
    )
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    valid_communities = [c for c in all_communities if len(c) > 1]
    size_one_communities = [c for c in all_communities if len(c) == 1]
    node_to_cluster_id = build_node_to_cluster_map(all_communities) if args.inter_cluster_optimizer else None

    for _epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out, moe_loss, struct_loss = model(
            x=features,
            edge_index=edge_index,
            num_nodes=num_nodes,
            idx=train_idx,
            labels=train_labels,
            is_train=True,
            valid_communities=valid_communities,
            size_one_communities=size_one_communities,
            S_lookup=S_lookup,
            node_to_cluster_id=node_to_cluster_id,
        )
        loss = F.cross_entropy(out, train_labels) + args.moe_weight * moe_loss + args.structure_weight * struct_loss
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out, _, _ = model(
            x=features,
            edge_index=edge_index,
            num_nodes=num_nodes,
            idx=test_idx,
            labels=test_labels,
            is_train=False,
            valid_communities=valid_communities,
            size_one_communities=size_one_communities,
            S_lookup=S_lookup,
            node_to_cluster_id=node_to_cluster_id,
        )
    return accuracy(out, test_labels)


def main():
    args = get_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    data = normalize_sa2gfm_loaded_object(_torch_load(args.data_path))
    feat = graph_feature_first(data)
    if feat is None:
        raise ValueError(
            f"No node features in {args.data_path} (expected one of enhanced_x_64 / x / feat / …)."
        )
    if graph_attr(data, "enhanced_x_64", "enhanced_x", required=False) is None:
        print(
            "[SA2GFM downstream] Warning: no `enhanced_x_64`; using raw features. "
            "Full pipeline expects node_feature_enhance output."
        )
    edge_index = graph_edge_index_first(data)
    y = _coerce_label_tensor(graph_label_first(data))
    features = _coerce_feature_tensor(feat).to(device)
    edge_index = _coerce_edge_index_tensor(edge_index).to(device)
    num_nodes = features.shape[0]
    feat_dim = int(features.shape[1])
    if feat_dim != args.unify_dim:
        warnings.warn(
            f"Node features are {feat_dim}-D but --unify_dim={args.unify_dim}; "
            f"setting unify_dim={feat_dim} so IntraClusterOptimizer matches input x "
            "(typical when using raw x instead of enhanced_x_64).",
            UserWarning,
            stacklevel=2,
        )
        args.unify_dim = feat_dim

    all_communities = load_communities(args.community_file, num_nodes=num_nodes)

    S_lookup = None
    if args.inter_cluster_optimizer:
        S_precomputed = precompute_appnp_matrix(edge_index, num_nodes, args.appnp_alpha, args.appnp_k, device)
        S_lookup = SparseLookup(S_precomputed, num_nodes)
        del S_precomputed
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pretrain_datasets = get_pretrain_datasets(args.dataset)
    pretrained_models = []
    save_dir = args.pre_train_model_dir_single
    for name in pretrain_datasets:
        ckpt = _resolve_ckpt_pt(save_dir, name)
        if ckpt is not None:
            pretrained_models.append(_torch_load(ckpt, map_location=device).to(device))

    if not pretrained_models:
        fb = _resolve_ckpt_pt(save_dir, args.dataset)
        if fb is not None:
            m = _torch_load(fb, map_location=device).to(device)
            warnings.warn(
                f"No expert ckpts for {pretrain_datasets}; using single MoE expert from {fb} "
                "(smoke / dev only). For full MoE run: "
                "python scripts/sa2gfm/pretrain_experts_for_downstream.py --target "
                f"{args.dataset.strip()} -- -c scripts/sa2gfm/configs/pretrain_smoke.yaml",
                UserWarning,
                stacklevel=2,
            )
            pretrained_models = [m]
        else:
            raise FileNotFoundError(
                f"No expert checkpoints under {save_dir} for {pretrain_datasets}, "
                f"and no fallback matching dataset {args.dataset!r} (tried case-insensitive *.pt). "
                "Train experts: python scripts/sa2gfm/pretrain_experts_for_downstream.py --target <dataset> "
                "(optional: pass through pretrain flags after --)."
            )

    multi_model = None
    test_idx = list(range(num_nodes - 1000, num_nodes))
    test_labels_tensor = y[test_idx].to(device, dtype=torch.long)

    if not args.no_swanlab:
        import swanlab

        swanlab.init(
            project="sa2gfm_downstream",
            config=vars(args),
            requirements_collect=False,
        )

    if args.split_id >= 0:
        split_range = [args.split_id]
    else:
        split_range = range(args.num_splits)

    ds_key = args.dataset.strip().lower()
    shot_dir_name = f"{args.shot_num}shot"
    y_cpu = y.detach().cpu()
    accs = []
    for i in split_range:
        train_idx, train_labels = load_few_shot_split(
            args.down_data_dir,
            i,
            y=y_cpu,
            shot_num=args.shot_num,
            test_reserve=1000,
            seed=args.seed,
            ds_key=ds_key,
            shot_dir_name=shot_dir_name,
        )
        acc = train_one_split(
            args,
            features,
            edge_index,
            num_nodes,
            pretrained_models,
            multi_model,
            train_idx,
            train_labels,
            test_idx,
            test_labels_tensor,
            all_communities,
            S_lookup,
            split_id=i,
        )
        accs.append(acc)
        print(f"split {i:4d} | test_acc = {acc:.4f}  (train_epochs={args.epochs}, test evaluated once)")
        if not args.no_swanlab:
            import swanlab

            swanlab.log({"split": i, "test_acc": acc})

    arr = np.array(accs, dtype=np.float64)
    print(
        f"\n--- summary over {len(accs)} split(s) ---\n"
        f"mean test_acc = {arr.mean():.4f}  std = {arr.std():.4f}  min = {arr.min():.4f}  max = {arr.max():.4f}"
    )
    if not args.no_swanlab:
        import swanlab

        swanlab.log(
            {
                "summary_mean": float(arr.mean()),
                "summary_std": float(arr.std()),
                "num_splits_ran": len(accs),
            }
        )


if __name__ == "__main__":
    main()
