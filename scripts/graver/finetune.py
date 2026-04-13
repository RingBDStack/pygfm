#!/usr/bin/env python
"""
GRAVER DownPrompt few-shot node classification: load PrePrompt ckpt, PCA-align target domain,
graphon estimate + MoE/CoE routing -> prototype cosine matching + entropy-regularized finetune.

Examples:
  python scripts/graver/finetune.py --dataset Cora --k_shot 1 \\
    --ckpt ckpts/graver/cora/preprompt_cora.pth --split_id 0
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from pygfm.private.utlis.domain_alignment import DomainAlignment
from pygfm.baseline_models.graver import GRAVERDownPromptModel
from pygfm.public.utils.runtime import compute_prototypes, load_single_graph_dataset, set_seed
from pygfm.public.cli.yaml_config import parse_args_with_optional_yaml
from pygfm.public.cli.export_yaml import add_export_yaml_arguments, handle_export_args
from pygfm.public.cli.default_ckpt import resolve_preprompt_ckpt


# ---------------------------------------------------------------------------
# Graphon estimation helpers
# ---------------------------------------------------------------------------

def estimate_graphon(edge_index: torch.Tensor, labels: torch.Tensor,
                     num_nodes: int, resolution: int = 10) -> list[torch.Tensor]:
    """
    Estimate a simple graphon matrix per class.
    Returns ``num_classes`` tensors of shape [resolution, resolution].
    """
    import scipy.sparse as sp

    ei = edge_index.cpu().numpy()
    adj = sp.coo_matrix(
        (np.ones(ei.shape[1]), (ei[0], ei[1])),
        shape=(num_nodes, num_nodes),
    ).tocsr()

    num_classes = int(labels.max().item()) + 1
    graphons = []
    for c in range(num_classes):
        idx_c = (labels == c).nonzero(as_tuple=False).view(-1).cpu().numpy()
        if len(idx_c) < 2:
            graphons.append(torch.zeros(resolution, resolution))
            continue
        sub = adj[np.ix_(idx_c, idx_c)].toarray().astype(np.float32)
        n = sub.shape[0]
        if n <= resolution:
            pad = np.zeros((resolution, resolution), dtype=np.float32)
            pad[:n, :n] = sub
            graphons.append(torch.from_numpy(pad))
        else:
            step = n / resolution
            g = np.zeros((resolution, resolution), dtype=np.float32)
            for i in range(resolution):
                for j in range(resolution):
                    ri = slice(int(i * step), int((i + 1) * step))
                    rj = slice(int(j * step), int((j + 1) * step))
                    block = sub[ri, rj]
                    g[i, j] = block.mean() if block.size > 0 else 0.0
            graphons.append(torch.from_numpy(g))
    return graphons


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_graph(data_root: str, dataset: str):
    """Same as ``load_all_datasets``: flat ``Cora.pt`` etc. under ``data_root``."""
    return load_single_graph_dataset(data_root, dataset)


def _parse():
    p = argparse.ArgumentParser(
        description="GRAVER DownPrompt few-shot node finetune",
        epilog="YAML: -c PATH; export: --export-default-yaml / --export-run-yaml PATH (needs pyyaml)",
    )
    p.add_argument("--dataset", type=str, default="Cora")
    p.add_argument("--k_shot", type=int, default=1, choices=[1, 5])
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="PrePrompt checkpoint; if omitted, auto-detect under ckpts/graver/ by dataset",
    )
    p.add_argument("--downstream_root", type=str, default="downstream_data/graver")
    p.add_argument("--splits_path", type=str, default=None)
    p.add_argument("--split_id", type=int, default=0)
    p.add_argument("--task_num", type=int, default=0, help="If >0, run only first task_num splits")
    p.add_argument("--data_root", type=str, default="datasets/graver")
    p.add_argument("--seed", type=int, default=39)
    # Training hyperparameters
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_entropy", type=float, default=0.2)
    p.add_argument("--test_reserve", type=int, default=1000)
    p.add_argument("--gen_num_nodes", type=int, default=10)
    p.add_argument("--combine_type", type=str, default="mul")
    p.add_argument("--graphon_resolution", type=int, default=10)
    p.add_argument("--row_norm", action="store_true")
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--swanlab_project", type=str, default="gfm-toolbox-graver")
    add_export_yaml_arguments(p)
    return p, parse_args_with_optional_yaml(p)


def main():
    p, args = _parse()
    handle_export_args(p, args)
    args.ckpt = resolve_preprompt_ckpt(ROOT, "graver", args.dataset, args.ckpt)
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    use_swanlab = not args.no_swanlab
    if use_swanlab:
        try:
            import swanlab
            swanlab.init(project=args.swanlab_project,
                         experiment_name=f"graver_ft_{args.dataset}_{args.k_shot}shot")
        except ImportError:
            use_swanlab = False

    # ---- Load PrePrompt ckpt ----
    ckpt = torch.load(args.ckpt, map_location=device)
    input_dim = ckpt["input_dim"]
    hidden_dim = ckpt["hidden_dim"]
    num_sources = ckpt["num_sources"]
    ordered_names = ckpt["ordered_names"]

    # ---- Load target graph ----
    data, num_classes = _load_graph(args.data_root, args.dataset)
    x_np = data.x.cpu().numpy().astype(np.float64)
    if args.row_norm:
        rs = x_np.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        x_np /= rs
    aligner = DomainAlignment(n_components=input_dim)
    aligner.fit(x_np)
    x = torch.from_numpy(aligner.transform(x_np).astype(np.float32)).to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)

    # ---- Per-source graphon (simplified: class subgraphs on target) ----
    graphon_per_class = estimate_graphon(edge_index, y, x.size(0), args.graphon_resolution)
    num_labels_list = [len(graphon_per_class)] * num_sources
    graphon_list = [graphon_per_class] * num_sources

    # ---- Load splits ----
    if args.splits_path:
        spath = args.splits_path
    else:
        spath = os.path.join(args.downstream_root, args.dataset, f"{args.k_shot}shot", "splits.pt")
    down = torch.load(spath, map_location="cpu")
    splits = down["splits"]
    n_splits = len(splits)
    split_ids = list(range(min(args.task_num, n_splits))) if args.task_num > 0 else [args.split_id]

    test_start = max(0, y.size(0) - args.test_reserve)
    test_idx = torch.arange(test_start, y.size(0), device=device)
    test_y = y[test_idx]

    # ---- Finetune loop ----
    xent = nn.CrossEntropyLoss()
    acc_list = []

    for sid in split_ids:
        split = splits[sid]
        train_idx = torch.tensor(split["indices"], dtype=torch.long, device=device)
        train_y = torch.tensor(split["labels"], dtype=torch.long, device=device)

        model = GRAVERDownPromptModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_sources=num_sources,
            num_classes=num_classes,
            num_labels_list=num_labels_list,
            init_k=ckpt.get("init_k", 2),
            delta_k=ckpt.get("delta_k", 0),
            routit=ckpt.get("routit", 1),
            tau=ckpt.get("tau", 1.0),
            dropout=ckpt.get("dropout", 0.2),
            num_layers=ckpt.get("num_layers", 1),
            gen_num_nodes=args.gen_num_nodes,
            combine_type=args.combine_type,
            device=device,
        )
        model.load_preprompt_checkpoint(ckpt, strict=False)
        model.freeze_pretrain_parts()

        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr,
        )

        for epoch in range(args.max_epochs):
            model.train()
            opt.zero_grad()
            probs, entropy = model(x, edge_index, train_idx, graphon_list, train_y, train=True)
            loss = xent(probs, train_y) + args.lambda_entropy * entropy.mean()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            probs_test, _ = model(x, edge_index, test_idx, graphon_list)
            pred = probs_test.argmax(1)
            acc = (pred == test_y).float().mean().item()
        acc_list.append(acc)
        print(f"[{args.dataset}] {args.k_shot}-shot split {sid} test acc: {acc:.4f}")
        if use_swanlab:
            try:
                import swanlab
                swanlab.log({f"acc_split_{sid}": acc})
            except Exception:
                pass

    if len(acc_list) > 1:
        t = torch.tensor(acc_list)
        print(f"mean {t.mean():.4f} std {t.std():.4f} (n={len(acc_list)})")


if __name__ == "__main__":
    main()
