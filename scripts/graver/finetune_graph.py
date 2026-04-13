#!/usr/bin/env python
"""
GRAVER DownPrompt graph-level few-shot: adds scatter_mean graph pooling on top of the node pipeline.

Examples:
  python scripts/graver/finetune_graph.py --dataset Cora --k_shot 1 \\
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
from pygfm.private.utlis.downstream_data_gen import build_test_subgraphs
from pygfm.baseline_models.graver import GRAVERDownPromptGraphModel
from pygfm.public.utils.runtime import load_single_graph_dataset, set_seed

from finetune import estimate_graphon


def _load_graph(data_root: str, dataset: str):
    return load_single_graph_dataset(data_root, dataset)


def _parse():
    p = argparse.ArgumentParser(description="GRAVER graph-level few-shot finetune")
    p.add_argument("--dataset", type=str, default="Cora")
    p.add_argument("--k_shot", type=int, default=1, choices=[1, 5])
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--downstream_root", type=str, default="downstream_data/graver")
    p.add_argument("--splits_path", type=str, default=None)
    p.add_argument("--split_id", type=int, default=0)
    p.add_argument("--task_num", type=int, default=0)
    p.add_argument("--data_root", type=str, default="datasets/graver")
    p.add_argument("--seed", type=int, default=39)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_entropy", type=float, default=0.2)
    p.add_argument("--test_reserve", type=int, default=1000)
    p.add_argument("--gen_num_nodes", type=int, default=10)
    p.add_argument("--combine_type", type=str, default="mul")
    p.add_argument("--graphon_resolution", type=int, default=10)
    p.add_argument("--max_one_hop", type=int, default=10)
    p.add_argument("--max_two_hop", type=int, default=4)
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--swanlab_project", type=str, default="gfm-toolbox-graver")
    return p.parse_args()


def main():
    args = _parse()
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    input_dim = ckpt["input_dim"]
    hidden_dim = ckpt["hidden_dim"]
    num_sources = ckpt["num_sources"]

    data, num_classes = _load_graph(args.data_root, args.dataset)
    x_np = data.x.cpu().numpy().astype(np.float64)
    al = DomainAlignment(n_components=input_dim)
    al.fit(x_np)
    x = torch.from_numpy(al.transform(x_np).astype(np.float32)).to(device)
    ei = data.edge_index.to(device)
    y = data.y.to(device)

    graphon_per_class = estimate_graphon(ei, y, x.size(0), args.graphon_resolution)
    num_labels_list = [len(graphon_per_class)] * num_sources
    graphon_list = [graphon_per_class] * num_sources

    sp = args.splits_path or os.path.join(
        args.downstream_root, args.dataset, f"{args.k_shot}shot_graph_batch", "splits.pt"
    )
    if not os.path.isfile(sp):
        raise FileNotFoundError(
            f"Missing {sp}\nGenerate graph-level splits first."
        )
    down = torch.load(sp, map_location="cpu")
    splits = down["splits"]
    nsp = len(splits)
    sids = list(range(min(args.task_num, nsp))) if args.task_num > 0 else [args.split_id]

    test_start = max(0, y.size(0) - args.test_reserve)
    test_indices = list(range(test_start, y.size(0)))
    test_y = y[test_start:].to(device)
    tlist, tbatch = build_test_subgraphs(
        ei.cpu(), test_indices, args.max_one_hop, args.max_two_hop, args.seed
    )
    tlist, tbatch = tlist.to(device), tbatch.to(device)

    xent = nn.CrossEntropyLoss()
    accs = []

    for sid in sids:
        split = splits[sid]
        sidx = split["idx"].to(device)
        sb = split["batch"].to(device)
        slabels = split["labels"].to(device)

        model = GRAVERDownPromptGraphModel(
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

        for _ in range(args.max_epochs):
            model.train()
            opt.zero_grad()
            probs, entropy = model.forward_graph(
                x, ei, sidx, sb, graphon_list, slabels, train=True,
            )
            loss = xent(probs, slabels) + args.lambda_entropy * entropy.mean()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            probs_t, _ = model.forward_graph(x, ei, tlist, tbatch, graphon_list)
            pred = probs_t.argmax(1)
            acc = (pred == test_y).float().mean().item()
        accs.append(acc)
        print(f"[{args.dataset}] graph {args.k_shot}-shot split {sid} test acc: {acc:.4f}")

    if len(accs) > 1:
        t = torch.tensor(accs)
        print(f"mean {t.mean():.4f} std {t.std():.4f}")


if __name__ == "__main__":
    main()
