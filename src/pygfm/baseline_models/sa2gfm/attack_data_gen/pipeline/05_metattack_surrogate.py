#!/usr/bin/env python3
"""
Step 5 (optional) — Train DeepRobust GCN surrogate for Metattack (Step 6).
Writes `outputs/surrogate_deeprobust/{dataset}_surrogate.pt` and `_indices.pt`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from deeprobust.graph.defense import GCN

from pygfm.baseline_models.sa2gfm.attack_data_gen.lib.data_utils import load_graph
from pygfm.baseline_models.sa2gfm.paths import paths
from pygfm.public.cli.yaml_config import parse_args_with_config


def main():
    parser = argparse.ArgumentParser(description="SA2GFM attack: DeepRobust GCN surrogate for Metattack")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train_ratio", type=float, default=0.05)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--feat_name", type=str, default="enhanced_x_64")
    args = parse_args_with_config(parser, script_file=Path(__file__))

    paths.ensure_output_dirs()
    save_dir = paths.surrogate_deeprobust_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    data = load_graph(args.dataset)
    features = getattr(data, args.feat_name)
    edge_index = data.edge_index
    labels = data.y
    n = features.size(0)

    adj = torch.zeros(n, n)
    adj[edge_index[0], edge_index[1]] = 1
    adj = torch.maximum(adj, adj.T)
    adj_sp = sp.csr_matrix(adj.cpu().numpy())
    features_sp = sp.csr_matrix(features.cpu().numpy())

    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    idx_all = torch.randperm(n)
    idx_train = idx_all[:n_train]
    idx_val = idx_all[n_train : n_train + n_val]

    surrogate = GCN(
        nfeat=features.shape[1],
        nclass=labels.max().item() + 1,
        nhid=16,
        dropout=0,
        with_relu=False,
        with_bias=False,
        device=args.device,
    ).to(args.device)
    surrogate.fit(
        features_sp,
        adj_sp,
        labels.cpu().numpy(),
        idx_train.cpu().numpy(),
        idx_val.cpu().numpy(),
        patience=30,
    )

    torch.save(surrogate.state_dict(), save_dir / f"{args.dataset}_surrogate.pt")
    torch.save(
        {"idx_train": idx_train, "idx_val": idx_val},
        save_dir / f"{args.dataset}_indices.pt",
    )
    print(f"Saved surrogate to {save_dir}")


if __name__ == "__main__":
    main()
