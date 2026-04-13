#!/usr/bin/env python3
"""
Step 4 — Untargeted random feature / structure perturbations for ratios
{0.2, 0.4, 0.6, 0.8}, saved to `outputs/attacked_data_random/`
(`{dataset}_feature_p{ratio}.pt`, `{dataset}_structure_p{ratio}.pt`).
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix
from tqdm import tqdm

from pygfm.baseline_models.sa2gfm.attack_data_gen.lib.data_utils import load_graph
from pygfm.baseline_models.sa2gfm.paths import paths
from pygfm.public.cli.yaml_config import parse_args_with_config


def perturb_structure(edge_index, ratio, num_nodes: int):
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tolil()
    degrees = np.array(adj.sum(axis=0)).flatten()
    row, col = adj.nonzero()
    edges = np.stack([row, col], axis=1)
    edges = edges[edges[:, 0] < edges[:, 1]]
    safe_mask = (degrees[edges[:, 0]] > 1) & (degrees[edges[:, 1]] > 1)
    safe_edges = edges[safe_mask]
    if safe_edges.shape[0] == 0:
        return edge_index
    num_del = int(ratio * safe_edges.shape[0])
    if num_del == 0:
        return edge_index
    del_edges = safe_edges[np.random.choice(safe_edges.shape[0], num_del, replace=False)]
    for u, v in tqdm(del_edges, desc="del edges", unit="e"):
        adj[u, v] = 0
        adj[v, u] = 0
    ei, _ = from_scipy_sparse_matrix(adj)
    return ei


def perturb_features(x: torch.Tensor, ratio: float):
    x_np = x.cpu().numpy().copy()
    n, fdim = x_np.shape
    total = n * fdim
    num_perturb = int(ratio * total)
    if num_perturb == 0:
        return torch.FloatTensor(x_np)
    indices = np.array([(i, j) for i in range(n) for j in range(fdim)])
    chosen = indices[np.random.choice(len(indices), num_perturb, replace=False)]
    for i, j in tqdm(chosen, desc="feat perturb", unit="cell"):
        v = x_np[i, j]
        x_np[i, j] = 1.0 - v if v in (0.0, 1.0) else v + np.random.normal(0, 0.1)
    return torch.FloatTensor(x_np)


def main():
    parser = argparse.ArgumentParser(description="SA2GFM attack: random feature/structure perturbation")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--type", type=str, choices=["feature", "structure", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    args = parse_args_with_config(parser, script_file=Path(__file__))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    paths.ensure_output_dirs()
    out_dir = paths.attack_random_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_graph(args.dataset).to(device)
    ratios = [0.2, 0.4, 0.6, 0.8]
    n = data.y.size(0)

    for ratio in ratios:
        if args.type in ("structure", "both"):
            ei = perturb_structure(data.edge_index, ratio, n)
            d = data.clone()
            d.edge_index = ei
            path = out_dir / f"{args.dataset}_structure_p{ratio}.pt"
            torch.save(d.cpu(), path)
            print(f"saved {path}")
        if args.type in ("feature", "both"):
            xf = perturb_features(data.enhanced_x_64, ratio)
            d = data.clone()
            d.enhanced_x_64 = xf.to(device)
            path = out_dir / f"{args.dataset}_feature_p{ratio}.pt"
            torch.save(d.cpu(), path)
            print(f"saved {path}")


if __name__ == "__main__":
    main()
