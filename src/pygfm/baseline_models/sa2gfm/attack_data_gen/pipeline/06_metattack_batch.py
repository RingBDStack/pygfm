#!/usr/bin/env python3
"""
Step 6 (optional) — Batched Metattack; saves `outputs/metattack_batch/{dataset}/batch_*.pkl`.
Requires Step 5 outputs in `outputs/surrogate_deeprobust/`.
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from joblib import Parallel, delayed
from torch_geometric.utils import to_scipy_sparse_matrix

from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack
from deeprobust.graph.utils import sparse_mx_to_torch_sparse_tensor

from pygfm.baseline_models.sa2gfm.attack_data_gen.lib.data_utils import load_graph
from pygfm.baseline_models.sa2gfm.paths import paths
from pygfm.public.cli.yaml_config import parse_args_with_config


def normalize(mx):
    rowsum = torch.spmm(mx, torch.ones((mx.shape[0], 1)).to(mx.device))
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, mx.to_dense()), d_mat_inv_sqrt).to_sparse()


def run_one_batch_attack(
    features_torch, adj, adj_torch, labels, idx_train, idx_unlabeled,
    surrogate_state, n_perturb, ll_constraint, batch_id,
):
    num_nodes = adj.shape[0]
    nfeat = features_torch.shape[1]
    nclass = labels.max().item() + 1
    surrogate = GCN(
        nfeat=nfeat, nclass=nclass, nhid=16, dropout=0,
        with_relu=False, with_bias=False, device="cpu",
    ).to("cpu")
    surrogate.load_state_dict(surrogate_state)
    surrogate.features = features_torch
    surrogate.adj_norm = normalize(adj_torch)
    surrogate.output = surrogate.forward(surrogate.features, surrogate.adj_norm)

    attacker = Metattack(
        surrogate,
        nnodes=num_nodes,
        feature_shape=features_torch.shape,
        attack_structure=True,
        attack_features=True,
        device="cpu",
        lambda_=0.5,
        train_iters=100,
        lr=0.1,
    ).to("cpu")
    attacker.attack(
        ori_features=features_torch,
        ori_adj=adj,
        labels=labels.cpu().numpy(),
        idx_train=idx_train,
        idx_unlabeled=idx_unlabeled,
        n_perturbations=n_perturb,
        ll_constraint=ll_constraint,
    )
    modified_adj = attacker.modified_adj.cpu().numpy()
    modified_feat = attacker.modified_features.cpu().numpy()
    diff_adj = modified_adj - adj.toarray()
    edges_flipped = [
        tuple(sorted(e)) for e in np.array(diff_adj.nonzero()).T if diff_adj[e[0], e[1]] != 0
    ]
    diff_feat = modified_feat - features_torch.cpu().numpy()
    features_flipped = [tuple(idx) for idx in np.array(diff_feat.nonzero()).T]
    return {
        "batch_id": batch_id,
        "edges_flipped": edges_flipped,
        "features_flipped": features_flipped,
    }


def main():
    parser = argparse.ArgumentParser(description="SA2GFM attack: batched Metattack")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--perturb_ratio", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--ll_constraint", action="store_true")
    args = parse_args_with_config(parser, script_file=Path(__file__))

    paths.ensure_output_dirs()
    sur_dir = paths.surrogate_deeprobust_dir
    save_dir = paths.metattack_batch_dir / args.dataset
    save_dir.mkdir(parents=True, exist_ok=True)

    data = load_graph(args.dataset)
    features = data.enhanced_x_64
    edge_index = data.edge_index
    labels = data.y
    num_nodes = features.shape[0]
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    adj_torch = sparse_mx_to_torch_sparse_tensor(adj).float()
    features_torch = torch.FloatTensor(features.cpu().numpy())

    sur_path = sur_dir / f"{args.dataset}_surrogate.pt"
    idx_path = sur_dir / f"{args.dataset}_indices.pt"
    if not sur_path.is_file() or not idx_path.is_file():
        raise FileNotFoundError(f"Run 05_metattack_surrogate.py first; missing {sur_path} or {idx_path}")

    surrogate = GCN(
        nfeat=features_torch.shape[1],
        nclass=labels.max().item() + 1,
        nhid=16,
        dropout=0,
        with_relu=False,
        with_bias=False,
        device="cpu",
    ).to("cpu")
    surrogate.load_state_dict(torch.load(sur_path, map_location="cpu"))
    surrogate_state = surrogate.state_dict()

    idx_dict = torch.load(idx_path, map_location="cpu")
    idx_train = idx_dict["idx_train"].cpu().numpy()
    idx_unlabeled = np.setdiff1d(np.arange(num_nodes), idx_train)

    ll_constraint = args.ll_constraint
    d_min = 2
    degree_sequence = np.array(adj.sum(axis=1)).flatten()
    degrees_ge_d_min = degree_sequence[degree_sequence >= d_min]
    if len(degrees_ge_d_min) < 2:
        print("powerlaw check failed -> ll_constraint=False")
        ll_constraint = False

    num_edges = adj.nnz // 2
    total_perturb = int(args.perturb_ratio * num_edges)
    num_batches = (total_perturb + args.batch_size - 1) // args.batch_size
    batch_perturbs = [args.batch_size] * num_batches
    if total_perturb % args.batch_size != 0:
        batch_perturbs[-1] = total_perturb - args.batch_size * (num_batches - 1)

    reports = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(run_one_batch_attack)(
            features_torch, adj, adj_torch, labels, idx_train, idx_unlabeled,
            surrogate_state, n_batch, ll_constraint, b + 1,
        )
        for b, n_batch in enumerate(batch_perturbs)
    )
    for report in reports:
        with open(save_dir / f"batch_{report['batch_id']}.pkl", "wb") as f:
            pickle.dump(report, f)
    print(f"Done -> {save_dir}")


if __name__ == "__main__":
    main()
