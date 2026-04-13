#!/usr/bin/env python3
"""
Step 2 — Nettack targeted attacks; write batch pickle reports under
`outputs/attack_post/{dataset}_p{p}/` (poisoning + evasion).
"""
from __future__ import annotations

import argparse
import gc
import os
import pickle
import random
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from joblib import Parallel, delayed
from numba.core.errors import NumbaPendingDeprecationWarning
from tqdm import tqdm

from deeprobust.graph.targeted_attack import Nettack
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix

from pygfm.baseline_models.sa2gfm.attack_data_gen.lib.data_utils import load_graph
from pygfm.baseline_models.sa2gfm.attack_data_gen.lib.gcn_surrogate import SimpleGCN
from pygfm.baseline_models.sa2gfm.paths import paths
from pygfm.public.cli.yaml_config import parse_args_with_config

warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)


def check_powerlaw_compliance(adj):
    d_min = 2
    degree_sequence = np.array(adj.sum(axis=1)).flatten()
    degrees_ge_d_min = degree_sequence[degree_sequence >= d_min]
    if len(degrees_ge_d_min) < 2:
        return False
    n = len(degrees_ge_d_min)
    s_d = np.sum(np.log(degrees_ge_d_min))
    denominator = s_d - n * np.log(d_min - 0.5)
    if np.isclose(denominator, 0):
        return False
    return True


def attack_single_node_and_get_diff(
    node, adj_orig, features_orig, labels, surrogate_model, n_perturbations, allow_structure_attack
):
    try:
        adj_to_attack = deepcopy(adj_orig)
        features_to_attack = deepcopy(features_orig)
        attacker = Nettack(
            model=surrogate_model,
            nnodes=features_to_attack.shape[0],
            attack_structure=allow_structure_attack,
            attack_features=True,
            device="cpu",
        )
        w1 = surrogate_model.gc1.weight
        w2 = surrogate_model.gc2.weight
        attacker.W = (w1 @ w2).detach().cpu().numpy()
        attacker.attack(
            features_to_attack,
            adj_to_attack,
            labels,
            node,
            n_perturbations=n_perturbations,
            verbose=False,
        )
        diff_adj = attacker.modified_adj - adj_orig
        added_edges = {
            tuple(sorted(e)) for e in np.array(diff_adj.nonzero()).T if diff_adj[e[0], e[1]] > 0
        }
        removed_edges = {
            tuple(sorted(e)) for e in np.array(diff_adj.nonzero()).T if diff_adj[e[0], e[1]] < 0
        }
        diff_features = attacker.modified_features - features_orig
        feature_changes = {
            tuple(map(int, idx)): float(diff_features[idx[0], idx[1]])
            for idx in np.array(diff_features.nonzero()).T
        }
        return {"added_edges": added_edges, "removed_edges": removed_edges, "feature_changes": feature_changes}
    except Exception as e:
        print(f"⚠️ Node {node} failed: {e}")
        return None


def run_attack_final_version(
    edge_index,
    features,
    labels,
    surrogate_model,
    target_nodes,
    n_perturbations,
    mode,
    batch_size,
    n_jobs,
    allow_structure_attack,
    save_dir,
):
    adj_orig = to_scipy_sparse_matrix(edge_index, num_nodes=features.shape[0]).tolil()
    features_orig = sp.csr_matrix(features.cpu().numpy())
    labels_np = labels.cpu().numpy()
    num_batches = (len(target_nodes) + batch_size - 1) // batch_size
    print(f"{mode}: {len(target_nodes)} targets, {num_batches} batches, n_jobs={n_jobs}")

    for b in range(num_batches):
        batch_nodes = target_nodes[b * batch_size : (b + 1) * batch_size]
        if not batch_nodes:
            continue
        batch_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(attack_single_node_and_get_diff)(
                node,
                adj_orig,
                features_orig,
                labels_np,
                surrogate_model,
                n_perturbations,
                allow_structure_attack,
            )
            for node in tqdm(batch_nodes, desc=f"{mode} batch {b+1}/{num_batches}")
        )
        valid_reports = list(filter(None, batch_results))
        batch_file = os.path.join(save_dir, f"{mode}_batch_{b+1}.pkl")
        with open(batch_file, "wb") as f:
            pickle.dump(valid_reports, f)
        print(f"  saved {len(valid_reports)} reports -> {batch_file}")
        del batch_results
        del valid_reports
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="SA2GFM attack: Nettack batch reports")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--p", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--n_jobs", type=int, default=30)
    args = parse_args_with_config(parser, script_file=Path(__file__))

    paths.ensure_output_dirs()
    ckpt = paths.checkpoints_dir / f"gcn_{args.dataset}.pth"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing {ckpt}; run pipeline/01_train_gcn_surrogate.py first.")

    save_dir = paths.attack_post_dir / f"{args.dataset}_p{args.p}"
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    data = load_graph(args.dataset).to(device)
    n = data.num_nodes
    n_cls = len(torch.unique(data.y))

    model = SimpleGCN(
        in_channels=data.enhanced_x_64.shape[1],
        hidden_channels=16,
        out_channels=n_cls,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    model.gc1.weight = model.gc1.lin.weight.T.contiguous()
    model.gc2.weight = model.gc2.lin.weight.T.contiguous()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=n)
    allow_structure = check_powerlaw_compliance(adj)
    print("structure+feature" if allow_structure else "feature-only (powerlaw check failed)")

    degrees = np.array(adj.sum(axis=1)).flatten()
    eligible = np.where(degrees >= 2)[0]
    if len(eligible) == 0:
        raise ValueError("No nodes with degree >= 2")

    n_poison = min(int(0.1 * n), len(eligible))
    poison_nodes = random.sample(list(eligible), n_poison)

    last_k = min(1000, n)
    evasion_cand = [i for i in eligible if i >= n - last_k] or list(eligible)
    n_evasion = min(int(0.1 * last_k), len(evasion_cand))
    evasion_nodes = random.sample(evasion_cand, n_evasion)

    n_jobs = args.n_jobs if args.n_jobs > 0 else (os.cpu_count() or 1)

    run_attack_final_version(
        data.edge_index,
        data.enhanced_x_64,
        data.y,
        model,
        poison_nodes,
        args.p,
        "poisoning",
        args.batch_size,
        n_jobs,
        allow_structure,
        str(save_dir),
    )
    run_attack_final_version(
        data.edge_index,
        data.enhanced_x_64,
        data.y,
        model,
        evasion_nodes,
        args.p,
        "evasion",
        args.batch_size,
        n_jobs,
        allow_structure,
        str(save_dir),
    )
    print(f"Done. Reports in {save_dir}")


if __name__ == "__main__":
    main()
