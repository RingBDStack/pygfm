#!/usr/bin/env python3
"""
Step 3 — Merge `{poisoning,evasion}_batch_*.pkl` from Step 2 into final PyG `Data` tensors:
  `outputs/attack_post/{dataset}_p{p}_final/{dataset}_{poisoning,evasion}_final.pt`
"""
from __future__ import annotations

import argparse
import pickle
from glob import glob
from pathlib import Path

import scipy.sparse as sp
import torch
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix
from tqdm import tqdm

from pygfm.baseline_models.sa2gfm.attack_data_gen.lib.data_utils import load_graph
from pygfm.baseline_models.sa2gfm.paths import paths
from pygfm.public.cli.yaml_config import parse_args_with_config


def apply_attacks_and_save(dataset_name: str, mode: str, report_dir: Path, save_dir: Path):
    data = load_graph(dataset_name)
    num_nodes = data.num_nodes
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=num_nodes).tolil()
    features = sp.csr_matrix(data.enhanced_x_64.cpu().numpy())

    pattern = str(report_dir / f"{mode}_batch_*.pkl")
    report_files = sorted(glob(pattern))
    all_added, all_removed, all_feature_changes = set(), set(), {}
    print(f"{mode}: reading {len(report_files)} report files from {report_dir}")
    for fpath in tqdm(report_files, desc=mode):
        with open(fpath, "rb") as fin:
            reports = pickle.load(fin)
            for report in reports:
                all_added.update(report["added_edges"])
                all_removed.update(report["removed_edges"])
                all_feature_changes.update(report["feature_changes"])

    conflicts = all_added & all_removed
    if conflicts:
        print(f"  edge conflicts {len(conflicts)}, prefer remove")
        all_added -= conflicts

    perturbed_adj = adj.copy().tolil()
    perturbed_features = features.copy().tolil()
    for u, v in all_added:
        perturbed_adj[u, v] = 1
        perturbed_adj[v, u] = 1
    for u, v in all_removed:
        perturbed_adj[u, v] = 0
        perturbed_adj[v, u] = 0
    for (r, c), val in all_feature_changes.items():
        perturbed_features[r, c] += val

    perturbed_edge_index, _ = from_scipy_sparse_matrix(perturbed_adj)
    perturbed_features = torch.FloatTensor(perturbed_features.toarray())
    save_dir.mkdir(parents=True, exist_ok=True)
    out_data = data.clone()
    out_data.edge_index = perturbed_edge_index
    out_data.enhanced_x_64 = perturbed_features
    out_path = save_dir / f"{dataset_name}_{mode}_final.pt"
    torch.save(out_data, out_path)
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="SA2GFM attack: assemble final attacked graphs")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--p-values", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parse_args_with_config(parser, script_file=Path(__file__))

    paths.ensure_output_dirs()
    base = paths.attack_post_dir
    for p in args.p_values:
        report_dir = base / f"{args.dataset}_p{p}"
        if not report_dir.is_dir():
            print(f"skip missing {report_dir}")
            continue
        save_dir = base / f"{args.dataset}_p{p}_final"
        for mode in ("poisoning", "evasion"):
            apply_attacks_and_save(args.dataset, mode, report_dir, save_dir)


if __name__ == "__main__":
    main()
