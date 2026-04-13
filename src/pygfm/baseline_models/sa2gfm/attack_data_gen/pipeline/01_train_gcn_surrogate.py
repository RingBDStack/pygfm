#!/usr/bin/env python3
"""
Step 1 — Train PyG GCN surrogates saved as `checkpoints/gcn_{dataset}.pth` for Nettack (Step 2).
Uses random 60/20/20 split on all nodes (same idea as original train_gat.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from pygfm.baseline_models.sa2gfm.attack_data_gen.lib.data_utils import load_graph
from pygfm.baseline_models.sa2gfm.attack_data_gen.lib.gcn_surrogate import SimpleGCN
from pygfm.baseline_models.sa2gfm.paths import paths
from pygfm.public.cli.yaml_config import parse_args_with_config

DATASETS_DEFAULT = ["cora", "citeseer", "pubmed", "wikics", "arxiv", "P-home", "P-tech"]


def train_one_epoch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.enhanced_x_64, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    logits = model(data.enhanced_x_64, data.edge_index)
    preds = logits.argmax(dim=1)
    accs = []
    for mask in (data.train_mask, data.val_mask, data.test_mask):
        acc = (preds[mask] == data.y[mask]).sum() / mask.sum()
        accs.append(acc.item())
    return accs


def train_one_dataset(dataset_name: str, epochs: int, device: torch.device):
    paths.ensure_output_dirs()
    data = load_graph(dataset_name).to(device)
    n = data.y.size(0)
    n_cls = torch.unique(data.y).shape[0]

    perm = torch.randperm(n)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    train_mask = torch.zeros(n, dtype=torch.bool, device=device)
    val_mask = torch.zeros(n, dtype=torch.bool, device=device)
    test_mask = torch.zeros(n, dtype=torch.bool, device=device)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train : n_train + n_val]] = True
    test_mask[perm[n_train + n_val :]] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    model = SimpleGCN(
        in_channels=data.enhanced_x_64.shape[1],
        hidden_channels=16,
        out_channels=n_cls,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for ep in range(1, epochs + 1):
        loss = train_one_epoch(model, data, opt)
        tr, va, te = evaluate(model, data)
        if ep % 20 == 0 or ep == epochs:
            print(f"[{dataset_name}] epoch {ep:04d} loss={loss:.4f} train={tr:.4f} val={va:.4f} test={te:.4f}")

    out_path = paths.checkpoints_dir / f"gcn_{dataset_name}.pth"
    torch.save(model.state_dict(), out_path)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="SA2GFM attack: train GCN surrogates for Nettack")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS_DEFAULT,
        help="Dataset names (must match `ori/{name}.pt`).",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parse_args_with_config(parser, script_file=Path(__file__))

    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    for name in args.datasets:
        print(f"\n=== Training surrogate GCN: {name} ===")
        train_one_dataset(name, args.epochs, device)


if __name__ == "__main__":
    main()
