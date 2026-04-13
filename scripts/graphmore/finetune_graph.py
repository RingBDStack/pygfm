#!/usr/bin/env python
"""
GraphMoRE DownPromptGraph: load LP pretrained weights -> joint finetune experts + gating + classifier.

Graph classification:
1. Load graph classification data (e.g. MUTAG, PTC_MR)
2. Build DownPromptGraphModel + load LP weights
3. Ego subgraphs + shortest paths on disjoint-union batches
4. Node embed -> scatter mean -> graph logits

Examples:
  python scripts/graphmore/finetune_graph.py --dataset MUTAG --backbone gcn
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
warnings.filterwarnings("ignore")

from pygfm.baseline_models.graphmore import GraphMoREDownPromptGraphModel
from pygfm.baseline_models.graphmore.preprompt import (
    EgoGraphSampler,
    compute_shortest_path_distances,
)
from pygfm.public.utils.runtime import load_multi_graph_pyg_dataset, set_seed

try:
    from geoopt.optim import RiemannianAdam
except ImportError:
    raise ImportError("GraphMoRE requires geoopt. Install with: pip install geoopt")

from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def collate_to_disjoint(batch_data):
    """Concatenate a DataLoader batch into disjoint-union format."""
    xs, eis, batches, ys = [], [], [], []
    offset = 0
    for i, data in enumerate(batch_data):
        xs.append(data.x)
        eis.append(data.edge_index + offset)
        batches.append(torch.full((data.x.size(0),), i, dtype=torch.long))
        ys.append(data.y)
        offset += data.x.size(0)
    return (
        torch.cat(xs, dim=0),
        torch.cat(eis, dim=1),
        torch.cat(batches, dim=0),
        torch.cat(ys, dim=0),
    )


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(description="GraphMoRE graph classification finetune")
    p.add_argument("--dataset", type=str, default="MUTAG")
    p.add_argument(
        "--data_root",
        type=str,
        default="datasets/graphmore",
        help="Same as load_all_datasets: TU subdir or PyG layout",
    )
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--seed", type=int, default=3047)
    p.add_argument("--exp_iters", type=int, default=10)
    p.add_argument("--fold", type=int, default=10, help="K-fold cross validation")

    p.add_argument("--init_curvs", type=float, nargs="+", default=[-3, -1, 0, 1, 3])
    p.add_argument("--hidden_features", type=int, default=64)
    p.add_argument("--embed_features", type=int, default=32)
    p.add_argument("--sample_hops", type=int, nargs="+", default=[2, 3])

    p.add_argument("--backbone", type=str, default="gcn", choices=["gcn", "gat", "sage"])
    p.add_argument("--hidden_cls", type=int, default=32)
    p.add_argument("--n_layers_cls", type=int, default=2)
    p.add_argument("--drop_edge_cls", type=float, default=0.0)
    p.add_argument("--drop_feat_cls", type=float, default=0.0)
    p.add_argument("--coef_dis", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=32)

    p.add_argument("--lr_riemann", type=float, default=0.01)
    p.add_argument("--lr_cls", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--log_interval", type=int, default=20)

    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--swanlab_project", type=str, default="gfm-toolbox-graphmore")
    return p.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = _parse()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, _nc = load_multi_graph_pyg_dataset(args.data_root, args.dataset)
    if dataset[0].x is None:
        max_degree = 0
        for data in dataset:
            d = data.edge_index[0].bincount()
            max_degree = max(max_degree, d.max().item())
        from torch_geometric.transforms import OneHotDegree

        dataset = TUDataset(
            root=args.data_root,
            name=args.dataset,
            transform=OneHotDegree(max_degree=max_degree),
        )

    in_features = dataset[0].x.size(1)
    n_classes = dataset.num_classes
    print(
        f"GraphMoRE Graph Classification | dataset={args.dataset} "
        f"n_graphs={len(dataset)} in_features={in_features} n_classes={n_classes}"
    )

    all_accs = []

    for exp_iter in range(args.exp_iters):
        print(f"\n--- Experiment {exp_iter + 1}/{args.exp_iters} ---")
        perm = torch.randperm(len(dataset))
        fold_size = len(dataset) // args.fold
        test_idx = perm[: fold_size]
        train_idx = perm[fold_size:]

        train_loader = DataLoader(dataset[train_idx], batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset[test_idx], batch_size=args.batch_size)

        model = GraphMoREDownPromptGraphModel(
            in_features=in_features,
            embed_features=args.embed_features,
            init_curvs=args.init_curvs,
            sample_hops=args.sample_hops,
            hidden_features_expert=args.hidden_features,
            backbone=args.backbone,
            hidden_features_cls=args.hidden_cls,
            num_classes=n_classes,
            n_layers_cls=args.n_layers_cls,
            drop_edge_cls=args.drop_edge_cls,
            drop_feat_cls=args.drop_feat_cls,
            coef_dis=args.coef_dis,
            device=device,
        )

        if args.ckpt and os.path.exists(args.ckpt):
            ckpt = torch.load(args.ckpt, map_location=device)
            loaded = model.load_preprompt_checkpoint(ckpt)
            print(f"Loaded {len(loaded)} keys from {args.ckpt}")

        r_optim = RiemannianAdam(
            model.get_riemannian_params(),
            lr=args.lr_riemann,
            weight_decay=args.weight_decay,
            stabilize=100,
        )
        e_optim = torch.optim.Adam(
            model.get_euclidean_params(),
            lr=args.lr_cls,
            weight_decay=args.weight_decay,
        )

        best_test_acc = 0.0
        patience_count = 0

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for batch_data in train_loader:
                batch_list = batch_data.to_data_list()
                x, ei, batch, y = collate_to_disjoint(batch_list)
                x, ei, batch, y = x.to(device), ei.to(device), batch.to(device), y.to(device)

                sampler = EgoGraphSampler(args.sample_hops)
                sub_f, sub_e, sub_b = sampler.sample(x, ei)

                r_optim.zero_grad()
                e_optim.zero_grad()

                logits, loss_dis = model.forward_graph(
                    x, ei, batch, sub_f, sub_e, sub_b
                )
                loss = F.cross_entropy(logits, y) + args.coef_dis * loss_dis
                loss.backward()
                r_optim.step()
                e_optim.step()
                total_loss += loss.item()

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch_data in test_loader:
                    batch_list = batch_data.to_data_list()
                    x, ei, batch, y = collate_to_disjoint(batch_list)
                    x, ei, batch = x.to(device), ei.to(device), batch.to(device)
                    sampler = EgoGraphSampler(args.sample_hops)
                    sub_f, sub_e, sub_b = sampler.sample(x, ei)
                    pred = model.predict_graph(x, ei, batch, sub_f, sub_e, sub_b)
                    all_preds.append(pred.argmax(dim=-1).cpu())
                    all_labels.append(y)

            preds = torch.cat(all_preds)
            labels = torch.cat(all_labels)
            test_acc = accuracy_score(labels.numpy(), preds.numpy())

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                patience_count = 0
            else:
                patience_count += 1
                if patience_count > args.patience:
                    break

            if epoch % args.log_interval == 0:
                print(
                    f"  Epoch {epoch}: loss={total_loss / len(train_loader):.4f} "
                    f"test_acc={test_acc:.4f} best={best_test_acc:.4f}"
                )

        print(f"  Best test accuracy: {best_test_acc:.4f}")
        all_accs.append(best_test_acc)

    print(f"\n=== Results over {args.exp_iters} runs ===")
    print(f"  Accuracy: {np.mean(all_accs)*100:.2f} ± {np.std(all_accs)*100:.2f}")


if __name__ == "__main__":
    main()
