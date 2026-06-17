#!/usr/bin/env python
"""
GraphMoRE PrePrompt: multi-Riemannian experts + topology-aware gating for link prediction.

Per-graph LP training:
1. Split edges into train/val/test
2. Build multi-resolution ego subgraphs on train edges
3. Shortest paths on the graph (distortion loss)
4. Joint train experts + gating + Fermi-Dirac decoder
5. Save checkpoint for finetune.py

Examples:
  python scripts/graphmore/pretrain.py --dataset Cora
  python scripts/graphmore/pretrain.py --dataset airport --embed_features 16
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from pygfm.public.repo_paths import driver_script_repo_root

import numpy as np
import torch

ROOT = driver_script_repo_root(__file__)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
warnings.filterwarnings("ignore")

from pygfm.baseline_models.graphmore import GraphMoREPrePromptModel
from pygfm.baseline_models.graphmore.preprompt import (
    EgoGraphSampler,
    compute_shortest_path_distances,
)
from pygfm.public.utils.runtime import load_single_graph_dataset, set_seed
from pygfm.public.cli.yaml_config import parse_args_with_optional_yaml
from pygfm.public.cli.export_yaml import add_export_yaml_arguments, handle_export_args

try:
    from geoopt.optim import RiemannianAdam
except ImportError:
    raise ImportError("GraphMoRE requires geoopt. Install with: pip install geoopt")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(name: str, root: str = "datasets/graphmore"):
    """
    Same as ``load_all_datasets``: flat ``Cora.pt`` under ``root``, or ``data.pt`` / PyG layout in subdirs.
    Returns (features, edge_index, neg_edges, labels, masks, n_classes); masks kept for API compatibility.
    """
    from torch_geometric.utils import negative_sampling

    data, n_classes = load_single_graph_dataset(root, name)
    if all(getattr(data, k, None) is not None for k in ("train_mask", "val_mask", "test_mask")):
        masks = (data.train_mask, data.val_mask, data.test_mask)
    else:
        masks = _random_split(data.y.tolist(), 0.15, 0.15)

    edge_index = data.edge_index.long()
    neg_edges = negative_sampling(edge_index, num_neg_samples=edge_index.size(1))
    return data.x, edge_index, neg_edges, data.y, masks, n_classes


def _random_split(labels, val_prop, test_prop, seed=3047):
    import random

    random.seed(seed)
    num_class = max(labels) + 1
    label_dict = {i: [] for i in range(num_class)}
    for idx, lab in enumerate(labels):
        label_dict[lab].append(idx)

    idx_train, idx_val, idx_test = [], [], []
    for i in range(num_class):
        random.shuffle(label_dict[i])
        nv = round(val_prop * len(label_dict[i]))
        nt = round(test_prop * len(label_dict[i]))
        idx_val += label_dict[i][:nv]
        idx_test += label_dict[i][nv : nv + nt]
        idx_train += label_dict[i][nv + nt :]

    n = max(max(idx_train), max(idx_val), max(idx_test)) + 1
    train_m = torch.zeros(n, dtype=torch.bool)
    val_m = torch.zeros(n, dtype=torch.bool)
    test_m = torch.zeros(n, dtype=torch.bool)
    train_m[idx_train] = True
    val_m[idx_val] = True
    test_m[idx_test] = True
    return train_m, val_m, test_m


def mask_edges(edge_index, neg_edges, val_prop=0.05, test_prop=0.1):
    """Split edges into train/val/test."""
    n = edge_index.size(1)
    n_val = int(val_prop * n)
    n_test = int(test_prop * n)
    perm = torch.randperm(n)
    edge_index = edge_index[:, perm]

    e_val = edge_index[:, :n_val]
    e_test = edge_index[:, n_val : n_val + n_test]
    e_train = edge_index[:, n_val + n_test :]

    perm_neg = torch.randperm(neg_edges.size(1))
    neg_edges = neg_edges[:, perm_neg]
    neg_val = neg_edges[:, :n_val]
    neg_test = neg_edges[:, n_val : n_val + n_test]
    neg_train = torch.cat([neg_edges, e_val, e_test], dim=-1)

    return (e_train, e_val, e_test), (neg_train, neg_val, neg_test)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(
        description="GraphMoRE LP pretrain",
        epilog="YAML: -c PATH; export: --export-default-yaml / --export-run-yaml PATH (needs pyyaml)",
    )
    p.add_argument("--dataset", type=str, default="Cora")
    p.add_argument(
        "--data_root",
        type=str,
        default="datasets/graphmore",
        help="Same as load_all_datasets: directory with flat Cora.pt, Pubmed.pt, ...",
    )
    p.add_argument("--save_dir", type=str, default="ckpts/graphmore")
    p.add_argument("--seed", type=int, default=3047)

    p.add_argument("--init_curvs", type=float, nargs="+", default=[-3, -1, 0, 1, 3])
    p.add_argument("--hidden_features", type=int, default=64)
    p.add_argument("--embed_features", type=int, default=32)
    p.add_argument("--sample_hops", type=int, nargs="+", default=[2, 3])

    p.add_argument("--r", type=float, default=2.0, help="Fermi-Dirac r")
    p.add_argument("--t", type=float, default=1.0, help="Fermi-Dirac t")
    p.add_argument("--coef_dis", type=float, default=0.1)

    p.add_argument("--lr_riemann", type=float, default=0.01)
    p.add_argument("--lr_gating", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--min_epoch", type=int, default=200)
    p.add_argument("--eval_freq", type=int, default=1)
    p.add_argument("--log_interval", type=int, default=50)

    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--swanlab_project", type=str, default="gfm-toolbox-graphmore")
    add_export_yaml_arguments(p)
    return p, parse_args_with_optional_yaml(p)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p, args = _parse()
    handle_export_args(p, args)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"GraphMoRE LP Pretrain | dataset={args.dataset} device={device}")
    features, edge_index, neg_edges, labels, masks, n_classes = load_dataset(
        args.dataset, args.data_root
    )
    features = features.to(device)
    edge_index = edge_index.to(device)
    neg_edges = neg_edges.to(device)
    in_features = features.size(1)

    use_swanlab = not args.no_swanlab
    if use_swanlab:
        try:
            import swanlab
            swanlab.init(
                project=args.swanlab_project,
                experiment_name=f"graphmore_lp_{args.dataset}",
                config=vars(args),
            )
        except ImportError:
            use_swanlab = False

    pos_edges, neg_edges_split = mask_edges(edge_index, neg_edges)
    e_train, e_val, e_test = pos_edges
    neg_train, neg_val, neg_test = neg_edges_split

    print("Sampling ego subgraphs (this may take a few minutes)...")
    t0 = time.time()
    sampler = EgoGraphSampler(args.sample_hops)
    sub_feats, sub_eis, sub_batches = sampler.sample(features, e_train)
    print(f"Ego-graph sampling done in {time.time() - t0:.1f}s")

    print("Computing shortest path distances...")
    t0 = time.time()
    dis_shortest = compute_shortest_path_distances(e_train)
    print(f"Shortest path computation done in {time.time() - t0:.1f}s")

    model = GraphMoREPrePromptModel(
        in_dim=in_features,
        hidden_dim=args.hidden_features,
        embed_dim=args.embed_features,
        init_curvs=args.init_curvs,
        sample_hops=args.sample_hops,
        r=args.r,
        t=args.t,
        coef_dis=args.coef_dis,
        device=device,
    )

    r_optim = RiemannianAdam(
        model.experts.parameters(), lr=args.lr_riemann, weight_decay=args.weight_decay, stabilize=100
    )
    g_optim = torch.optim.Adam(
        model.gating.parameters(), lr=args.lr_gating, weight_decay=args.weight_decay
    )

    best_ap = 0.0
    best_epoch = 0
    early_stop_count = 0
    best_state = None

    for epoch in range(args.epochs + 1):
        model.train()
        r_optim.zero_grad()
        g_optim.zero_grad()

        neg_sample = neg_train[:, np.random.randint(0, neg_train.shape[1], e_train.shape[1])]
        loss, auc, ap = model(
            features, e_train, e_train, neg_sample,
            sub_feats, sub_eis, sub_batches, dis_shortest,
        )
        loss.backward()
        r_optim.step()
        g_optim.step()

        if epoch % args.log_interval == 0:
            print(f"Epoch {epoch}: loss={loss.item():.4f} AUC={auc:.4f} AP={ap:.4f}")

        if epoch % args.eval_freq == 0:
            model.eval()
            with torch.no_grad():
                _, val_auc, val_ap = model(
                    features, e_train, e_val, neg_val,
                    sub_feats, sub_eis, sub_batches, dis_shortest,
                )
            if val_ap > best_ap:
                best_ap = val_ap
                best_epoch = epoch
                early_stop_count = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                with torch.no_grad():
                    _, test_auc, test_ap = model(
                        features, e_train, e_test, neg_test,
                        sub_feats, sub_eis, sub_batches, dis_shortest,
                    )
            else:
                early_stop_count += 1 if epoch >= args.min_epoch else 0
                if early_stop_count > args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if use_swanlab and epoch % args.log_interval == 0:
                try:
                    import swanlab
                    swanlab.log(
                        {"lp/loss": loss.item(), "lp/val_auc": val_auc, "lp/val_ap": val_ap},
                        step=epoch,
                    )
                except Exception:
                    pass

    print(f"Best epoch={best_epoch} | test AUC={test_auc:.4f} test AP={test_ap:.4f}")

    save_dir = os.path.join(args.save_dir, args.dataset.lower())
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "pretrain_lp.pth")
    payload = {
        "model": best_state,
        "in_features": in_features,
        "hidden_features": args.hidden_features,
        "embed_features": args.embed_features,
        "init_curvs": args.init_curvs,
        "sample_hops": args.sample_hops,
        "dataset": args.dataset,
        "test_auc": test_auc,
        "test_ap": test_ap,
        "best_epoch": best_epoch,
    }
    torch.save(payload, out_path)
    print(f"Saved: {out_path}")

    if use_swanlab:
        try:
            import swanlab
            swanlab.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
