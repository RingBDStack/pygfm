#!/usr/bin/env python
"""
GraphMoRE DownPrompt node classification: load LP weights -> joint finetune experts + gating + classifier.

Steps:
1. Load LP checkpoint (Riemannian experts + gating)
2. Build DownPromptModel and load weights
3. Sample ego subgraphs + shortest paths on full graph
4. Joint train: CE + distortion
   - Riemannian params -> RiemannianAdam
   - Gating + classifier -> Adam
5. Report Weighted-F1, Macro-F1

Examples:
  python scripts/graphmore/finetune.py --dataset Cora --backbone gcn
  python scripts/graphmore/finetune.py --dataset airport --backbone sage --embed_features 16
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

from pygfm.baseline_models.graphmore import GraphMoREDownPromptModel
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

from sklearn.metrics import f1_score


# ---------------------------------------------------------------------------
# Data loading (same as pretrain.py)
# ---------------------------------------------------------------------------

def load_dataset(name: str, root: str = "datasets/graphmore"):
    """Same as ``load_all_datasets`` / flat ``Cora.pt``."""
    import random

    data, n_classes = load_single_graph_dataset(root, name)
    if all(getattr(data, k, None) is not None for k in ("train_mask", "val_mask", "test_mask")):
        masks = (data.train_mask, data.val_mask, data.test_mask)
    else:
        labels_list = data.y.tolist()
        num_class = max(labels_list) + 1
        label_dict = {i: [] for i in range(num_class)}
        for idx, lab in enumerate(labels_list):
            label_dict[lab].append(idx)
        random.seed(3047)
        idx_train, idx_val, idx_test = [], [], []
        for i in range(num_class):
            random.shuffle(label_dict[i])
            nv = round(0.15 * len(label_dict[i]))
            nt = round(0.15 * len(label_dict[i]))
            idx_val += label_dict[i][:nv]
            idx_test += label_dict[i][nv : nv + nt]
            idx_train += label_dict[i][nv + nt :]
        n = data.x.size(0)
        train_m, val_m, test_m = (
            torch.zeros(n, dtype=torch.bool),
            torch.zeros(n, dtype=torch.bool),
            torch.zeros(n, dtype=torch.bool),
        )
        train_m[idx_train] = True
        val_m[idx_val] = True
        test_m[idx_test] = True
        masks = (train_m, val_m, test_m)
    return data.x, data.edge_index.long(), data.y, masks, n_classes


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    preds = logits[mask].argmax(dim=-1).cpu()
    trues = labels[mask].cpu()
    acc = (preds == trues).float().mean().item()
    wf1 = f1_score(trues, preds, average="weighted")
    mf1 = f1_score(trues, preds, average="macro")
    return acc, wf1, mf1


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(
        description="GraphMoRE NC finetune",
        epilog="YAML: -c PATH; export: --export-default-yaml / --export-run-yaml PATH (needs pyyaml)",
    )
    p.add_argument("--dataset", type=str, default="Cora")
    p.add_argument(
        "--data_root",
        type=str,
        default="datasets/graphmore",
        help="Same as load_all_datasets: flat Cora.pt, etc.",
    )
    p.add_argument("--ckpt", type=str, default=None, help="LP pretrained checkpoint")
    p.add_argument("--seed", type=int, default=3047)
    p.add_argument("--exp_iters", type=int, default=10)

    p.add_argument("--init_curvs", type=float, nargs="+", default=[-3, -1, 0, 1, 3])
    p.add_argument("--hidden_features", type=int, default=64)
    p.add_argument("--embed_features", type=int, default=32)
    p.add_argument("--sample_hops", type=int, nargs="+", default=[2, 3])

    p.add_argument("--backbone", type=str, default="gcn", choices=["gcn", "gat", "sage"])
    p.add_argument("--hidden_cls", type=int, default=32)
    p.add_argument("--n_layers_cls", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--drop_edge_cls", type=float, default=0.0)
    p.add_argument("--drop_feat_cls", type=float, default=0.0)
    p.add_argument("--coef_dis", type=float, default=1e-4)

    p.add_argument("--lr_riemann", type=float, default=0.01)
    p.add_argument("--lr_gating", type=float, default=0.01)
    p.add_argument("--lr_cls", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--min_epoch", type=int, default=200)
    p.add_argument("--eval_freq", type=int, default=1)
    p.add_argument("--log_interval", type=int, default=100)

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

    features, edge_index, labels, masks, n_classes = load_dataset(
        args.dataset, args.data_root
    )
    features = features.to(device)
    edge_index = edge_index.to(device)
    labels = labels.to(device)
    train_mask, val_mask, test_mask = [m.to(device) for m in masks]
    in_features = features.size(1)

    ckpt_path = args.ckpt or os.path.join(
        "ckpts/graphmore", args.dataset.lower(), "pretrain_lp.pth"
    )

    use_swanlab = not args.no_swanlab
    if use_swanlab:
        try:
            import swanlab
            swanlab.init(
                project=args.swanlab_project,
                experiment_name=f"graphmore_nc_{args.dataset}_{args.backbone}",
                config=vars(args),
            )
        except ImportError:
            use_swanlab = False

    print("Sampling ego subgraphs...")
    t0 = time.time()
    sampler = EgoGraphSampler(args.sample_hops)
    sub_feats, sub_eis, sub_batches = sampler.sample(features, edge_index)
    print(f"Ego-graph sampling done in {time.time() - t0:.1f}s")

    print("Computing shortest path distances...")
    t0 = time.time()
    dis_shortest = compute_shortest_path_distances(edge_index)
    print(f"Shortest path computation done in {time.time() - t0:.1f}s")

    all_accs, all_wf1s, all_mf1s = [], [], []

    for exp_iter in range(args.exp_iters):
        print(f"\n--- Experiment iter {exp_iter + 1}/{args.exp_iters} ---")

        model = GraphMoREDownPromptModel(
            in_features=in_features,
            embed_features=args.embed_features,
            init_curvs=args.init_curvs,
            sample_hops=args.sample_hops,
            hidden_features_expert=args.hidden_features,
            backbone=args.backbone,
            hidden_features_cls=args.hidden_cls,
            num_classes=n_classes,
            n_layers_cls=args.n_layers_cls,
            n_heads=args.n_heads,
            drop_edge_cls=args.drop_edge_cls,
            drop_feat_cls=args.drop_feat_cls,
            coef_dis=args.coef_dis,
            device=device,
        )

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            loaded = model.load_preprompt_checkpoint(ckpt)
            print(f"Loaded {len(loaded)} keys from {ckpt_path}")
        else:
            print(f"Warning: checkpoint {ckpt_path} not found, training from scratch")

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

        best_val_acc = 0.0
        best_epoch = 0
        test_acc, test_wf1, test_mf1 = 0.0, 0.0, 0.0
        early_stop_count = 0

        for epoch in range(args.epochs + 1):
            model.train()
            r_optim.zero_grad()
            e_optim.zero_grad()

            logits, loss_dis = model(
                features, edge_index, sub_feats, sub_eis, sub_batches, dis_shortest
            )
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])
            loss = loss + args.coef_dis * loss_dis
            loss.backward()
            r_optim.step()
            e_optim.step()

            if epoch % args.eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    eval_logits = model.predict(
                        features, edge_index, sub_feats, sub_eis, sub_batches
                    )
                val_acc, val_wf1, val_mf1 = evaluate(eval_logits, labels, val_mask)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    early_stop_count = 0
                    test_acc, test_wf1, test_mf1 = evaluate(
                        eval_logits, labels, test_mask
                    )
                else:
                    if epoch >= args.min_epoch:
                        early_stop_count += 1
                    if early_stop_count > args.patience:
                        break

            if epoch % args.log_interval == 0:
                print(
                    f"  Epoch {epoch}: loss={loss.item():.4f} "
                    f"val_acc={best_val_acc:.4f} test_wf1={test_wf1:.4f}"
                )

        print(
            f"  Best epoch={best_epoch} | "
            f"test_acc={test_acc:.4f} test_wf1={test_wf1:.4f} test_mf1={test_mf1:.4f}"
        )
        all_accs.append(test_acc)
        all_wf1s.append(test_wf1)
        all_mf1s.append(test_mf1)

        if use_swanlab:
            try:
                import swanlab
                swanlab.log(
                    {
                        "nc/test_acc": test_acc,
                        "nc/test_wf1": test_wf1,
                        "nc/test_mf1": test_mf1,
                    },
                    step=exp_iter,
                )
            except Exception:
                pass

    print(f"\n=== Results over {args.exp_iters} runs ===")
    print(f"  Accuracy:    {np.mean(all_accs)*100:.2f} ± {np.std(all_accs)*100:.2f}")
    print(f"  Weighted-F1: {np.mean(all_wf1s)*100:.2f} ± {np.std(all_wf1s)*100:.2f}")
    print(f"  Macro-F1:    {np.mean(all_mf1s)*100:.2f} ± {np.std(all_mf1s)*100:.2f}")

    if use_swanlab:
        try:
            import swanlab
            swanlab.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
