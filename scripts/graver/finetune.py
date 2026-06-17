#!/usr/bin/env python
"""
GRAVER DownPrompt few-shot node classification.

Supports ``experiment_type``:
- ``standard``: on-the-fly graphon on target graph + PCA-aligned features
- ``cross-dataset``: external graphon files + GRAVER export features (dual-trial eval, early stopping)

Examples:
  python scripts/graver/finetune.py --dataset Cora --k_shot 1 \\
    --ckpt ckpts/graver/cora/preprompt.pkl --experiment_type cross-dataset
"""
from __future__ import annotations

import argparse
import os
import sys
from pygfm.public.repo_paths import driver_script_repo_root

import numpy as np
import torch
import torch.nn as nn

ROOT = driver_script_repo_root(__file__)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from pygfm.private.utlis.domain_alignment import DomainAlignment
from pygfm.baseline_models.graver import (
    GRAVERDownPromptModel,
    edge_index_to_sparse_adj,
    estimate_graphon,
    load_cross_dataset_graphons,
    load_graver_node_features,
    load_graver_preprompt_checkpoint,
    make_forward_rng,
)
from pygfm.public.utils.runtime import load_single_graph_dataset, set_seed
from pygfm.public.cli.yaml_config import parse_args_with_optional_yaml
from pygfm.public.cli.export_yaml import add_export_yaml_arguments, handle_export_args
from pygfm.public.cli.default_ckpt import resolve_preprompt_ckpt


def _parse():
    p = argparse.ArgumentParser(
        description="GRAVER DownPrompt few-shot node finetune",
        epilog="YAML: -c PATH; export: --export-default-yaml / --export-run-yaml PATH (needs pyyaml)",
    )
    p.add_argument("--dataset", type=str, default="Cora")
    p.add_argument("--k_shot", type=int, default=1, choices=[1, 5])
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="PrePrompt checkpoint (.pth / .pkl); auto-detect under ckpts/graver/ if omitted",
    )
    p.add_argument(
        "--experiment_type",
        type=str,
        default="standard",
        choices=["standard", "cross-dataset"],
        help="standard: on-the-fly graphon; cross-dataset: load external graphon + paper protocol",
    )
    p.add_argument("--downstream_root", type=str, default="downstream_data/graver")
    p.add_argument("--splits_path", type=str, default=None)
    p.add_argument("--split_id", type=int, default=0)
    p.add_argument("--task_num", type=int, default=0, help="If >0, run splits 0 .. task_num-1")
    p.add_argument("--data_root", type=str, default="datasets/graver")
    p.add_argument("--graphon_root", type=str, default="datasets/graver/graphon")
    p.add_argument("--seed", type=int, default=39)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=0, help="Early stopping patience (0 = disabled)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_entropy", type=float, default=0.2)
    p.add_argument("--test_reserve", type=int, default=1000)
    p.add_argument("--gen_num_nodes", type=int, default=10)
    p.add_argument("--combine_type", type=str, default="mul")
    p.add_argument("--graphon_resolution", type=int, default=10)
    p.add_argument("--row_norm", action="store_true")
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--swanlab_project", type=str, default="gfm-toolbox-graver")
    add_export_yaml_arguments(p)
    return p, parse_args_with_optional_yaml(p)


def _resolve_splits_path(args) -> str:
    if args.splits_path:
        return args.splits_path
    return os.path.join(args.downstream_root, args.dataset, f"{args.k_shot}shot", "splits.pt")


def _dual_trial_acc_cross(
    model,
    x,
    adj,
    test_idx,
    seq,
    graphon_list,
    test_y,
    *,
    seed: int,
    split_id: int,
) -> float:
    accs: list[float] = []
    for phase in (0, 1):
        rng = make_forward_rng(seed, split_id, 0, phase)
        with torch.no_grad():
            probs, _ = model(
                x, adj, test_idx, seq[test_idx], graphon_list, rng=rng,
            )
            pred = probs.argmax(1)
            accs.append((pred == test_y).float().mean().item())
    return float(sum(accs) / len(accs))


def _finetune_standard_split(
    *,
    model,
    x,
    edge_index,
    train_idx,
    train_y,
    test_idx,
    test_y,
    graphon_list,
    args,
    sid: int,
) -> float:
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    xent = nn.CrossEntropyLoss()
    best_loss = float("inf")
    stale = 0

    for epoch in range(args.max_epochs):
        model.train()
        opt.zero_grad()
        probs, entropy = model.forward_standard(
            x, edge_index, train_idx, graphon_list, train_y, train=True,
        )
        loss = xent(probs, train_y) + args.lambda_entropy * entropy.mean()
        loss.backward()
        opt.step()

        loss_val = float(loss.detach())
        if args.patience > 0:
            if loss_val + 1e-8 < best_loss:
                best_loss = loss_val
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break

    model.eval()
    with torch.no_grad():
        probs_test, _ = model.forward_standard(x, edge_index, test_idx, graphon_list)
        pred = probs_test.argmax(1)
        return (pred == test_y).float().mean().item()


def _finetune_cross_split(
    *,
    model,
    x,
    adj,
    edge_index,
    train_idx,
    train_y,
    test_idx,
    test_y,
    graphon_list,
    args,
    sid: int,
) -> float:
    seq = model.embed_backbone(x, edge_index)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    xent = nn.CrossEntropyLoss()
    best_loss = float("inf")
    stale = 0

    for epoch in range(args.max_epochs):
        model.train()
        opt.zero_grad()
        rng = make_forward_rng(args.seed, sid, epoch, 0)
        probs, entropy = model(
            x,
            adj,
            train_idx,
            seq[train_idx],
            graphon_list,
            train_y,
            train=True,
            rng=rng,
        )
        loss = xent(probs, train_y) + args.lambda_entropy * entropy.mean()
        loss.backward()
        opt.step()

        loss_val = float(loss.detach())
        if args.patience > 0:
            if loss_val + 1e-8 < best_loss:
                best_loss = loss_val
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break

    model.eval()
    return _dual_trial_acc_cross(
        model, x, adj, test_idx, seq, graphon_list, test_y,
        seed=args.seed, split_id=sid,
    )


def main():
    p, args = _parse()
    handle_export_args(p, args)
    args.ckpt = resolve_preprompt_ckpt(ROOT, "graver", args.dataset, args.ckpt)
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    use_swanlab = not args.no_swanlab
    if use_swanlab:
        try:
            import swanlab

            swanlab.init(
                project=args.swanlab_project,
                experiment_name=f"graver_ft_{args.dataset}_{args.k_shot}shot",
            )
        except ImportError:
            use_swanlab = False

    ckpt = load_graver_preprompt_checkpoint(
        args.ckpt,
        target_dataset=args.dataset,
        experiment_type=args.experiment_type,
        map_location=device,
    )
    num_sources = int(ckpt["num_sources"])

    data, num_classes = load_single_graph_dataset(args.data_root, args.dataset)
    y = data.y.to(device)

    if args.experiment_type == "cross-dataset":
        x = load_graver_node_features(data, device)
        edge_index = data.edge_index.to(device)
        adj = edge_index_to_sparse_adj(edge_index, x.size(0))
        graphon_list, num_labels_list = load_cross_dataset_graphons(
            args.graphon_root, args.dataset, args.experiment_type,
        )
    else:
        x_np = data.x.cpu().numpy().astype(np.float64)
        if args.row_norm:
            rs = x_np.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1.0
            x_np /= rs
        input_dim = int(ckpt["input_dim"])
        aligner = DomainAlignment(n_components=input_dim)
        aligner.fit(x_np)
        x = torch.from_numpy(aligner.transform(x_np).astype(np.float32)).to(device)
        edge_index = data.edge_index.to(device)
        adj = None
        graphon_per_class = estimate_graphon(
            edge_index, y, x.size(0), args.graphon_resolution,
        )
        num_labels_list = [len(graphon_per_class)] * num_sources
        graphon_list = [graphon_per_class] * num_sources

    spath = _resolve_splits_path(args)
    down = torch.load(spath, map_location="cpu", weights_only=False)
    splits = down["splits"]
    n_splits = len(splits)
    split_ids = (
        list(range(min(args.task_num, n_splits)))
        if args.task_num > 0
        else [args.split_id]
    )

    test_start = max(0, y.size(0) - args.test_reserve)
    test_idx = torch.arange(test_start, y.size(0), device=device)
    test_y = y[test_idx]

    acc_list: list[float] = []
    for sid in split_ids:
        split = splits[sid]
        train_idx = torch.tensor(split["indices"], dtype=torch.long, device=device)
        train_y = torch.tensor(split["labels"], dtype=torch.long, device=device)

        model = GRAVERDownPromptModel.for_finetune(
            ckpt,
            num_labels_list,
            num_classes,
            seed=args.seed,
            gen_num_nodes=args.gen_num_nodes,
            combine_type=args.combine_type,
            device=device,
        )

        if args.experiment_type == "cross-dataset":
            acc = _finetune_cross_split(
                model=model,
                x=x,
                adj=adj,
                edge_index=edge_index,
                train_idx=train_idx,
                train_y=train_y,
                test_idx=test_idx,
                test_y=test_y,
                graphon_list=graphon_list,
                args=args,
                sid=sid,
            )
        else:
            acc = _finetune_standard_split(
                model=model,
                x=x,
                edge_index=edge_index,
                train_idx=train_idx,
                train_y=train_y,
                test_idx=test_idx,
                test_y=test_y,
                graphon_list=graphon_list,
                args=args,
                sid=sid,
            )

        acc_list.append(acc)
        print(f"[{args.dataset}] {args.k_shot}-shot split {sid} test acc: {acc:.4f}")
        if use_swanlab:
            try:
                import swanlab

                swanlab.log({f"acc_split_{sid}": acc})
            except Exception:
                pass

    if len(acc_list) > 1:
        t = torch.tensor(acc_list)
        print(f"mean {t.mean():.4f} std {t.std():.4f} (n={len(acc_list)})")


if __name__ == "__main__":
    main()
