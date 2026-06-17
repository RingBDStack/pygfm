#!/usr/bin/env python
"""
BRIDGE DownPrompt few-shot node classification: load PrePrompt, PCA target domain, prototype init + MoE/spectral-regularized finetune.

Examples:
  python scripts/bridge/finetune.py --dataset Cora --k_shot 1 \\
    --ckpt ckpts/bridge/cora_run/preprompt_cora.pth --split_id 0 --no_swanlab

  # Match pretrain: reuse PCA from aligners.pkl next to ckpt when --dataset is in ordered_names; else fit fresh (e.g. leave-one-out target).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from pygfm.public.repo_paths import driver_script_repo_root

import numpy as np
import torch
import torch.nn.functional as F
ROOT = driver_script_repo_root(__file__)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pygfm.private.utlis.domain_alignment import DomainAlignment
from pygfm.baseline_models.bridge import BridgeDownPromptModel
from pygfm.public.utils.runtime import compute_spectral_components, load_single_graph_dataset, set_seed
from pygfm.public.cli.yaml_config import parse_args_with_optional_yaml
from pygfm.public.cli.export_yaml import add_export_yaml_arguments, handle_export_args
from pygfm.public.cli.default_ckpt import resolve_preprompt_ckpt


def _aligned_features_from_ckpt_dir(
    ckpt_path: str, x_np: np.ndarray, aligned_dim: int, dataset: str
) -> np.ndarray:
    """Reuse source aligner from aligners.pkl when dataset matches; else fit PCA (typical for leave-one-out target)."""
    p = Path(ckpt_path).resolve().parent / "aligners.pkl"
    if p.is_file():
        try:
            import joblib

            blob = joblib.load(p)
            names = blob.get("ordered_names") or []
            aligners = blob.get("aligners") or []
            if dataset in names:
                i = names.index(dataset)
                if i < len(aligners):
                    return aligners[i].transform(x_np)
        except Exception:
            pass
    aligner = DomainAlignment(n_components=aligned_dim)
    aligner.fit(x_np)
    return aligner.transform(x_np)


def _parse():
    p = argparse.ArgumentParser(
        description="BRIDGE DownPrompt few-shot node finetune",
        epilog="YAML: -c PATH; export: --export-default-yaml / --export-run-yaml PATH (needs pyyaml)",
    )
    p.add_argument("--dataset", type=str, default="Cora")
    p.add_argument("--k_shot", type=int, default=1, choices=[1, 5])
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="PrePrompt checkpoint; if omitted, try ckpts/bridge/preprompt.pth and dataset subdirs; "
        "pass explicit ckpt if pretrain used a custom --datasets subfolder",
    )
    p.add_argument("--downstream_root", type=str, default="downstream_data/bridge")
    p.add_argument("--splits_path", type=str, default=None)
    p.add_argument("--split_id", type=int, default=0)
    p.add_argument("--task_num", type=int, default=0, help="If >0, run only first task_num splits")
    p.add_argument("--data_root", type=str, default="datasets/bridge")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_epochs", type=int, default=300)
    p.add_argument("--lr_router", type=float, default=1e-3)
    p.add_argument("--lr_proto", type=float, default=1e-2)
    p.add_argument("--weight_ent", type=float, default=0.1)
    p.add_argument("--weight_spec", type=float, default=0.01)
    p.add_argument("--test_reserve", type=int, default=1000)
    p.add_argument("--spectral_k", type=int, default=10)
    p.add_argument("--row_norm", action="store_true")
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--swanlab_project", type=str, default="gfm-toolbox-bridge")
    add_export_yaml_arguments(p)
    return p, parse_args_with_optional_yaml(p)


def _load_graph(data_root: str, dataset: str):
    return load_single_graph_dataset(data_root, dataset)


def main():
    p, args = _parse()
    handle_export_args(p, args)
    args.ckpt = resolve_preprompt_ckpt(ROOT, "bridge", args.dataset, args.ckpt)
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    use_swanlab = not args.no_swanlab
    if use_swanlab:
        try:
            import swanlab
            swanlab.init(project=args.swanlab_project, experiment_name=f"bridge_ft_{args.dataset}_{args.k_shot}shot")
        except ImportError:
            use_swanlab = False

    ckpt = torch.load(args.ckpt, map_location=device)
    aligned_dim = ckpt["aligned_dim"]
    hidden_dim = ckpt["hidden_dim"]
    num_sources = ckpt["num_sources"]

    data, num_classes = _load_graph(args.data_root, args.dataset)
    x_np = data.x.cpu().numpy().astype(np.float64)
    if args.row_norm:
        rs = x_np.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        x_np = x_np / rs
    x_aligned = _aligned_features_from_ckpt_dir(args.ckpt, x_np, aligned_dim, args.dataset)
    x = torch.from_numpy(x_aligned.astype(np.float32)).to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)

    if args.splits_path:
        spath = args.splits_path
    else:
        spath = os.path.join(args.downstream_root, args.dataset, f"{args.k_shot}shot", "splits.pt")
    down = torch.load(spath, map_location="cpu")
    splits = down["splits"]
    n = len(splits)
    split_ids = list(range(min(args.task_num, n))) if args.task_num > 0 else [args.split_id]
    if args.task_num <= 0 and not (0 <= args.split_id < n):
        raise IndexError(f"split_id {args.split_id} not in [0,{n-1}]")

    test_start = max(0, y.size(0) - args.test_reserve)
    test_idx = torch.arange(test_start, y.size(0), device=device)
    test_y = y[test_idx]

    eivec, eival = compute_spectral_components(edge_index, y.size(0), device=device, k=args.spectral_k)

    acc_list = []
    for sid in split_ids:
        split = splits[sid]
        train_idx = torch.tensor(split["indices"], dtype=torch.long, device=device)
        train_y = torch.tensor(split["labels"], dtype=torch.long, device=device)
        train_mask = torch.zeros(y.size(0), dtype=torch.bool, device=device)
        train_mask[train_idx] = True

        model = BridgeDownPromptModel(
            aligned_dim=aligned_dim,
            hidden_dim=hidden_dim,
            num_sources=num_sources,
            num_classes=num_classes,
            domain_name=args.dataset,
            dropout=0.0,
            device=device,
        )
        model.load_preprompt_checkpoint(ckpt, strict=False)
        model.freeze_preprompt_parts()

        model.eval()
        with torch.no_grad():
            h0 = model.embed_backbone_unmasked(x, edge_index)
            protos = []
            for c in range(num_classes):
                m = (y == c) & train_mask
                protos.append(h0[m].mean(0) if m.any() else torch.zeros(hidden_dim, device=device))
            model.prototypes.data.copy_(torch.stack(protos))

        opt = torch.optim.Adam(
            [
                {"params": model.routing_net.parameters(), "lr": args.lr_router},
                {"params": [model.graph_prompt], "lr": args.lr_router},
                {"params": [model.prototypes], "lr": args.lr_proto},
            ],
            lr=args.lr_router,
            weight_decay=1e-5,
        )

        best = 0.0
        for _ in range(args.max_epochs):
            model.train()
            opt.zero_grad()
            _, logits, reg, ent = model(x, edge_index, eivec=eivec, eival=eival)
            loss = F.cross_entropy(logits[train_mask], y[train_mask]) + args.weight_ent * ent + args.weight_spec * reg
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            _, logits, _, _ = model(x, edge_index, eivec=eivec, eival=eival)
            pred = logits[test_idx].argmax(1)
            acc = (pred == test_y).float().mean().item()
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
