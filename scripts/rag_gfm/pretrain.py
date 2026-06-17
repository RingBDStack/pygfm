#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG-GFM PrePrompt pretraining (baseline-specific; separate from MDGPT pretrain).

PrePrompt: per-domain NodeLevelPrompt + shared GCN + NodeNodeContrastiveLoss.
Model: pygfm.baseline_models.rag_gfm; shared GCN/prompt/loss from pygfm.private.utlis.

Usage:
  python scripts/rag_gfm/pretrain.py --target_dataset Cora
  python scripts/rag_gfm/pretrain.py --datasets Cora,Citeseer,Pubmed --save_dir ckpts/rag_gfm
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from pygfm.public.repo_paths import driver_script_repo_root

ROOT = driver_script_repo_root(__file__)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch
from torch_geometric.data import Data, Batch

from pygfm.private.utlis.domain_alignment import DomainAlignment
from pygfm.public.utils.loss_func import sample_negative_pairs
from pygfm.private.utlis.rag_gfm.motif_builder import load_node_data_for_motif
from pygfm.baseline_models.rag_gfm import PrePromptModel
from pygfm.public.utils import set_seed, early_stopping
from pygfm.public.cli.export_yaml import add_export_yaml_arguments, handle_export_args
from pygfm.public.cli.yaml_config import parse_args_with_optional_yaml


# Default pretrain datasets for RAG-GFM (aligned with common model_node_rag setups)
DEFAULT_PRETRAIN_DATASETS = ["Cora", "Citeseer", "Pubmed", "Photo", "Computers"]


def load_rag_gfm_datasets(data_root: str, dataset_names: list) -> list:
    """Load multiple datasets; returns [{"name": str, "data": Data}, ...]."""
    out = []
    for name in dataset_names:
        data = load_node_data_for_motif(data_root, name)
        if data is None:
            warnings.warn(f"Dataset not found: {name}, skipped", UserWarning)
            continue
        out.append({"name": name, "data": data})
    return out


def main():
    p = argparse.ArgumentParser(description="RAG-GFM PrePrompt pretraining (baseline script)")
    p.add_argument("--data_root", type=str, default="datasets/rag_gfm", help="Graph data root")
    p.add_argument("--save_dir", type=str, default="ckpts/rag_gfm", help="Directory to save checkpoints")
    p.add_argument("--save_name", type=str, default="preprompt.pth", help="Checkpoint filename")
    p.add_argument(
        "--target_dataset",
        type=str,
        default=None,
        help="Leave-one-out target: exclude from pretrain; save to save_dir/{target}/preprompt_{target}.pth",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated pretrain datasets; default DEFAULT_PRETRAIN_DATASETS; optional --target_dataset for LOO",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unify_dim", type=int, default=50, help="PCA unified feature dimension")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--num_neg", type=int, default=50, help="Negative samples per node")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--max_epochs", type=int, default=10000)
    p.add_argument("--prompt_mode", type=str, default="mul", choices=["add", "mul"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=500)
    p.add_argument("--no_swanlab", action="store_true", help="Disable SwanLab logging")
    p.add_argument(
        "--swanlab_project",
        type=str,
        default="gfmtoolbox_raggfm",
        help="SwanLab project name (default RAG-GFM project)",
    )
    p.add_argument("--swanlab_run_name", type=str, default=None, help="SwanLab run name (auto if omitted)")
    add_export_yaml_arguments(p)
    args = parse_args_with_optional_yaml(p)
    handle_export_args(p, args, script_file=Path(__file__))

    # Resolve relative paths against repo root
    if not os.path.isabs(args.data_root):
        args.data_root = str(ROOT / args.data_root)
    if not os.path.isabs(args.save_dir):
        args.save_dir = str(ROOT / args.save_dir)

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Optional SwanLab initialization
    use_swanlab = not args.no_swanlab
    if use_swanlab:
        try:
            import swanlab
            run_name = args.swanlab_run_name or f"pretrain_target_{args.target_dataset or 'all'}"
            swanlab.init(project=args.swanlab_project, experiment_name=run_name, config=vars(args))
        except ImportError:
            use_swanlab = False

    # Resolve pretrain dataset names
    if args.datasets:
        names = [s.strip() for s in args.datasets.split(",") if s.strip()]
    else:
        names = list(DEFAULT_PRETRAIN_DATASETS)
    if args.target_dataset:
        if args.target_dataset not in names:
            names = list(DEFAULT_PRETRAIN_DATASETS)
        if args.target_dataset in names:
            names = [n for n in names if n != args.target_dataset]
        args.save_dir = os.path.join(args.save_dir, args.target_dataset.lower())
        args.save_name = f"preprompt_{args.target_dataset.lower()}.pth"
        print(f"Target (leave-one-out): {args.target_dataset} | Pretrain domains: {names}")
    else:
        print(f"Pretrain domains: {names}")

    # Load data
    sources = load_rag_gfm_datasets(args.data_root, names)
    if not sources:
        raise RuntimeError("No datasets loaded; check --data_root and --datasets")
    ordered_names = [s["name"] for s in sources]

    # Feature alignment + merged graph
    source_list = []
    aligners = []
    for idx, s in enumerate(sources):
        feat = s["data"].x.cpu().numpy()
        aligner = DomainAlignment(n_components=args.unify_dim)
        aligner.fit(feat)
        aligners.append(aligner)
        aligned = torch.from_numpy(aligner.transform(feat)).float()
        d = Data(
            x=aligned,
            edge_index=s["data"].edge_index,
            y=getattr(s["data"], "y", None),
        )
        d.domain_id = torch.full((aligned.size(0),), idx, dtype=torch.long)
        source_list.append(d)

    big_batch = Batch.from_data_list(source_list).to(device)
    tuples = sample_negative_pairs(
        big_batch.edge_index,
        big_batch.num_nodes,
        num_neg=args.num_neg,
        seed=args.seed,
    ).to(device)
    # batch vector = domain id
    domain_batch = big_batch.batch

    # Model and optimizer (RAG-GFM PrePrompt; shared GCN etc.)
    model = PrePromptModel(
        input_dim=args.unify_dim,
        hidden_dim=args.hidden_dim,
        num_domains=len(sources),
        num_layers=args.num_layers,
        prompt_mode=args.prompt_mode,
        temperature=args.temperature,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    # Pretraining loop
    print(">> RAG-GFM PrePrompt pretraining ...")
    best_loss = 1e9
    cnt_wait = 0
    for epoch in range(args.max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = model(
            big_batch.x,
            big_batch.edge_index,
            domain_batch,
            tuples,
        )
        loss.backward()
        optimizer.step()

        should_stop, best_loss, cnt_wait = early_stopping(
            loss.item(), best_loss, cnt_wait, args.patience
        )
        if epoch % args.log_interval == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")
        if use_swanlab:
            try:
                swanlab.log({"loss": loss.item(), "epoch": epoch}, step=epoch)
            except Exception:
                pass
        if should_stop:
            print("Early stopping.")
            break

    # Save
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, args.save_name)
    save_dict = {
        "model": model.state_dict(),
        "unify_dim": args.unify_dim,
        "hidden_dim": args.hidden_dim,
        "num_domains": len(sources),
        "ordered_names": ordered_names,
        "prompt_mode": args.prompt_mode,
    }
    if args.target_dataset:
        save_dict["target"] = args.target_dataset
    torch.save(save_dict, ckpt_path)
    print(f"Saved: {ckpt_path}")
    if use_swanlab:
        try:
            swanlab.log({"best_loss": best_loss, "final_epoch": epoch})
        except Exception:
            pass

    try:
        import joblib
        aligner_path = os.path.join(args.save_dir, "aligners.pkl")
        joblib.dump({"aligners": aligners, "ordered_names": ordered_names}, aligner_path)
        print(f"Saved aligners: {aligner_path}")
    except Exception:
        pass

    print("RAG-GFM pretrain done.")


if __name__ == "__main__":
    main()
