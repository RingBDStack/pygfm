#!/usr/bin/env python
"""
GCoT PrePrompt pretraining (GCN + NodeNodeContrastiveLoss only); saves ckpt for downstream.
"""
from __future__ import annotations

import argparse
import os
import warnings

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore")

import torch
from torch_geometric.data import Data, Batch
import sys
from pathlib import Path
from pygfm.public.repo_paths import driver_script_repo_root

ROOT = driver_script_repo_root(__file__)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pygfm.private.utlis.domain_alignment import DomainAlignment
from pygfm.private.utlis.loss_calculation import sample_negative_pairs
from pygfm.baseline_models import GCoTPrePromptModel
from pygfm.public.utils.runtime import set_seed, load_all_datasets, early_stopping
from pygfm.public.cli.yaml_config import parse_args_with_optional_yaml
from pygfm.public.cli.export_yaml import add_export_yaml_arguments, handle_export_args


def main():
    p = argparse.ArgumentParser(
        description="GCoT PrePrompt pretraining (GCN + LP)",
        epilog="YAML: -c PATH; export: --export-default-yaml / --export-run-yaml PATH (needs pyyaml)",
    )
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="ckpts/gcot", help="Base save directory")
    p.add_argument("--save_name", type=str, default="preprompt.pth")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unify_dim", type=int, default=50)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--num_neg", type=int, default=50)
    p.add_argument("--num_domains", type=int, default=4)
    p.add_argument("--datasets", type=str, default=None)
    p.add_argument("--target", type=str, default=None)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--max_epochs", type=int, default=10000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=500)
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--swanlab_project", type=str, default="gfm-toolbox-gcot")
    p.add_argument("--swanlab_run_name", type=str, default=None)
    add_export_yaml_arguments(p)
    args = parse_args_with_optional_yaml(p)
    handle_export_args(p, args)

    # --- Note: removed early args.save_dir rewrite for args.target (handled below) ---

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root or os.environ.get("GFM_DATA_ROOT", "datasets/gcot")

    all_raw = load_all_datasets(data_root=data_root)
    name2item = {d["name"]: d for d in all_raw}

    # --- Unified save path layout ---
    if args.target:
        # 1. Leave-one-out
        if args.target not in name2item:
            raise ValueError(f"Unknown target '{args.target}'.")
        sources = [d for d in all_raw if d["name"] != args.target]
        sub_folder = args.target.lower()
    elif args.datasets:
        # 2. Explicit dataset list
        wanted = [x.strip() for x in args.datasets.split(",") if x.strip()]
        sources = [name2item[n] for n in wanted]
        sub_folder = "_".join([s["name"].lower() for s in sources])
    else:
        # 3. Default: first N domains
        sources = all_raw[: args.num_domains]
        sub_folder = f"top_{args.num_domains}"

    # --- Force save_dir from sub_folder ---
    ordered_names = [s["name"] for s in sources]
    args.save_dir = os.path.join("ckpts/gcot", sub_folder)
    args.save_name = "preprompt.pth"  # fixed name; directory disambiguates runs
    
    # Ensure directory exists for torch.save
    os.makedirs(args.save_dir, exist_ok=True)

    print(f">> Pretrain domains: {ordered_names}")
    print(f">> Final Save Dir: {args.save_dir}")

    # 1. Feature alignment + merged graph
    source_list = []
    aligners = []
    for idx, s in enumerate(sources):
        feat = s["ds"][0].x.numpy()
        aligner = DomainAlignment(n_components=args.unify_dim)
        aligner.fit(feat)
        aligners.append(aligner)
        aligned = torch.from_numpy(aligner.transform(feat)).float()
        d = Data(x=aligned, edge_index=s["ds"][0].edge_index, y=s["ds"][0].y)
        d.domain_id = torch.full((aligned.size(0),), idx, dtype=torch.long)
        source_list.append(d)

    big_batch = Batch.from_data_list(source_list).to(device)
    tuples = sample_negative_pairs(
        big_batch.edge_index, big_batch.num_nodes, num_neg=args.num_neg, seed=args.seed
    ).to(device)

    # 2. Model and optimizer
    model = GCoTPrePromptModel(
        input_dim=args.unify_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        temperature=args.temperature,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    use_swanlab = not args.no_swanlab
    if use_swanlab:
        try:
            import swanlab
            # Sync experiment name with layout
            run_name = args.swanlab_run_name or f"pretrain_{sub_folder}"
            swanlab.init(project=args.swanlab_project, experiment_name=run_name, config=vars(args))
        except ImportError:
            use_swanlab = False

    # 3. Pretraining loop
    print(">> GCoT PrePrompt pretraining ...")
    best_loss = 1e9
    cnt_wait = 0
    for epoch in range(args.max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = model(big_batch.x, big_batch.edge_index, tuples)
        loss.backward()
        optimizer.step()
        
        should_stop, best_loss, cnt_wait = early_stopping(loss.item(), best_loss, cnt_wait, args.patience)
        
        if use_swanlab:
            try:
                swanlab.log({"loss": loss.item(), "epoch": epoch}, step=epoch)
            except Exception:
                pass
        if epoch % args.log_interval == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")
        if should_stop:
            print("Early stopping.")
            break

    # 4. Save checkpoint (model + aligners)
    ckpt_path = os.path.join(args.save_dir, args.save_name)
    save_dict = {
        "model": model.state_dict(),
        "unify_dim": args.unify_dim,
        "hidden_dim": args.hidden_dim,
        "num_domains": len(sources),
        "num_layers": args.num_layers,
        "ordered_names": ordered_names,
    }
    if args.target:
        save_dict["target"] = args.target
    
    torch.save(save_dict, ckpt_path)
    print(f"Saved Checkpoint: {ckpt_path}")

    try:
        import joblib
        aligner_path = os.path.join(args.save_dir, "aligners.pkl")
        joblib.dump({"aligners": aligners, "ordered_names": ordered_names}, aligner_path)
        print(f"Saved Aligner: {aligner_path}")
    except Exception:
        pass
    print("Pretrain done.")


if __name__ == "__main__":
    main()