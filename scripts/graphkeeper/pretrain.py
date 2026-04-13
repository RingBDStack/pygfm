#!/usr/bin/env python
"""
GraphKeeper domain-incremental pretraining.

1. Walk domains in order (ordered domain learning).
2. ``set_incremental_optimizer`` freezes old-domain prompt params (knowledge preservation).
3. Per-domain early stopping; weights accumulate across domains.

Save path (BRIDGE-style): ckpts/graphkeeper/{datasets}/preprompt.pth (or preprompt_{target}.pth for leave-one-out).
  --datasets is the output subdir name; source domains via --domains (comma-separated).
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore")

import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _abs_project_path(p: str) -> str:
    p = os.path.expanduser(p.strip())
    if not p: return str(ROOT)
    if os.path.isabs(p): return p
    return str(ROOT / p)

from pygfm.private.utlis.domain_alignment import DomainAlignment
from pygfm.baseline_models.graphkeeper import GraphKeeperPrePromptModel, set_incremental_optimizer
from pygfm.public.utils.runtime import early_stopping, load_all_datasets, set_seed
from pygfm.public.cli.yaml_config import parse_args_with_optional_yaml
from pygfm.public.cli.export_yaml import add_export_yaml_arguments, handle_export_args


def main():
    print(f"[graphkeeper/incremental_pretrain] pid={os.getpid()} start")
    p = argparse.ArgumentParser(
        description="GraphKeeper Domain-Incremental Pre-training",
        epilog="YAML: -c PATH; export: --export-default-yaml / --export-run-yaml PATH (needs pyyaml)",
    )
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Override default save root; default is always ckpts/graphkeeper/{datasets}/",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Output subdir under ckpts/graphkeeper/{datasets}/; "
        "if omitted, use lowercase --target when set, else default",
    )
    p.add_argument("--save_name", type=str, default="preprompt.pth")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unify_dim", type=int, default=50)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--num_domains", type=int, default=4)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument(
        "--domains",
        type=str,
        default=None,
        help="Source domains for incremental pretrain, comma-separated (--target / --dataset priority in code)",
    )
    p.add_argument("--target", type=str, default=None)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--max_epochs", type=int, default=10000)
    p.add_argument("--prompt_mode", type=str, default="mul", choices=["add", "mul"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--contrastive_weight", type=float, default=0.0)
    p.add_argument("--lp_max_edges", type=int, default=8192)
    p.add_argument("--log_interval", type=int, default=200)
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--swanlab_project", type=str, default="gfm-toolbox-graphkeeper")
    add_export_yaml_arguments(p)
    args = parse_args_with_optional_yaml(p)
    handle_export_args(p, args)

    def _safe_subdir(name: str) -> str:
        s = name.strip().replace(os.sep, "_").replace("/", "_").replace("\\", "_")
        return s if s else "default"

    if args.datasets:
        sub = _safe_subdir(args.datasets)
    elif args.target:
        sub = args.target.lower()
    else:
        sub = "default"
    if args.save_dir is None:
        args.save_dir = os.path.join("ckpts", "graphkeeper", sub.lower())
    if args.target:
        args.save_name = f"preprompt_{args.target.lower()}.pth"

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_root = _abs_project_path(args.data_root or os.environ.get("GFM_DATA_ROOT") or "datasets/graphkeeper")
    all_raw = load_all_datasets(data_root=data_root)
    name2item = {d["name"]: d for d in all_raw}

    # Domain order for incremental learning
    if args.target:
        sources = [d for d in all_raw if d["name"] != args.target]
    elif args.dataset:
        sources = [name2item[args.dataset.strip()]]
    elif args.domains:
        wanted = [x.strip() for x in args.domains.split(",") if x.strip()]
        sources = [name2item[n] for n in wanted]
    else:
        sources = all_raw[: args.num_domains]
    
    ordered_names = [s["name"] for s in sources]
    print(f"Incremental Learning Order: {ordered_names}")

    # Init model (per-domain prompt slots)
    model = GraphKeeperPrePromptModel(
        input_dim=args.unify_dim,
        hidden_dim=args.hidden_dim,
        num_domains=len(sources),
        num_layers=args.num_layers,
        prompt_mode=args.prompt_mode,
        temperature=args.temperature,
        contrastive_weight=args.contrastive_weight,
        lp_max_edges=args.lp_max_edges,
        device=device,
    )

    all_aligners = []

    # --- Sequential incremental pretraining loop ---
    for idx, s in enumerate(sources):
        domain_name = s["name"]
        print(f"\n>>> [Stage {idx+1}/{len(sources)}] Incremental domain: {domain_name}")

        # 1. Load current-domain data
        feat = s["ds"][0].x.numpy()
        aligner = DomainAlignment(n_components=args.unify_dim)
        aligner.fit(feat)
        all_aligners.append(aligner)
        
        # Align and move to device
        x = torch.from_numpy(aligner.transform(feat)).float().to(device)
        edge_index = s["ds"][0].edge_index.to(device)
        # domain_id batch -> model.pretexts[idx]
        batch = torch.full((x.size(0),), idx, dtype=torch.long, device=device)

        # 2. Knowledge retention: freeze old prompts, train current
        optimizer = set_incremental_optimizer(model, idx, args.lr)

        # 3. Single-domain training for this stage
        best_loss = 1e9
        cnt_wait = 0
        
        if not args.no_swanlab:
            try:
                import swanlab
                swanlab.init(
                    project=args.swanlab_project, 
                    experiment_name=f"gk_stage_{idx}_{domain_name}", 
                    config=vars(args)
                )
            except: pass

        for epoch in range(args.max_epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward routes to the correct pretext branch via batch idx
            loss = model(x, edge_index, batch, tuples=None)
            
            loss.backward()
            optimizer.step()

            should_stop, best_loss, cnt_wait = early_stopping(loss.item(), best_loss, cnt_wait, args.patience)
            
            if epoch % args.log_interval == 0:
                print(f"Domain {domain_name} | Epoch {epoch:4d} | Loss: {loss.item():.4f}")
            
            if should_stop:
                print(f"Domain {domain_name} training finished.")
                break
        
        if not args.no_swanlab:
            try: swanlab.finish()
            except: pass

    # --- Final save ---
    if not os.path.isabs(args.save_dir):
        args.save_dir = str(Path.cwd() / args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    
    ckpt_path = os.path.join(args.save_dir, args.save_name)
    save_dict = {
        "model": model.state_dict(),
        "unify_dim": args.unify_dim,
        "hidden_dim": args.hidden_dim,
        "num_domains": len(sources),
        "ordered_names": ordered_names,
        "prompt_mode": args.prompt_mode,
        "contrastive_weight": args.contrastive_weight,
        "lora_rank": 128,
    }
    torch.save(save_dict, ckpt_path)
    print(f"\nFinal Incremental Model Saved: {ckpt_path}")

    # Save domain aligners list
    try:
        import joblib
        ap = os.path.join(args.save_dir, "aligners.pkl")
        joblib.dump({"aligners": all_aligners, "ordered_names": ordered_names}, ap)
        print(f"Saved aligners: {ap}")
    except: pass

    print("GraphKeeper Pretrain Done.")

if __name__ == "__main__":
    main()