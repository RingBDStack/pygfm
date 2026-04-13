#!/usr/bin/env python
"""
MDGFM PrePrompt pretraining (per-domain pretext + shared sumtext + GCN + NodeNodeContrastiveLoss); saves ckpt for downstream.
Same tier as MDGPT/SAMGPT; uses GFM load_all_datasets, DomainAlignment, sample_negative_pairs, early_stopping.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pygfm.private.utlis.domain_alignment import DomainAlignment
from pygfm.private.utlis.loss_calculation import sample_negative_pairs
from pygfm.baseline_models import MDGFMPrePromptModel
from pygfm.public.utils.runtime import set_seed, load_all_datasets, early_stopping
from pygfm.public.cli.yaml_config import parse_args_with_optional_yaml
from pygfm.public.cli.export_yaml import add_export_yaml_arguments, handle_export_args


def main():
    p = argparse.ArgumentParser(
        description="MDGFM PrePrompt pretraining (pretext+sumtext+GCN+LP)",
        epilog="YAML: -c PATH; export: --export-default-yaml / --export-run-yaml PATH (needs pyyaml)",
    )
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument(
        "--save_dir",
        type=str,
        default="ckpts/mdgfm",
        help="With --target: ckpts/mdgfm/{target}; with --datasets: ckpts/mdgfm/{normalized-lowercase-names}",
    )
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
    p.add_argument("--prompt_mode", type=str, default="mul", choices=["add", "mul"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=500)
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--swanlab_project", type=str, default="gfm-toolbox-mdgfm")
    p.add_argument("--swanlab_run_name", type=str, default=None)
    add_export_yaml_arguments(p)
    args = parse_args_with_optional_yaml(p)
    handle_export_args(p, args)

    if args.target:
        args.save_dir = os.path.join("ckpts/mdgfm", args.target.lower())
        args.save_name = f"preprompt_{args.target.lower()}.pth"
    elif args.datasets:
        ds_key = "_".join(x.strip().lower() for x in args.datasets.split(",") if x.strip())
        args.save_dir = os.path.join("ckpts/mdgfm", ds_key)
        args.save_name = f"preprompt_{ds_key}.pth"

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root or os.environ.get("GFM_DATA_ROOT", "datasets/mdgfm")

    all_raw = load_all_datasets(data_root=data_root)
    name2item = {d["name"]: d for d in all_raw}
    if args.target:
        if args.target not in name2item:
            raise ValueError(f"Unknown target '{args.target}'.")
        sources = [d for d in all_raw if d["name"] != args.target]
        ordered_names = [s["name"] for s in sources]
    elif args.datasets:
        wanted = [x.strip() for x in args.datasets.split(",") if x.strip()]
        sources = [name2item[n] for n in wanted]
        ordered_names = [s["name"] for s in sources]
    else:
        sources = all_raw[: args.num_domains]
        ordered_names = [s["name"] for s in sources]
    print("Pretrain domains:", ordered_names)

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

    model = MDGFMPrePromptModel(
        input_dim=args.unify_dim,
        hidden_dim=args.hidden_dim,
        num_domains=len(sources),
        num_layers=args.num_layers,
        prompt_mode=args.prompt_mode,
        temperature=args.temperature,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    use_swanlab = not args.no_swanlab
    if use_swanlab:
        try:
            import swanlab
            run_name = args.swanlab_run_name or f"pretrain_target_{args.target or 'all'}"
            swanlab.init(project=args.swanlab_project, experiment_name=run_name, config=vars(args))
        except ImportError:
            use_swanlab = False

    print(">> MDGFM PrePrompt pretraining ...")
    best_loss = 1e9
    cnt_wait = 0
    for epoch in range(args.max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = model(big_batch.x, big_batch.edge_index, big_batch.batch, tuples)
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
    if args.target:
        save_dict["target"] = args.target
    torch.save(save_dict, ckpt_path)
    print(f"Saved: {ckpt_path}")
    try:
        import joblib
        aligner_path = os.path.join(args.save_dir, "aligners.pkl")
        joblib.dump({"aligners": aligners, "ordered_names": ordered_names}, aligner_path)
        print(f"Saved aligners: {aligner_path}")
    except Exception:
        pass
    print("Pretrain done.")


if __name__ == "__main__":
    main()
