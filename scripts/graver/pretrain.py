#!/usr/bin/env python
"""
GRAVER PrePrompt: leave-one-out sources, per-source sigmoid masks + DisenGCN + link contrastive loss.
Saves ckpt for scripts/graver/finetune.py and finetune_graph.py.

Examples:
  python scripts/graver/pretrain.py --target Cora
  python scripts/graver/pretrain.py --datasets Cora,Citeseer,Pubmed,Photo
"""
from __future__ import annotations

import argparse
import os
import sys
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

from pygfm.private.utlis.domain_alignment import DomainAlignment
from pygfm.private.utlis.loss_calculation import sample_negative_pairs
from pygfm.baseline_models.graver import GRAVERPrePromptModel
from pygfm.public.utils.runtime import set_seed, load_all_datasets
from pygfm.public.cli.yaml_config import parse_args_with_optional_yaml
from pygfm.public.cli.export_yaml import add_export_yaml_arguments, handle_export_args


def _parse():
    p = argparse.ArgumentParser(
        description="GRAVER PrePrompt pretraining",
        epilog="YAML: -c PATH; export: --export-default-yaml / --export-run-yaml PATH (needs pyyaml)",
    )
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="ckpts/graver")
    p.add_argument("--save_name", type=str, default="preprompt.pth")
    p.add_argument("--seed", type=int, default=39)
    p.add_argument("--unify_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_neg", type=int, default=50)
    p.add_argument("--datasets", type=str, default=None)
    p.add_argument("--target", type=str, default=None, help="Leave-one-out target domain")
    # DisenGCN hyperparameters
    p.add_argument("--init_k", type=int, default=2)
    p.add_argument("--delta_k", type=int, default=0)
    p.add_argument("--routit", type=int, default=1)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--num_layers", type=int, default=1)
    # Training hyperparameters
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_epochs", type=int, default=10000)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--row_norm", action="store_true")
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--swanlab_project", type=str, default="gfm-toolbox-graver")
    p.add_argument("--swanlab_run_name", type=str, default=None)
    p.add_argument("--log_interval", type=int, default=10)
    add_export_yaml_arguments(p)
    return p, parse_args_with_optional_yaml(p)


def main():
    p, args = _parse()
    handle_export_args(p, args)
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root or os.environ.get("GFM_DATA_ROOT", "datasets/graver")

    if args.target:
        args.save_dir = os.path.join("ckpts/graver", args.target.lower())
        args.save_name = f"preprompt_{args.target.lower()}.pth"

    use_swanlab = not args.no_swanlab
    if use_swanlab:
        try:
            import swanlab
            swanlab.init(
                project=args.swanlab_project,
                experiment_name=args.swanlab_run_name or f"graver_pretrain_{args.target or 'all'}",
                config=vars(args),
            )
        except ImportError:
            use_swanlab = False

    all_raw = load_all_datasets(data_root=data_root)
    name2item = {d["name"]: d for d in all_raw}

    if args.target:
        if args.target not in name2item:
            raise ValueError(f"Unknown target {args.target!r}; available: {list(name2item)}")
        sources = [d for d in all_raw if d["name"] != args.target]
    elif args.datasets:
        wanted = [x.strip() for x in args.datasets.split(",") if x.strip()]
        sources = [name2item[n] for n in wanted]
    else:
        sources = all_raw[:4]

    ordered_names = [s["name"] for s in sources]
    num_sources = len(sources)
    print(f"GRAVER PrePrompt | source domains: {ordered_names}")

    # ---- Data: PCA align + per-source edge_index ----
    pre_dir = Path("pre_data_graver")
    pre_dir.mkdir(parents=True, exist_ok=True)

    features_list = []
    edge_index_list = []
    node_counts = []

    for s in sources:
        feat_raw = s["ds"][0].x
        feat_np = (feat_raw.cpu().numpy() if torch.is_tensor(feat_raw) else np.asarray(feat_raw)).astype(np.float64)
        if args.row_norm:
            rs = feat_np.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1.0
            feat_np /= rs

        pca_path = pre_dir / f"{s['name']}_pca_{args.unify_dim}.npy"
        if pca_path.exists():
            aligned = np.load(pca_path)
        else:
            aligner = DomainAlignment(n_components=args.unify_dim)
            aligner.fit(feat_np)
            aligned = aligner.transform(feat_np)
            np.save(pca_path, aligned)

        x_t = torch.from_numpy(aligned.astype(np.float32)).to(device)
        ei = s["ds"][0].edge_index.to(device)
        features_list.append(x_t)
        edge_index_list.append(ei)
        node_counts.append(x_t.size(0))

    # ---- Negative sampling on concatenated graph ----
    offset = 0
    combined_edges = []
    for ei, nc in zip(edge_index_list, node_counts):
        combined_edges.append(ei.cpu() + offset)
        offset += nc
    combined_ei = torch.cat(combined_edges, dim=1)
    total_nodes = sum(node_counts)
    negative_samples = sample_negative_pairs(combined_ei, total_nodes, num_neg=args.num_neg, seed=args.seed)

    # ---- Model ----
    model = GRAVERPrePromptModel(
        input_dim=args.unify_dim,
        hidden_dim=args.hidden_dim,
        num_sources=num_sources,
        init_k=args.init_k,
        delta_k=args.delta_k,
        routit=args.routit,
        tau=args.tau,
        dropout=args.dropout,
        num_layers=args.num_layers,
        temperature=1.0,
        device=device,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- Train loop ----
    from tqdm import tqdm

    best_loss = float("inf")
    cnt_wait = 0
    pbar = tqdm(range(args.max_epochs))
    for epoch in pbar:
        model.train()
        opt.zero_grad()
        loss = model(features_list, edge_index_list, negative_samples)
        loss.backward()
        opt.step()

        loss_val = loss.item()
        if epoch % args.log_interval == 0:
            pbar.set_description(f"loss={loss_val:.4f}")
            if use_swanlab:
                try:
                    import swanlab
                    swanlab.log({"pretrain/loss": loss_val}, step=epoch)
                except Exception:
                    pass

        if loss_val < best_loss:
            best_loss = loss_val
            cnt_wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            cnt_wait += 1
        if cnt_wait >= args.patience:
            print(f"Early stopping at epoch {epoch}, best loss: {best_loss:.4f}")
            break

    # ---- Save ----
    os.makedirs(args.save_dir, exist_ok=True)
    out = os.path.join(args.save_dir, args.save_name)
    payload = {
        "model": best_state,
        "input_dim": args.unify_dim,
        "hidden_dim": args.hidden_dim,
        "num_sources": num_sources,
        "ordered_names": ordered_names,
        "init_k": args.init_k,
        "delta_k": args.delta_k,
        "routit": args.routit,
        "tau": args.tau,
        "dropout": args.dropout,
        "num_layers": args.num_layers,
    }
    if args.target:
        payload["target"] = args.target
    torch.save(payload, out)
    print(f"Saved: {out}  (best loss: {best_loss:.4f})")

    if use_swanlab:
        try:
            import swanlab
            swanlab.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
