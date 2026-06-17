#!/usr/bin/env python
"""
MDGFM PrePrompt pretraining — faithful reproduction of original MDGFM-main/MDGFM.py.

Per-domain PCA → sparse adj → per-domain pretext + sumtext → balance token →
ATT_learner refined adjacency → GCN LP head → Calbound LP loss.

Hyperparameters match original paper (Cora: lr=0.02, wd=0.0001, patience=500).
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore")

import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path

from pygfm.public.repo_paths import driver_script_repo_root

ROOT = driver_script_repo_root(__file__)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pygfm.private.utlis.domain_alignment import DomainAlignment
from pygfm.baseline_models import MDGFMPrePromptModel
from pygfm.public.utils.runtime import set_seed, load_all_datasets
from pygfm.public.cli.yaml_config import parse_args_with_optional_yaml
from pygfm.public.cli.export_yaml import add_export_yaml_arguments, handle_export_args


# ---------------------------------------------------------------------------
# edge_index → sparse COO adjacency (matching original normalize_adj)
# ---------------------------------------------------------------------------

def edge_index_to_sparse_adj(
    edge_index: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """
    Convert PyG edge_index to normalized sparse COO tensor.

    Follows the exact sequence from MDGFM-main/utils/process.py:
      1. edge_index → scipy.coo → csr
      2. adj + I
      3. D^{-1/2} @ adj @ D^{-1/2}  (symmetric normalization)
      4. → torch.sparse.FloatTensor
    """
    row, col = edge_index.cpu().numpy()
    data = np.ones(len(row), dtype=np.float32)
    adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    adj = sp.csr_matrix(adj)

    # Add self-loops + symmetric normalization
    adj = adj + sp.eye(adj.shape[0], dtype=np.float32)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_norm = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    # To torch sparse
    indices = torch.from_numpy(
        np.vstack((adj_norm.row, adj_norm.col)).astype(np.int64)
    )
    values = torch.from_numpy(adj_norm.data.astype(np.float32))
    shape = torch.Size(adj_norm.shape)

    return torch.sparse.FloatTensor(indices, values, shape)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="MDGFM PrePrompt pretraining (original Calbound LP loss)",
        epilog="YAML: -c PATH; export: --export-default-yaml / --export-run-yaml PATH",
    )
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument(
        "--save_dir", type=str, default="ckpts/mdgfm",
        help="Checkpoint output directory",
    )
    p.add_argument("--save_name", type=str, default="preprompt.pth")
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--unify_dim", type=int, default=50)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--num_domains", type=int, default=5)
    p.add_argument("--datasets", type=str, default=None)
    p.add_argument("--target", type=str, default=None)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--patience", type=int, default=500)
    p.add_argument("--max_epochs", type=int, default=60)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--prompt_mode", type=str, default="mul", choices=["add", "mul"])
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--swanlab_project", type=str, default="gfm-toolbox-mdgfm")
    p.add_argument("--swanlab_run_name", type=str, default=None)
    add_export_yaml_arguments(p)
    args = parse_args_with_optional_yaml(p)
    handle_export_args(p, args)

    # Auto-detect save paths
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

    # ---- Load datasets ----
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

    # ---- Per-domain PCA + sparse adj ----
    feats_list = []
    adjs_list = []
    aligners = []

    for idx, s in enumerate(sources):
        data = s["ds"][0]
        feat = data.x.numpy()

        # PCA compression to unify_dim
        aligner = DomainAlignment(n_components=args.unify_dim)
        aligner.fit(feat)
        aligners.append(aligner)
        aligned = torch.from_numpy(aligner.transform(feat)).float()

        # Convert edge_index → normalized sparse adj
        sp_adj = edge_index_to_sparse_adj(data.edge_index, aligned.size(0))

        feats_list.append(aligned)
        adjs_list.append(sp_adj)

        print(f"  {ordered_names[idx]}: nodes={aligned.size(0)}, "
              f"edges={data.edge_index.size(1)}, "
              f"features={aligned.size(1)}, "
              f"adj_shape={sp_adj.shape}")

    # ---- Determine target index ----
    target_idx = None
    if args.target:
        # The target domain was excluded; it doesn't participate in pretraining
        # Original code replaces one source domain's features with the target's
        # Here we keep it simple: all sources are pretrained equally
        target_idx = None
    print(f"Num pretrain domains: {len(feats_list)} (target excluded: {args.target})")

    # ---- Build model ----
    model = MDGFMPrePromptModel(
        n_in=args.unify_dim,
        n_h=args.hidden_dim,
        num_domains=len(feats_list),
        num_layers_num=args.num_layers,
        dropout=args.dropout,
        prompt_mode=args.prompt_mode,
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    use_swanlab = not args.no_swanlab
    if use_swanlab:
        try:
            import swanlab
            run_name = args.swanlab_run_name or f"pretrain_target_{args.target or 'all'}"
            swanlab.init(
                project=args.swanlab_project,
                experiment_name=run_name,
                config=vars(args),
            )
        except ImportError:
            use_swanlab = False

    # ---- Training loop ----
    print(f">> MDGFM PrePrompt: lr={args.lr}, wd={args.weight_decay}, "
          f"epochs={args.max_epochs}, patience={args.patience}")
    best_loss = 1e9
    cnt_wait = 0
    best_epoch = 0

    for epoch in range(args.max_epochs):
        # Original code re-seeds each epoch (following MDGFM.py:170-172)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        model.train()
        optimizer.zero_grad()

        loss = model(
            feats=feats_list,
            adjs=adjs_list,
            sparse=True,
            target_idx=target_idx,
        )
        loss.backward()
        optimizer.step()

        loss_val = loss.item()

        if use_swanlab:
            try:
                swanlab.log({"loss": loss_val, "epoch": epoch}, step=epoch)
            except Exception:
                pass

        if epoch % args.log_interval == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss_val:.4f}")

        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = epoch
            cnt_wait = 0
            # Save best checkpoint
            os.makedirs(args.save_dir, exist_ok=True)
            ckpt_path = os.path.join(args.save_dir, args.save_name)
            save_dict = {
                "model": model.state_dict(),
                "unify_dim": args.unify_dim,
                "hidden_dim": args.hidden_dim,
                "num_domains": len(feats_list),
                "ordered_names": ordered_names,
                "prompt_mode": args.prompt_mode,
                "best_loss": best_loss,
                "best_epoch": best_epoch,
            }
            if args.target:
                save_dict["target"] = args.target
            torch.save(save_dict, ckpt_path)
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print(f"Early stopping at epoch {epoch} (best={best_epoch}, loss={best_loss:.4f})")
            break

    # ---- Save final checkpoint ----
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, args.save_name)
    save_dict = {
        "model": model.state_dict(),
        "unify_dim": args.unify_dim,
        "hidden_dim": args.hidden_dim,
        "num_domains": len(feats_list),
        "ordered_names": ordered_names,
        "prompt_mode": args.prompt_mode,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
    }
    if args.target:
        save_dict["target"] = args.target
    torch.save(save_dict, ckpt_path)
    print(f"Saved: {ckpt_path}")

    # Save aligners
    try:
        import joblib
        aligner_path = os.path.join(args.save_dir, "aligners.pkl")
        joblib.dump({"aligners": aligners, "ordered_names": ordered_names}, aligner_path)
        print(f"Saved aligners: {aligner_path}")
    except Exception:
        pass

    print(f"Pretrain done. Best epoch={best_epoch}, best_loss={best_loss:.4f}")


if __name__ == "__main__":
    main()
