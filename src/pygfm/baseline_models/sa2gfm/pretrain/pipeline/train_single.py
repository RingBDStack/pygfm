#!/usr/bin/env python3
"""
Single-dataset contrastive pretraining (JointContrastiveModel).
Does NOT implement multi-graph joint pretrain (--joint_pretrain); that path is intentionally omitted.

MoE downstream loads experts by name list get_pretrain_datasets(target)—usually *other* graphs, not target.
Prefer ``python scripts/sa2gfm/pretrain_experts_for_downstream.py --target <dataset>`` for all MoE experts.

Reads: ``resolve_ori_graph_pt(dataset)`` — ``ori/*.pt``, flat ``*.pt``, or ``sa2gfm/*.pt`` (expects ``enhanced_x`` / ``enhanced_x_64``).
Writes: ``{SA2GFM_DATA_ROOT}/save_model/{dataset}.pt`` (full ``nn.Module``, same as downstream ``torch.load``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root on path when running this file directly without editable install.
_SA2GFM_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_SA2GFM_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_SA2GFM_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from pygfm.baseline_models.sa2gfm.paths import paths
from pygfm.baseline_models.sa2gfm.pretrain.pipeline.data_utils import (
    get_negative_samples,
    load_dataset_pt,
    sparse_mx_to_torch_sparse_tensor,
)
from pygfm.baseline_models.sa2gfm.pretrain.pipeline.model import JointContrastiveModel
from pygfm.public.utils import set_seed
from pygfm.public.cli.yaml_config import parse_args_with_config


def parse_args():
    p = argparse.ArgumentParser(description="SA2GFM single-dataset pretrain (no joint merge)")
    p.add_argument("--dataset", type=str, default="cora", help="Override with the same key in a -c YAML file")
    p.add_argument("--seed", type=int, default=39)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--hid_units", type=int, default=256)
    p.add_argument("--out_channels", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--l2_coef", type=float, default=0.0)
    p.add_argument("--nb_epochs", type=int, default=3000)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=10)
    p.add_argument("--neg_samples", type=int, default=50)
    p.add_argument("--kl_weight", type=float, default=0.0)
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Default: {save_model_dir}/{dataset}.pt",
    )
    p.add_argument("--no_swanlab", action="store_true", help="Disable SwanLab logging")
    return parse_args_with_config(p, script_file=Path(__file__))


def train():
    args = parse_args()
    if not str(getattr(args, "dataset", "") or "").strip():
        raise SystemExit("SA2GFM pretrain: set --dataset or `dataset` in the YAML passed via -c")
    set_seed(args.seed)
    paths.save_model_dir.mkdir(parents=True, exist_ok=True)

    data_path = paths.resolve_ori_graph_pt(args.dataset)

    features_np, adj_sp, num_nodes = load_dataset_pt(str(data_path))
    print(f"Loaded {data_path}: nodes={num_nodes}, feat_dim={features_np.shape[1]}, edges~={adj_sp.nnz // 2}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    neg = torch.from_numpy(get_negative_samples(adj_sp, num_nodes, args.neg_samples)).to(device)

    features = torch.FloatTensor(features_np).to(device)
    adj = sparse_mx_to_torch_sparse_tensor(adj_sp).to(device)

    model = JointContrastiveModel(
        in_channels=features.shape[1],
        hidden_channels=args.hid_units,
        out_channels=args.out_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    out_path = Path(args.output) if args.output else paths.save_model_dir / f"{args.dataset}.pt"
    torch.save(model, out_path)
    print(f"Initial checkpoint -> {out_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

    if not args.no_swanlab:
        import swanlab

        swanlab.init(project="sa2gfm_pretrain", config=vars(args), requirements_collect=False)

    best_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in tqdm(range(1, args.nb_epochs + 1), desc="pretrain"):
        model.train()
        optimizer.zero_grad()
        contrastive_loss, kl_loss = model([features], [adj], neg)
        total_loss = contrastive_loss + args.kl_weight * kl_loss
        if torch.isnan(total_loss):
            continue
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if epoch % args.eval_steps != 0:
            continue

        print(
            f"epoch {epoch:5d} | L_con={contrastive_loss.item():.4f} L_kl={kl_loss.item():.4f} L={total_loss.item():.4f}"
        )
        if not args.no_swanlab:
            import swanlab

            swanlab.log(
                {
                    "epoch": epoch,
                    "contrastive_loss": contrastive_loss.item(),
                    "kl_loss": kl_loss.item(),
                    "total_loss": total_loss.item(),
                }
            )

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_epoch = epoch
            patience_counter = 0
            torch.save(model, out_path)
        else:
            patience_counter += 1

        if patience_counter >= args.patience // args.eval_steps:
            print(f"Early stop at epoch {epoch} (best {best_epoch}, loss {best_loss:.4f})")
            break

    print(f"Done. Best loss {best_loss:.4f} @ epoch {best_epoch}. Saved: {out_path}")


if __name__ == "__main__":
    train()
