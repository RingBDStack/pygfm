#!/usr/bin/env python
"""
MDGFM DownPrompt few-shot node classification — faithful reproduction.

Matches original MDGFM-main/MDGFM.py downstream flow:
prefeatureprompt → balance token → ATT_learner k-NN → alpha-blended adj →
GCN → prototype cosine similarity → softmax.

Hyperparameters match original (Cora: test_reserve=10000, downk=30, patience=500).
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
warnings.filterwarnings("ignore")

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from pathlib import Path

from pygfm.public.repo_paths import driver_script_repo_root

ROOT = driver_script_repo_root(__file__)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pygfm.baseline_models import MDGFMPrePromptModel, MDGFMDownPromptModel
from pygfm.private.utlis.domain_alignment import DomainAlignment
from pygfm.public.utils import set_seed
from pygfm.public.utils.runtime import load_single_graph_dataset_or_reddit
from pygfm.public.cli.yaml_config import parse_args_with_optional_yaml
from pygfm.public.cli.export_yaml import add_export_yaml_arguments, handle_export_args
from pygfm.public.cli.default_ckpt import resolve_preprompt_ckpt


# ---------------------------------------------------------------------------
# edge_index → sparse adj (same as pretrain)
# ---------------------------------------------------------------------------

def edge_index_to_sparse_adj(
    edge_index: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    row, col = edge_index.cpu().numpy()
    data = np.ones(len(row), dtype=np.float32)
    adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    adj = sp.csr_matrix(adj)
    adj = adj + sp.eye(adj.shape[0], dtype=np.float32)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_norm = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    indices = torch.from_numpy(
        np.vstack((adj_norm.row, adj_norm.col)).astype(np.int64)
    )
    values = torch.from_numpy(adj_norm.data.astype(np.float32))
    shape = torch.Size(adj_norm.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# ---------------------------------------------------------------------------
# Dataset-specific constants (from original MDGFM.py)
# ---------------------------------------------------------------------------

DATASET_CONFIG = {
    "Cora":      {"downk": 30, "testnum": 10000},
    "Citeseer":  {"downk": 30, "testnum": 10000},
    "Pubmed":    {"downk": 30, "testnum": 100},
    "Chameleon": {"downk": 15, "testnum": 10000},
    "Squirrel":  {"downk": 15, "testnum": 10000},
    "Cornell":   {"downk": 15, "testnum": 10000},
    "Photo":     {"downk": 15, "testnum": 10000},
    "Computers": {"downk": 15, "testnum": 10000},
}


def _parse_args():
    p = argparse.ArgumentParser(
        description="MDGFM DownPrompt few-shot node finetune",
        epilog="YAML: -c PATH; export: --export-default-yaml / --export-run-yaml PATH",
    )
    p.add_argument("--dataset", type=str, default="Cora")
    p.add_argument("--k_shot", type=int, default=1, choices=[1, 5])
    p.add_argument("--ckpt", type=str, default=None,
                   help="PrePrompt checkpoint; auto-detect if omitted")
    p.add_argument("--downstream_root", type=str, default="downstream_data/mdgfm")
    p.add_argument("--splits_path", type=str, default=None)
    p.add_argument("--split_id", type=int, default=0)
    p.add_argument("--task_num", type=int, default=0)
    p.add_argument("--data_root", type=str, default="datasets/mdgfm")
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--patience", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--test_reserve", type=int, default=None)
    p.add_argument("--downk", type=int, default=None)
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--swanlab_project", type=str, default="gfm-toolbox-mdgfm")
    p.add_argument("--swanlab_run_name", type=str, default=None)
    add_export_yaml_arguments(p)
    return p, parse_args_with_optional_yaml(p)


def main():
    p, args = _parse_args()
    handle_export_args(p, args)

    # Dataset-specific defaults
    ds_cfg = DATASET_CONFIG.get(args.dataset, {"downk": 15, "testnum": 10000})
    downk = args.downk if args.downk is not None else ds_cfg["downk"]
    test_reserve = args.test_reserve if args.test_reserve is not None else ds_cfg["testnum"]

    args.ckpt = resolve_preprompt_ckpt(ROOT, "mdgfm", args.dataset, args.ckpt)
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    use_swanlab = not args.no_swanlab
    if use_swanlab:
        try:
            import swanlab
            run_name = args.swanlab_run_name or f"finetune_{args.dataset}_{args.k_shot}shot"
            swanlab.init(
                project=args.swanlab_project, experiment_name=run_name, config=vars(args)
            )
        except ImportError:
            use_swanlab = False

    # ---- Load checkpoint ----
    ckpt = torch.load(args.ckpt, map_location=device)
    unify_dim = ckpt["unify_dim"]
    hidden_dim = ckpt["hidden_dim"]
    prompt_mode = ckpt["prompt_mode"]
    num_domains = ckpt.get("num_domains", 5)

    # Rebuild PrePrompt to extract weights and GCN
    preprompt = MDGFMPrePromptModel(
        n_in=unify_dim,
        n_h=hidden_dim,
        num_domains=num_domains,
        num_layers_num=3,
        dropout=0.1,
        prompt_mode=prompt_mode,
        device=device,
    )
    preprompt.load_state_dict(ckpt["model"], strict=False)
    preprompt.to(device)
    preprompt.eval()

    all_weights = preprompt.get_weights()
    gcn = preprompt.gcn

    # ---- Load target dataset ----
    data, num_classes = load_single_graph_dataset_or_reddit(args.data_root, args.dataset)
    x_raw = data.x.cpu().numpy()

    # PCA compression
    aligner = DomainAlignment(n_components=unify_dim)
    aligner.fit(x_raw)
    x = torch.from_numpy(aligner.transform(x_raw)).float().to(device)

    # Convert edge_index → sparse adj
    sp_adj = edge_index_to_sparse_adj(data.edge_index, x.size(0)).to(device)

    y = data.y.to(device)

    # ---- Load few-shot splits ----
    splits_path = args.splits_path or os.path.join(
        args.downstream_root, args.dataset, f"{args.k_shot}shot", "splits.pt"
    )
    down_data = torch.load(splits_path, map_location="cpu")
    splits = down_data["splits"]
    n_splits = len(splits)

    if args.task_num and args.task_num > 0:
        split_ids = list(range(min(args.task_num, n_splits)))
    else:
        if not (0 <= args.split_id < n_splits):
            raise IndexError(f"split_id {args.split_id} out of range ({n_splits} splits)")
        split_ids = [args.split_id]

    test_start = max(0, len(y) - test_reserve)
    test_idx = torch.arange(test_start, len(y), device=device)
    test_labels = y[test_idx]

    # ---- Pre-compute pretrained embeddings for all nodes ----
    with torch.no_grad():
        pretrained_embs = preprompt.embed(x, sp_adj, sparse=True, LP=False)

    # ---- Few-shot evaluation ----
    acc_list = []
    for sid in split_ids:
        split = splits[sid]
        support_idx = torch.tensor(split["indices"], dtype=torch.long, device=device)
        support_labels = torch.tensor(split["labels"], dtype=torch.long, device=device)

        # Pretrained embeddings at support nodes
        pretrain_embs = pretrained_embs[0, support_idx]

        # Build DownPrompt
        down = MDGFMDownPromptModel(
            gcn=gcn,
            all_weights=all_weights,
            ft_in=hidden_dim,
            nb_classes=num_classes,
            feature_dim=unify_dim,
            prompt_mode=prompt_mode,
            device=device,
        )
        opt = torch.optim.Adam(down.parameters(), lr=args.lr)
        best_loss = 1e9
        cnt_wait = 0

        print(
            f">> MDGFM Finetune {args.dataset} | {args.k_shot}-shot | "
            f"split {sid} | support={len(support_idx)}, test={len(test_idx)} | "
            f"downk={downk}"
        )

        for step in range(args.max_steps):
            down.train()
            opt.zero_grad()

            logits = down(
                features=x,
                adj=sp_adj,
                sparse=True,
                idx=support_idx,
                seq=pretrain_embs,
                downk=downk,
                labels=support_labels,
                train=1,
            )
            loss = F.cross_entropy(logits, support_labels)
            loss.backward(retain_graph=True)
            opt.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                cnt_wait = 0
            else:
                cnt_wait += 1
            if cnt_wait >= args.patience:
                print(f"  Early stopping at step {step}, best loss={best_loss:.4f}")
                break

        # ---- Evaluate ----
        down.eval()
        with torch.no_grad():
            logits = down(
                features=x,
                adj=sp_adj,
                sparse=True,
                idx=test_idx,
                seq=pretrained_embs[0, test_idx],
                downk=downk,
                labels=None,
                train=0,
            )
        preds = torch.argmax(logits, dim=1)
        acc = (preds == test_labels).float().mean().item() * 100
        acc_list.append(acc)
        print(f"[{args.dataset}] {args.k_shot}-shot split {sid} test accuracy: {acc:.2f}%")

        if use_swanlab:
            try:
                swanlab.log({f"split_{sid}/test_acc": acc})
            except Exception:
                pass

    if len(acc_list) > 1:
        acc_tensor = torch.tensor(acc_list)
        mean_acc = acc_tensor.mean().item()
        std_acc = acc_tensor.std().item()
        print(
            f"[{args.dataset}] {args.k_shot}-shot {len(acc_list)} splits: "
            f"mean={mean_acc:.2f}%, std={std_acc:.2f}%"
        )
        if use_swanlab:
            try:
                swanlab.log({
                    "mean_test_acc": mean_acc,
                    "std_test_acc": std_acc,
                    "n_splits": len(acc_list),
                })
            except Exception:
                pass


if __name__ == "__main__":
    main()
