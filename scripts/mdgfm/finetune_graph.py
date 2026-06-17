#!/usr/bin/env python
"""
MDGFM DownPrompt few-shot graph classification — faithful reproduction.

Same node-level pipeline as finetune.py, with graph-level scatter_mean aggregation
for prototypes and query embeddings.
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

from pygfm.baseline_models import MDGFMPrePromptModel, MDGFMDownPromptGraphModel
from pygfm.private.utlis.domain_alignment import DomainAlignment
from pygfm.private.utlis.downstream_data_gen import build_test_subgraphs
from pygfm.public.utils import set_seed
from pygfm.public.utils.runtime import load_single_graph_dataset_or_reddit
from pygfm.public.cli.yaml_config import parse_args_with_optional_yaml
from pygfm.public.cli.export_yaml import add_export_yaml_arguments, handle_export_args


def edge_index_to_sparse_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
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
    indices = torch.from_numpy(np.vstack((adj_norm.row, adj_norm.col)).astype(np.int64))
    values = torch.from_numpy(adj_norm.data.astype(np.float32))
    shape = torch.Size(adj_norm.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


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
    p = argparse.ArgumentParser(description="MDGFM DownPrompt few-shot graph finetune")
    p.add_argument("--dataset", type=str, default="Cora")
    p.add_argument("--k_shot", type=int, default=1, choices=[1, 5])
    p.add_argument("--ckpt", type=str, required=True)
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
    p.add_argument("--max_one_hop", type=int, default=10)
    p.add_argument("--max_two_hop", type=int, default=4)
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--swanlab_project", type=str, default="gfm-toolbox-mdgfm")
    p.add_argument("--swanlab_run_name", type=str, default=None)
    return p, p.parse_args()


def main():
    p, args = _parse_args()

    ds_cfg = DATASET_CONFIG.get(args.dataset, {"downk": 15, "testnum": 10000})
    downk = args.downk if args.downk is not None else ds_cfg["downk"]
    test_reserve = args.test_reserve if args.test_reserve is not None else ds_cfg["testnum"]

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    use_swanlab = not args.no_swanlab
    if use_swanlab:
        try:
            import swanlab
            run_name = args.swanlab_run_name or f"finetune_graph_{args.dataset}_{args.k_shot}shot"
            swanlab.init(
                project=args.swanlab_project, experiment_name=run_name, config=vars(args)
            )
        except ImportError:
            use_swanlab = False

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    unify_dim = ckpt["unify_dim"]
    hidden_dim = ckpt["hidden_dim"]
    prompt_mode = ckpt["prompt_mode"]
    num_domains = ckpt.get("num_domains", 5)

    preprompt = MDGFMPrePromptModel(
        n_in=unify_dim, n_h=hidden_dim, num_domains=num_domains,
        num_layers_num=3, dropout=0.1, prompt_mode=prompt_mode, device=device,
    )
    preprompt.load_state_dict(ckpt["model"], strict=False)
    preprompt.to(device)
    preprompt.eval()

    all_weights = preprompt.get_weights()
    gcn = preprompt.gcn

    # Load target dataset
    data, num_classes = load_single_graph_dataset_or_reddit(args.data_root, args.dataset)
    x_raw = data.x.cpu().numpy()

    aligner = DomainAlignment(n_components=unify_dim)
    aligner.fit(x_raw)
    x = torch.from_numpy(aligner.transform(x_raw)).float().to(device)
    sp_adj = edge_index_to_sparse_adj(data.edge_index, x.size(0)).to(device)
    y = data.y.to(device)

    # Load splits
    splits_path = args.splits_path or os.path.join(
        args.downstream_root, args.dataset, f"{args.k_shot}shot_graph_batch", "splits.pt"
    )
    if not os.path.isfile(splits_path):
        raise FileNotFoundError(f"Graph batch splits not found: {splits_path}")
    down_data = torch.load(splits_path, map_location="cpu")
    splits = down_data["splits"]
    n_splits = len(splits)

    if args.task_num and args.task_num > 0:
        split_ids = list(range(min(args.task_num, n_splits)))
    else:
        if not (0 <= args.split_id < n_splits):
            raise IndexError(f"split_id {args.split_id} out of range")
        split_ids = [args.split_id]

    test_start = max(0, len(y) - test_reserve)
    test_indices = list(range(test_start, len(y)))
    test_labels = y[test_start:].to(device)
    testlist, testindex = build_test_subgraphs(
        data.edge_index.cpu(), test_indices,
        max_one_hop=args.max_one_hop, max_two_hop=args.max_two_hop, seed=args.seed,
    )
    testlist = testlist.to(device)
    testindex = testindex.to(device)

    acc_list = []
    for sid in split_ids:
        split = splits[sid]
        support_idx = split["idx"].to(device)
        support_batch = split["batch"].to(device)
        support_labels = split["labels"].to(device)

        num_support_graphs = int(support_batch.max().item()) + 1

        down = MDGFMDownPromptGraphModel(
            gcn=gcn, all_weights=all_weights,
            ft_in=hidden_dim, nb_classes=num_classes, feature_dim=unify_dim,
            prompt_mode=prompt_mode, device=device,
        )
        opt = torch.optim.Adam(down.parameters(), lr=args.lr)
        best_loss = 1e9
        cnt_wait = 0

        print(
            f">> MDGFM Finetune graph {args.dataset} | {args.k_shot}-shot | "
            f"split {sid} | support_graphs={num_support_graphs}, "
            f"test_graphs={len(test_indices)} | downk={downk}"
        )

        for step in range(args.max_steps):
            down.train()
            opt.zero_grad()
            logits = down(
                features=x, adj=sp_adj, sparse=True,
                support_idx=support_idx, support_batch=support_batch,
                support_labels=support_labels,
                downk=downk,
                query_idx=support_idx, query_batch=support_batch,
                train=1,
            )
            loss = F.cross_entropy(logits, support_labels)
            loss.backward()
            opt.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                cnt_wait = 0
            else:
                cnt_wait += 1
            if cnt_wait >= args.patience:
                print(f"  Early stopping at step {step}, best loss={best_loss:.4f}")
                break

        down.eval()
        with torch.no_grad():
            logits = down(
                features=x, adj=sp_adj, sparse=True,
                support_idx=support_idx, support_batch=support_batch,
                support_labels=support_labels,
                downk=downk,
                query_idx=testlist, query_batch=testindex,
                train=0,
            )
        preds = torch.argmax(logits, dim=1)
        acc = (preds == test_labels).float().mean().item() * 100
        acc_list.append(acc)
        print(f"[{args.dataset}] graph {args.k_shot}-shot split {sid} test accuracy: {acc:.2f}%")

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
            f"[{args.dataset}] graph {args.k_shot}-shot {len(acc_list)} splits: "
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
