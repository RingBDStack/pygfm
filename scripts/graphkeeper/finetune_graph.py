#!/usr/bin/env python
"""
GraphKeeper-style DownPrompt few-shot graph classification.
Auto-matches pretrain Aligner and the corresponding expert prompt (pretexts).
"""
from __future__ import annotations

import argparse
import os
import sys
import joblib
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pygfm.private.utlis.domain_alignment import DomainAlignment
from pygfm.private.utlis.downstream_data_gen import build_test_subgraphs
from pygfm.baseline_models.graphkeeper import (
    GraphKeeperDownPromptGraphModel,
    GraphKeeperPrePromptModel,
)
from pygfm.public.utils import set_seed
from pygfm.public.utils.runtime import load_single_graph_dataset


def _parse_args():
    p = argparse.ArgumentParser(description="GraphKeeper graph-level few-shot finetune")
    p.add_argument("--dataset", type=str, default="Cora")
    p.add_argument("--k_shot", type=int, default=1, choices=[1, 5])
    p.add_argument("--ckpt", type=str, required=True, help="Path to pretrained preprompt.pth")
    p.add_argument("--data_root", type=str, default="datasets/graphkeeper")
    p.add_argument("--downstream_root", type=str, default="downstream_data/graphkeeper")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--test_reserve", type=int, default=1000)
    p.add_argument("--max_one_hop", type=int, default=10)
    p.add_argument("--max_two_hop", type=int, default=4)
    p.add_argument("--lora_rank", type=int, default=128)
    p.add_argument("--lora_scale", type=float, default=1.0)
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--swanlab_project", type=str, default="gfm-toolbox-graphkeeper")
    return p.parse_args()


def _load_pyg_graph(data_root: str, dataset: str):
    return load_single_graph_dataset(data_root, dataset)


def main():
    args = _parse_args()
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Load pretrained checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    unify_dim = ckpt["unify_dim"]
    hidden_dim = ckpt["hidden_dim"]
    prompt_mode = ckpt["prompt_mode"]
    num_domains = ckpt.get("num_domains", 4)
    lora_rank = args.lora_rank if args.lora_rank is not None else ckpt.get("lora_rank", 128)

    pre_model = GraphKeeperPrePromptModel(
        input_dim=unify_dim,
        hidden_dim=hidden_dim,
        num_domains=num_domains,
        prompt_mode=prompt_mode,
        device=device,
    )
    pre_model.load_state_dict(ckpt["model"], strict=False)
    pre_model.to(device)
    pre_model.eval()

    # 2. Data load + aligner match
    data, num_classes = _load_pyg_graph(args.data_root, args.dataset)
    x_raw = data.x.cpu().numpy()

    # Load saved aligner when present
    aligner_path = Path(args.ckpt).parent / "aligners.pkl"
    expert_idx = 0 
    
    if aligner_path.exists():
        pkg = joblib.load(aligner_path)
        if args.dataset in pkg["ordered_names"]:
            expert_idx = pkg["ordered_names"].index(args.dataset)
            aligner = pkg["aligners"][expert_idx]
            print(f"[*] Found pre-trained aligner/expert for {args.dataset} (idx={expert_idx})")
        else:
            aligner = DomainAlignment(n_components=unify_dim)
            aligner.fit(x_raw)
            expert_idx = num_domains - 1  # default: last expert branch
            print(f"[!] New domain. Using last expert branch (idx={expert_idx})")
    else:
        aligner = DomainAlignment(n_components=unify_dim)
        aligner.fit(x_raw)

    x = torch.from_numpy(aligner.transform(x_raw)).float().to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)

    # 3. Downstream subgraph splits
    splits_path = os.path.join(args.downstream_root, args.dataset, f"{args.k_shot}shot_graph_batch", "splits.pt")
    if not os.path.isfile(splits_path):
        raise FileNotFoundError(f"Missing graph splits: {splits_path}")
    
    down_data = torch.load(splits_path, map_location="cpu")
    splits = down_data["splits"]

    # Build test subgraphs (graph-level: pool nodes to graph embedding)
    test_start = max(0, len(y) - args.test_reserve)
    test_indices = list(range(test_start, len(y)))
    test_labels = y[test_start:].to(device)
    test_list, test_batch = build_test_subgraphs(
        edge_index.cpu(), test_indices,
        max_one_hop=args.max_one_hop, max_two_hop=args.max_two_hop, seed=args.seed,
    )
    test_list, test_batch = test_list.to(device), test_batch.to(device)

    acc_list = []
    # Cap number of test splits
    split_ids = range(min(5, len(splits)))

    for sid in split_ids:
        split = splits[sid]
        support_idx = split["idx"].to(device)
        support_batch = split["batch"].to(device)
        support_labels = split["labels"].to(device)

        # 4. Build graph-level finetune model
        down_model = GraphKeeperDownPromptGraphModel(
            gcn=pre_model.gcn,  # shared frozen GCN
            input_dim=unify_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            lora_rank=lora_rank,
            lora_scale=args.lora_scale,
            prompt_mode=prompt_mode,
            device=device,
        )
        # Load matching expert branch weights
        down_model.prefeature.load_state_dict(pre_model.pretexts[expert_idx].state_dict())

        optimizer = torch.optim.Adam(
            list(down_model.prefeature.parameters()) + list(down_model.lora.parameters()),
            lr=args.lr,
        )

        best_loss = 1e9
        cnt_wait = 0

        # 5. Finetune loop
        for step in range(args.max_steps):
            down_model.train()
            optimizer.zero_grad()
            
            # Forward: model applies global_mean_pool for graph readout
            logits = down_model(
                x, edge_index,
                support_idx=support_idx, support_batch=support_batch,
                support_labels=support_labels,
                query_idx=support_idx, query_batch=support_batch,  # train loss on support
                train=True,
            )
            loss = F.cross_entropy(logits, support_labels)
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                cnt_wait = 0
            else:
                cnt_wait += 1
            if cnt_wait >= args.patience: break

        # 6. Test
        down_model.eval()
        with torch.no_grad():
            test_logits = down_model(
                x, edge_index,
                support_idx=support_idx, support_batch=support_batch,
                support_labels=support_labels,
                query_idx=test_list, query_batch=test_batch,
                train=False,
            )
            preds = test_logits.argmax(dim=1)
            acc = (preds == test_labels).float().mean().item()
            acc_list.append(acc)
            print(f"Split {sid} | Test Acc: {acc:.4f}")

    if acc_list:
        mean_acc = torch.tensor(acc_list).mean().item()
        print(f"\n>>> [{args.dataset}] Graph-level {args.k_shot}-shot Mean Acc: {mean_acc:.4f}")

if __name__ == "__main__":
    main()