#!/usr/bin/env python
"""
GraphPrompt DownPrompt few-shot graph classification finetune.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pygfm.baseline_models import GraphPromptPrePromptModel, GraphPromptDownPromptGraphModel
from pygfm.private.utlis.domain_alignment import DomainAlignment
from pygfm.private.utlis.downstream_data_gen import build_test_subgraphs
from pygfm.public.utils import set_seed
from pygfm.public.utils.runtime import load_single_graph_dataset


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GraphPrompt DownPrompt few-shot graph finetune")
    p.add_argument("--dataset", type=str, default="Cora")
    p.add_argument("--k_shot", type=int, default=1, choices=[1, 5])
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument(
        "--downstream_root",
        type=str,
        default="downstream_data/graphprompt",
    )
    p.add_argument("--splits_path", type=str, default=None)
    p.add_argument("--split_id", type=int, default=0)
    p.add_argument("--task_num", type=int, default=0)
    p.add_argument("--data_root", type=str, default="datasets/graphprompt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--test_reserve", type=int, default=1000)
    p.add_argument("--max_one_hop", type=int, default=10)
    p.add_argument("--max_two_hop", type=int, default=4)
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--swanlab_project", type=str, default="gfm-toolbox-graphprompt")
    p.add_argument("--swanlab_run_name", type=str, default=None)
    return p.parse_args()


def _load_pyg_graph(data_root, dataset):
    return load_single_graph_dataset(data_root, dataset)


def main():
    args = _parse_args()
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    use_swanlab = not args.no_swanlab
    if use_swanlab:
        try:
            import swanlab
            run_name = args.swanlab_run_name or f"finetune_graph_{args.dataset}_{args.k_shot}shot"
            swanlab.init(project=args.swanlab_project, experiment_name=run_name, config=vars(args))
        except ImportError:
            use_swanlab = False

    ckpt = torch.load(args.ckpt, map_location=device)
    unify_dim = ckpt["unify_dim"]
    hidden_dim = ckpt["hidden_dim"]
    prompt_mode = ckpt["prompt_mode"]
    num_domains = ckpt.get("num_domains", 4)
    num_layers = ckpt.get("num_layers", 3)

    preprompt = GraphPromptPrePromptModel(
        input_dim=unify_dim,
        hidden_dim=hidden_dim,
        num_domains=num_domains,
        num_layers=num_layers,
        prompt_mode=prompt_mode,
        temperature=1.0,
        device=device,
    )
    preprompt.load_state_dict(ckpt["model"], strict=False)
    preprompt.to(device)
    preprompt.eval()

    data, num_classes = _load_pyg_graph(args.data_root, args.dataset)
    x_raw = data.x.cpu().numpy()
    aligner = DomainAlignment(n_components=unify_dim)
    aligner.fit(x_raw)
    x = torch.from_numpy(aligner.transform(x_raw)).float().to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)

    if args.splits_path is not None:
        splits_path = args.splits_path
    else:
        splits_path = os.path.join(
            args.downstream_root,
            args.dataset,
            f"{args.k_shot}shot_graph_batch",
            "splits.pt",
        )
    if not os.path.isfile(splits_path):
        raise FileNotFoundError(
            f"Graph batch splits not found: {splits_path}\n"
            f"Run first: python scripts/graphprompt/generate_downstream.py graph_batch "
            f"--dataset {args.dataset} --k_shot {args.k_shot}"
        )
    down_data = torch.load(splits_path, map_location="cpu")
    splits = down_data["splits"]
    n_splits = len(splits)
    if args.task_num and args.task_num > 0:
        max_tasks = min(args.task_num, n_splits)
        split_ids = list(range(max_tasks))
    else:
        if not (0 <= args.split_id < n_splits):
            raise IndexError(f"split_id {args.split_id} out of range (0..{n_splits-1})")
        split_ids = [args.split_id]

    test_start = max(0, len(y) - args.test_reserve)
    test_indices = list(range(test_start, len(y)))
    test_labels = y[test_start:].to(device)
    testlist, testindex = build_test_subgraphs(
        edge_index.cpu(),
        test_indices,
        max_one_hop=args.max_one_hop,
        max_two_hop=args.max_two_hop,
        seed=args.seed,
    )
    testlist = testlist.to(device)
    testindex = testindex.to(device)

    gcn = preprompt.gcn
    acc_list = []

    for sid in split_ids:
        split = splits[sid]
        support_idx = split["idx"].to(device)
        support_batch = split["batch"].to(device)
        support_labels = split["labels"].to(device)

        down = GraphPromptDownPromptGraphModel(
            gcn=gcn,
            input_dim=unify_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            prompt_mode=prompt_mode,
            device=device,
        )
        down.prefeature.load_state_dict(preprompt.pretexts[0].state_dict(), strict=False)

        opt = torch.optim.Adam(down.prefeature.parameters(), lr=args.lr)
        best_loss = 1e9
        cnt_wait = 0

        num_support_graphs = int(support_batch.max().item()) + 1
        print(
            f">> Finetune graph {args.dataset} | {args.k_shot}-shot | split {sid} | "
            f"support_graphs={num_support_graphs}, test_graphs={len(test_indices)}"
        )
        for step in range(args.max_steps):
            down.train()
            opt.zero_grad()
            logits = down(
                x,
                edge_index,
                support_idx=support_idx,
                support_batch=support_batch,
                support_labels=support_labels,
                query_idx=support_idx,
                query_batch=support_batch,
                train=True,
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
            if use_swanlab and step % 100 == 0:
                try:
                    swanlab.log({f"split_{sid}/loss": loss.item()}, step=step)
                except Exception:
                    pass

        down.eval()
        with torch.inference_mode():
            logits = down(
                x,
                edge_index,
                support_idx=support_idx,
                support_batch=support_batch,
                support_labels=support_labels,
                query_idx=testlist,
                query_batch=testindex,
                train=False,
            )
            preds = logits.argmax(dim=1)
            acc = (preds == test_labels).float().mean().item()

        acc_list.append(acc)
        print(
            f"[{args.dataset}] graph {args.k_shot}-shot split {sid} "
            f"test accuracy: {acc:.4f}"
        )
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
            f"[{args.dataset}] graph {args.k_shot}-shot "
            f"{len(acc_list)} splits mean acc: {mean_acc:.4f}, std: {std_acc:.4f}"
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
