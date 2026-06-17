#!/usr/bin/env python
"""
MDGPT-style DownPrompt few-shot finetune (shared CLI pattern).

Features:
- Load PrePrompt checkpoint (pretrain.py output)
- Load few-shot splits from downstream_data/mdgpt/{dataset}/{k}shot/splits.pt
- Finetune k-shot DownPrompt on the chosen split(s) and evaluate on the test set

Example:
    python scripts/mdgpt/finetune.py \\
      --dataset Cora \\
      --k_shot 1 \\
      --ckpt ckpts/mdgpt/cora/preprompt_cora.pth \\
      --split_id 0
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from pygfm.public.repo_paths import driver_script_repo_root

import torch
import torch.nn.functional as F

ROOT = driver_script_repo_root(__file__)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_SCRIPTS_MDGPT = Path(__file__).resolve().parent
if str(_SCRIPTS_MDGPT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_MDGPT))

from pygfm.baseline_models import PrePromptModel, DownPromptModel
from pygfm.private.utlis.domain_alignment import DomainAlignment
from pygfm.public.utils import set_seed
from pygfm.public.utils.runtime import load_single_graph_dataset_or_reddit

from config_utils import parse_args_with_config


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MDGPT DownPrompt few-shot finetune")
    p.add_argument("--dataset", type=str, default="Cora", help="Target dataset name")
    p.add_argument("--k_shot", type=int, default=1, choices=[1, 5], help="k-shot")
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="PrePrompt checkpoint (from pretrain.py); CLI or YAML",
    )
    p.add_argument(
        "--downstream_root",
        type=str,
        default="downstream_data/mdgpt",
        help="Few-shot root (per baseline; from generate_downstream.py)",
    )
    p.add_argument(
        "--splits_path",
        type=str,
        default=None,
        help="Explicit splits.pt path; if empty use "
             "downstream_root/dataset/{k}shot/splits.pt",
    )
    p.add_argument("--split_id", type=int, default=0, help="Split index when task_num=0")
    p.add_argument(
        "--task_num",
        type=int,
        default=0,
        help=">0: first task_num splits; =0: only split_id",
    )
    p.add_argument("--data_root", type=str, default="datasets/mdgpt", help="Raw graph root (per-baseline layout)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3, help="DownPrompt finetune learning rate")
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--test_reserve", type=int, default=1000, help="Tail nodes reserved for testing")
    p.add_argument("--no_swanlab", action="store_true", help="Disable SwanLab logging")
    p.add_argument("--swanlab_project", type=str, default="gfm-toolbox-mdgpt", help="SwanLab project name")
    p.add_argument("--swanlab_run_name", type=str, default=None, help="SwanLab run name (auto if omitted)")
    return parse_args_with_config(p, script_file=Path(__file__))


def main():
    args = _parse_args()
    if not args.ckpt:
        raise SystemExit("Missing --ckpt: set pretrained checkpoint path on CLI or in YAML.")
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Optional SwanLab initialization
    use_swanlab = not args.no_swanlab
    if use_swanlab:
        try:
            import swanlab
            run_name = args.swanlab_run_name or f"finetune_{args.dataset}_{args.k_shot}shot"
            swanlab.init(project=args.swanlab_project, experiment_name=run_name, config=vars(args))
        except ImportError:
            use_swanlab = False

    # 1. Load PrePrompt checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    unify_dim = ckpt["unify_dim"]
    hidden_dim = ckpt["hidden_dim"]
    prompt_mode = ckpt["prompt_mode"]

    preprompt = PrePromptModel(
        input_dim=unify_dim,
        hidden_dim=hidden_dim,
        num_domains=4,
        num_layers=3,
        prompt_mode=prompt_mode,
        temperature=1.0,
        device=device,
    )
    preprompt.load_state_dict(ckpt["model"], strict=False)
    preprompt.to(device)
    preprompt.eval()

    # 2. Load target graph and PCA-align (prefer flat Cora.pt / data.pt to avoid Planetoid download)
    data, num_classes = load_single_graph_dataset_or_reddit(args.data_root, args.dataset)
    x_raw = data.x.cpu().numpy()
    aligner = DomainAlignment(n_components=unify_dim)
    aligner.fit(x_raw)
    x = torch.from_numpy(aligner.transform(x_raw)).float().to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)

    # 3. Load few-shot splits.pt
    if args.splits_path is not None:
        splits_path = args.splits_path
    else:
        splits_path = os.path.join(
            args.downstream_root,
            args.dataset,
            f"{args.k_shot}shot",
            "splits.pt",
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

    # Test nodes: last test_reserve (same as split generator)
    test_start = max(0, len(y) - args.test_reserve)
    test_idx = torch.arange(test_start, len(y), device=device)
    test_labels = y[test_idx]

    # 4. Finetune/evaluate DownPrompt per split
    gcn = preprompt.gcn
    acc_list = []

    for sid in split_ids:
        split = splits[sid]
        support_idx = torch.tensor(split["indices"], dtype=torch.long, device=device)
        support_labels = torch.tensor(split["labels"], dtype=torch.long, device=device)

        down = DownPromptModel(
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

        print(
            f">> Finetune {args.dataset} | {args.k_shot}-shot | split {sid} | "
            f"support={len(support_idx)}, test={len(test_idx)}"
        )
        for step in range(args.max_steps):
            down.train()
            opt.zero_grad()
            logits = down(
                x,
                edge_index,
                support_idx=support_idx,
                support_labels=support_labels,
                query_idx=support_idx,
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
                support_labels=support_labels,
                query_idx=test_idx,
                train=False,
            )
            preds = logits.argmax(dim=1)
            acc = (preds == test_labels).float().mean().item()

        acc_list.append(acc)
        print(
            f"[{args.dataset}] {args.k_shot}-shot split {sid} "
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
            f"[{args.dataset}] {args.k_shot}-shot "
            f"{len(acc_list)} splits mean acc: {mean_acc:.4f}, std: {std_acc:.4f}"
        )
        if use_swanlab:
            try:
                swanlab.log({"mean_test_acc": mean_acc, "std_test_acc": std_acc, "n_splits": len(acc_list)})
            except Exception:
                pass


if __name__ == "__main__":
    main()
