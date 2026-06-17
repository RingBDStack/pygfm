#!/usr/bin/env python
"""
GraphKeeper-style DownPrompt few-shot node classification: frozen GCN + trainable prompt + LoRA.
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

from pygfm.private.utlis.domain_alignment import DomainAlignment
from pygfm.baseline_models.graphkeeper import (
    GraphKeeperDownPromptModel,
    GraphKeeperPrePromptModel,
)
from pygfm.public.utils import set_seed
from pygfm.public.utils.runtime import load_single_graph_dataset_or_reddit
from pygfm.public.cli.yaml_config import parse_args_with_optional_yaml
from pygfm.public.cli.export_yaml import add_export_yaml_arguments, handle_export_args
from pygfm.public.cli.default_ckpt import resolve_preprompt_ckpt


def _parse_args():
    p = argparse.ArgumentParser(
        description="GraphKeeper DownPrompt few-shot finetune",
        epilog="YAML: -c PATH; export: --export-default-yaml / --export-run-yaml PATH (needs pyyaml)",
    )
    p.add_argument("--dataset", type=str, default="Cora")
    p.add_argument("--k_shot", type=int, default=1, choices=[1, 5])
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="PrePrompt checkpoint; auto-detect under ckpts/graphkeeper/; pass explicit path for custom pretrain subdir",
    )
    p.add_argument("--downstream_root", type=str, default="downstream_data/graphkeeper")
    p.add_argument("--splits_path", type=str, default=None)
    p.add_argument("--split_id", type=int, default=0)
    p.add_argument("--task_num", type=int, default=0)
    p.add_argument("--data_root", type=str, default="datasets/graphkeeper")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--test_reserve", type=int, default=1000)
    p.add_argument("--lora_rank", type=int, default=None, help="Default from ckpt['lora_rank'] or 128")
    p.add_argument("--lora_scale", type=float, default=1.0)
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--swanlab_project", type=str, default="gfm-toolbox-graphkeeper")
    p.add_argument("--swanlab_run_name", type=str, default=None)
    add_export_yaml_arguments(p)
    return p, parse_args_with_optional_yaml(p)


def _load_pyg_graph(data_root: str, dataset: str):
    return load_single_graph_dataset_or_reddit(data_root, dataset)


def main():
    p, args = _parse_args()
    handle_export_args(p, args)
    args.ckpt = resolve_preprompt_ckpt(ROOT, "graphkeeper", args.dataset, args.ckpt)
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    use_swanlab = not args.no_swanlab
    if use_swanlab:
        try:
            import swanlab
            swanlab.init(
                project=args.swanlab_project,
                experiment_name=args.swanlab_run_name or f"gk_ft_{args.dataset}_{args.k_shot}shot",
                config=vars(args),
            )
        except ImportError:
            use_swanlab = False

    ckpt = torch.load(args.ckpt, map_location=device)
    unify_dim = ckpt["unify_dim"]
    hidden_dim = ckpt["hidden_dim"]
    prompt_mode = ckpt["prompt_mode"]
    num_domains = ckpt.get("num_domains", 4)
    lora_rank = args.lora_rank if args.lora_rank is not None else ckpt.get("lora_rank", 128)

    preprompt = GraphKeeperPrePromptModel(
        input_dim=unify_dim,
        hidden_dim=hidden_dim,
        num_domains=num_domains,
        num_layers=3,
        prompt_mode=prompt_mode,
        temperature=1.0,
        contrastive_weight=0.0,
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

    if args.splits_path:
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
        split_ids = list(range(min(args.task_num, n_splits)))
    else:
        if not (0 <= args.split_id < n_splits):
            raise IndexError(f"split_id {args.split_id} out of range (0..{n_splits-1})")
        split_ids = [args.split_id]

    test_start = max(0, len(y) - args.test_reserve)
    test_idx = torch.arange(test_start, len(y), device=device)
    test_labels = y[test_idx]

    gcn = preprompt.gcn
    acc_list = []

    for sid in split_ids:
        split = splits[sid]
        support_idx = torch.tensor(split["indices"], dtype=torch.long, device=device)
        support_labels = torch.tensor(split["labels"], dtype=torch.long, device=device)

        down = GraphKeeperDownPromptModel(
            gcn=gcn,
            input_dim=unify_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            lora_rank=lora_rank,
            lora_scale=args.lora_scale,
            prompt_mode=prompt_mode,
            device=device,
        )
        down.prefeature.load_state_dict(preprompt.pretexts[0].state_dict(), strict=False)

        opt = torch.optim.Adam(
            list(down.prefeature.parameters()) + list(down.lora.parameters()),
            lr=args.lr,
        )
        best_loss = 1e9
        cnt_wait = 0

        print(
            f">> GraphKeeper finetune {args.dataset} | {args.k_shot}-shot | split {sid} | "
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
                    import swanlab
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
        print(f"[{args.dataset}] {args.k_shot}-shot split {sid} test accuracy: {acc:.4f}")
        if use_swanlab:
            try:
                import swanlab
                swanlab.log({f"split_{sid}/test_acc": acc})
            except Exception:
                pass

    if len(acc_list) > 1:
        t = torch.tensor(acc_list)
        print(
            f"[{args.dataset}] {args.k_shot}-shot {len(acc_list)} splits "
            f"mean acc: {t.mean().item():.4f}, std: {t.std().item():.4f}"
        )


if __name__ == "__main__":
    main()
