#!/usr/bin/env python
"""
HGPrompt DownPrompt few-shot finetuning (node classification).
Fixes ACM/DBLP class-count mis-detection and label out-of-range CUDA errors.
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

from pygfm.baseline_models import HGPromptPrePromptModel, HGPromptDownPromptModel
from pygfm.private.utlis.domain_alignment import DomainAlignment
from pygfm.public.utils import set_seed
from pygfm.public.utils.runtime import load_single_graph_dataset
from pygfm.public.cli.yaml_config import parse_args_with_optional_yaml
from pygfm.public.cli.export_yaml import add_export_yaml_arguments, handle_export_args
from pygfm.public.cli.default_ckpt import resolve_preprompt_ckpt


def _parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    p = argparse.ArgumentParser(
        description="HGPrompt DownPrompt few-shot finetune",
        epilog="YAML: -c PATH; export: --export-default-yaml / --export-run-yaml PATH (needs pyyaml)",
    )
    p.add_argument("--dataset", type=str, default="Cora")
    p.add_argument("--k_shot", type=int, default=1, choices=[1, 5])
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="PrePrompt checkpoint; if omitted, auto-detect under ckpts/hgprompt/ by dataset",
    )
    p.add_argument(
        "--downstream_root",
        type=str,
        default="downstream_data/hgprompt",
    )
    p.add_argument("--splits_path", type=str, default=None)
    p.add_argument("--split_id", type=int, default=0)
    p.add_argument("--task_num", type=int, default=0)
    p.add_argument("--data_root", type=str, default="datasets/hgprompt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--test_reserve", type=int, default=1000)
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--swanlab_project", type=str, default="gfm-toolbox-hgprompt")
    p.add_argument("--swanlab_run_name", type=str, default=None)
    add_export_yaml_arguments(p)
    return p, parse_args_with_optional_yaml(p)


def _load_pyg_graph(data_root, dataset):
    """Same as :func:`load_all_datasets`: flat ``ACM.pt`` / ``Cora.pt`` or subdirs like ``ACM/node.dat``."""
    return load_single_graph_dataset(data_root, dataset)


def main():
    p, args = _parse_args()
    handle_export_args(p, args)
    args.ckpt = resolve_preprompt_ckpt(ROOT, "hgprompt", args.dataset, args.ckpt)
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    use_swanlab = not args.no_swanlab
    if use_swanlab:
        try:
            import swanlab
            run_name = args.swanlab_run_name or f"finetune_{args.dataset}_{args.k_shot}shot"
            swanlab.init(project=args.swanlab_project, experiment_name=run_name, config=vars(args))
        except ImportError:
            use_swanlab = False

    ckpt = torch.load(args.ckpt, map_location=device)
    unify_dim = ckpt["unify_dim"]
    hidden_dim = ckpt["hidden_dim"]
    num_edge_types = ckpt.get("num_edge_types", 1)
    num_layers = ckpt.get("num_layers", 3)

    preprompt = HGPromptPrePromptModel(
        input_dim=unify_dim,
        hidden_dim=hidden_dim,
        num_edge_types=num_edge_types,
        num_layers=num_layers,
        temperature=1.0,
        device=device,
    )
    preprompt.load_state_dict(ckpt["model"], strict=False)
    preprompt.to(device)
    preprompt.eval()
    for p in preprompt.input_proj.parameters():
        p.requires_grad = False

    data, num_classes = _load_pyg_graph(args.data_root, args.dataset)
    x_raw = data.x.cpu().numpy()
    aligner = DomainAlignment(n_components=unify_dim)
    aligner.fit(x_raw)
    x = torch.from_numpy(aligner.transform(x_raw)).float().to(device)
    edge_index = data.edge_index.to(device)
    edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=device)
    y = data.y.to(device)

    with torch.no_grad():
        x_h = preprompt.input_proj(x)

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

    # --- Test-set filtering ---
    if args.test_reserve > 0:
        search_range = torch.arange(max(0, len(y) - args.test_reserve), len(y), device=device)
    else:
        search_range = torch.arange(len(y), device=device)

    valid_mask = (y[search_range] >= 0)
    test_idx = search_range[valid_mask]
    test_labels = y[test_idx]

    if len(test_idx) == 0:
        print("!! Warning: no labeled nodes in range; falling back to all labeled nodes as test set")
        all_valid_idx = (y >= 0).nonzero(as_tuple=True)[0]
        test_idx = all_valid_idx
        test_labels = y[test_idx]
        if len(test_idx) == 0:
            raise ValueError("No labeled nodes in dataset; check label.dat")

    print(f">> Effective test nodes: {len(test_idx)}")

    gcn = preprompt.gcn
    acc_list = []

    for sid in split_ids:
        split = splits[sid]
        support_idx = torch.tensor(split["indices"], dtype=torch.long, device=device)
        raw_support_labels = torch.tensor(split["labels"], dtype=torch.long, device=device)

        # Remap labels into [0, num_classes-1] (handles 1-based or sparse ids)
        unique_labels = torch.unique(y[y >= 0])  # globally valid label values
        label_map = {val.item(): i for i, val in enumerate(unique_labels.sort()[0])}
        
        # Map support labels to contiguous 0..C-1
        support_labels = torch.tensor([label_map.get(l.item(), 0) for l in raw_support_labels], device=device)
        
        # Keep test labels consistent with the same mapping
        # (could be done once outside the loop; kept local for safety)
        test_labels_mapped = torch.tensor([label_map.get(l.item(), 0) for l in test_labels], device=device)

        # Refresh num_classes after remap
        current_num_classes = len(label_map)
        
        down = HGPromptDownPromptModel(
            gcn=gcn,
            hidden_dim=hidden_dim,
            num_classes=current_num_classes,  # after label remap
            num_edge_types=num_edge_types,
            device=device,
        )
        down.hprompt.load_state_dict(preprompt.hprompt.state_dict(), strict=False)

        opt = torch.optim.Adam(down.hprompt.parameters(), lr=args.lr)
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
                x_h,
                edge_index,
                edge_type,
                support_idx=support_idx,
                support_labels=support_labels,
                query_idx=support_idx,
                train=True,
            )
            # logits shape: [len(support_idx), num_classes]
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
            
            if use_swanlab and step % 10 == 0:  # denser few-shot logging
                try:
                    swanlab.log({f"split_{sid}/loss": loss.item()}, step=step)
                except Exception:
                    pass

        down.eval()
        with torch.inference_mode():
            logits = down(
                x_h,
                edge_index,
                edge_type,
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
                swanlab.log({f"split_{sid}/test_acc": acc})
            except Exception:
                pass

    if len(acc_list) > 1:
        acc_tensor = torch.tensor(acc_list)
        mean_acc = acc_tensor.mean().item()
        std_acc = acc_tensor.std().item()
        print(f"[{args.dataset}] {args.k_shot}-shot Mean Acc: {mean_acc:.4f}, Std: {std_acc:.4f}")
        if use_swanlab:
            try:
                swanlab.log({"mean_test_acc": mean_acc, "std_test_acc": std_acc})
            except Exception:
                pass


if __name__ == "__main__":
    main()