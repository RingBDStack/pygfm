#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG-GFM graph-level finetuning: few-shot graph classification (subgraph pooling) with RAG-GFM PrePrompt + DownPromptGraph.

- Load RAG-GFM PrePrompt ckpt (scripts/rag_gfm/pretrain.py)
- Load downstream_data/rag_gfm/{dataset}/{k}shot_graph_batch/splits.pt
- Uses pygfm.baseline_models.mdgpt.DownPromptGraphModel and build_test_subgraphs

Usage:
  python scripts/rag_gfm/finetune_graph.py --dataset Cora --k_shot 1 --ckpt ckpts/rag_gfm/cora/preprompt_cora.pth
  python scripts/rag_gfm/finetune_graph.py --dataset Cora --k_shot 5 --ckpt ckpts/rag_gfm/preprompt.pth --task_num 10
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from pygfm.public.repo_paths import driver_script_repo_root

ROOT = driver_script_repo_root(__file__)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from pygfm.private.utlis.domain_alignment import DomainAlignment
from pygfm.private.utlis.rag_gfm.motif_builder import load_node_data_for_motif
from pygfm.private.utlis.downstream_data_gen import build_test_subgraphs
from pygfm.baseline_models.rag_gfm import PrePromptModel
from pygfm.baseline_models.mdgpt import DownPromptGraphModel
from pygfm.public.utils import set_seed
from pygfm.public.cli.yaml_config import parse_args_with_optional_yaml


def _apply_yaml_ckpt_aliases(args: argparse.Namespace) -> None:
    if getattr(args, "ckpt", None) and str(args.ckpt).strip():
        return
    try:
        import yaml
    except ImportError:
        return
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("-c", "--config", default=None)
    known, _ = pre.parse_known_args(sys.argv[1:])
    if not known.config:
        return
    path = Path(known.config).expanduser().resolve()
    if not path.is_file():
        return
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return
    for key in ("checkpoint", "preprompt_ckpt", "preprompt_path"):
        v = raw.get(key)
        if v is not None and str(v).strip():
            args.ckpt = str(v).strip()
            return


def _rag_gfm_default_preprompt_ckpt(dataset: str) -> Path:
    sub = str(dataset).strip().lower()
    return (ROOT / "ckpts" / "rag_gfm" / sub / f"preprompt_{sub}.pth").resolve()


def _resolve_rag_gfm_ckpt(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    raw = getattr(args, "ckpt", None)
    if raw is not None and str(raw).strip():
        ck = str(raw).strip()
    else:
        cand = _rag_gfm_default_preprompt_ckpt(args.dataset)
        if cand.is_file():
            ck = str(cand)
            print(f"[rag_gfm/finetune_graph] Using default PrePrompt ckpt: {ck}")
        else:
            parser.error(
                "Missing --ckpt (or YAML ``ckpt``). Tried default: "
                f"{cand} (not found). Run pretrain for this target or set ``ckpt``."
            )
    if not os.path.isabs(ck):
        ck = str((ROOT / ck).resolve())
    args.ckpt = ck


def _parse_args():
    p = argparse.ArgumentParser(description="RAG-GFM DownPrompt few-shot graph classification finetuning")
    p.add_argument("--dataset", type=str, default="Cora", help="Target dataset")
    p.add_argument("--k_shot", type=int, default=1, choices=[1, 5], help="k-shot")
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="RAG-GFM PrePrompt checkpoint. Default: ckpts/rag_gfm/<dataset>/preprompt_<dataset>.pth",
    )
    p.add_argument("--downstream_root", type=str, default="downstream_data/rag_gfm", help="Few-shot data root")
    p.add_argument("--splits_path", type=str, default=None, help="Explicit path to splits.pt")
    p.add_argument("--split_id", type=int, default=0)
    p.add_argument("--task_num", type=int, default=0, help="If >0, run first task_num splits")
    p.add_argument("--data_root", type=str, default="datasets/rag_gfm", help="Graph data root")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--test_reserve", type=int, default=1000)
    p.add_argument("--max_one_hop", type=int, default=10)
    p.add_argument("--max_two_hop", type=int, default=4)
    p.add_argument("--no_swanlab", action="store_true", help="Disable SwanLab logging")
    p.add_argument("--swanlab_project", type=str, default="gfmtoolbox_raggfm", help="SwanLab project name")
    p.add_argument("--swanlab_run_name", type=str, default=None, help="SwanLab run name (auto if omitted)")
    args = parse_args_with_optional_yaml(p)
    _apply_yaml_ckpt_aliases(args)
    _resolve_rag_gfm_ckpt(args, p)
    return args


def _load_graph(data_root: str, dataset: str):
    data = load_node_data_for_motif(data_root, dataset)
    if data is None:
        raise FileNotFoundError(f"Dataset not found: {dataset}, check {data_root}")
    if not hasattr(data, "y") or data.y is None:
        raise ValueError(f"Dataset {dataset} has no label tensor y")
    num_classes = int(data.y.max().item()) + 1
    return data, num_classes


def main():
    args = _parse_args()
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Optional SwanLab initialization
    use_swanlab = not args.no_swanlab
    if use_swanlab:
        try:
            import swanlab
            run_name = args.swanlab_run_name or f"finetune_graph_rag_gfm_{args.dataset}_{args.k_shot}shot"
            swanlab.init(project=args.swanlab_project, experiment_name=run_name, config=vars(args))
            print(f"[SwanLab] connected | project: {args.swanlab_project} | run: {run_name}")
        except ImportError:
            use_swanlab = False
            print("[SwanLab] not installed (pip install swanlab), skipping logging")
    else:
        print("[SwanLab] disabled (--no_swanlab)")

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Pretrain checkpoint not found: {args.ckpt}")

    try:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location=device)
    unify_dim = ckpt["unify_dim"]
    hidden_dim = ckpt["hidden_dim"]
    prompt_mode = ckpt["prompt_mode"]
    num_domains = ckpt.get("num_domains", 3)

    preprompt = PrePromptModel(
        input_dim=unify_dim,
        hidden_dim=hidden_dim,
        num_domains=num_domains,
        num_layers=3,
        prompt_mode=prompt_mode,
        temperature=1.0,
        device=device,
    )
    preprompt.load_state_dict(ckpt["model"], strict=False)
    preprompt.to(device)
    preprompt.eval()

    data, num_classes = _load_graph(args.data_root, args.dataset)
    x_raw = data.x.cpu().numpy()
    aligner = DomainAlignment(n_components=unify_dim)
    aligner.fit(x_raw)
    x = torch.from_numpy(aligner.transform(x_raw)).float().to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)

    if args.splits_path is not None:
        splits_path = args.splits_path
    else:
        splits_path = os.path.join(args.downstream_root, args.dataset, f"{args.k_shot}shot_graph_batch", "splits.pt")
    if not os.path.isfile(splits_path):
        raise FileNotFoundError(
            f"Graph batch splits not found: {splits_path}\n"
            f"Run: generate_downstream graph_batch --dataset {args.dataset} --k_shot {args.k_shot} "
            f"--downstream_root {args.downstream_root} --data_root {args.data_root}"
        )
    down_data = torch.load(splits_path, map_location="cpu")
    splits = down_data["splits"]
    n_splits = len(splits)
    if args.task_num > 0:
        split_ids = list(range(min(args.task_num, n_splits)))
    else:
        if not (0 <= args.split_id < n_splits):
            raise IndexError(f"split_id {args.split_id} out of range (0..{n_splits-1})")
        split_ids = [args.split_id]

    test_start = max(0, len(y) - args.test_reserve)
    test_indices = list(range(test_start, len(y)))
    test_labels = y[test_start:].to(device)
    testlist, testindex = build_test_subgraphs(
        edge_index.cpu(), test_indices,
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

        down = DownPromptGraphModel(
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
        print(f">> RAG-GFM finetune graph {args.dataset} | {args.k_shot}-shot | split {sid} | support_graphs={num_support_graphs}, test_graphs={len(test_indices)}")
        for step in range(args.max_steps):
            down.train()
            opt.zero_grad()
            logits = down(
                x, edge_index,
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
                x, edge_index,
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
        print(f"[{args.dataset}] graph {args.k_shot}-shot split {sid} test acc: {acc:.4f}")
        if use_swanlab:
            try:
                swanlab.log({f"split_{sid}/test_acc": acc})
            except Exception:
                pass

    if len(acc_list) > 1:
        mean_acc = sum(acc_list) / len(acc_list)
        import math
        std_acc = math.sqrt(sum((a - mean_acc) ** 2 for a in acc_list) / len(acc_list))
        print(f"[{args.dataset}] graph {args.k_shot}-shot {len(acc_list)} splits mean acc: {mean_acc:.4f}, std: {std_acc:.4f}")
        if use_swanlab:
            try:
                swanlab.log({"mean_test_acc": mean_acc, "std_test_acc": std_acc, "n_splits": len(acc_list)})
            except Exception:
                pass


if __name__ == "__main__":
    main()
