#!/usr/bin/env python
"""
BRIDGE DownPrompt graph-level few-shot: same graph_batch splits.pt format as MDGPT.

Examples:
  python scripts/bridge/finetune_graph.py --dataset Cora --k_shot 1 \\
    --ckpt ckpts/bridge/cora_run/preprompt_cora.pth --split_id 0 --no_swanlab
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pygfm.private.utlis.domain_alignment import DomainAlignment
from pygfm.private.utlis.downstream_data_gen import build_test_subgraphs
from pygfm.baseline_models.bridge import BridgeDownPromptGraphModel
from pygfm.public.utils.runtime import compute_spectral_components, load_single_graph_dataset, set_seed


def _aligned_features_from_ckpt_dir(
    ckpt_path: str, x_np: np.ndarray, aligned_dim: int, dataset: str
) -> np.ndarray:
    p = Path(ckpt_path).resolve().parent / "aligners.pkl"
    if p.is_file():
        try:
            import joblib

            blob = joblib.load(p)
            names = blob.get("ordered_names") or []
            aligners = blob.get("aligners") or []
            if dataset in names:
                i = names.index(dataset)
                if i < len(aligners):
                    return aligners[i].transform(x_np)
        except Exception:
            pass
    al = DomainAlignment(n_components=aligned_dim)
    al.fit(x_np)
    return al.transform(x_np)


def _parse():
    p = argparse.ArgumentParser(description="BRIDGE graph-level few-shot finetune")
    p.add_argument("--dataset", type=str, default="Cora")
    p.add_argument("--k_shot", type=int, default=1, choices=[1, 5])
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--downstream_root", type=str, default="downstream_data/bridge")
    p.add_argument("--splits_path", type=str, default=None)
    p.add_argument("--split_id", type=int, default=0)
    p.add_argument("--task_num", type=int, default=0)
    p.add_argument("--data_root", type=str, default="datasets/bridge")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_epochs", type=int, default=400)
    p.add_argument("--lr_router", type=float, default=1e-3)
    p.add_argument("--lr_proto", type=float, default=1e-2)
    p.add_argument("--weight_ent", type=float, default=0.1)
    p.add_argument("--weight_spec", type=float, default=0.01)
    p.add_argument("--test_reserve", type=int, default=1000)
    p.add_argument("--spectral_k", type=int, default=10)
    p.add_argument("--max_one_hop", type=int, default=10)
    p.add_argument("--max_two_hop", type=int, default=4)
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--swanlab_project", type=str, default="gfm-toolbox-bridge")
    return p.parse_args()


def _load_graph(data_root: str, dataset: str):
    return load_single_graph_dataset(data_root, dataset)


def main():
    args = _parse()
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    adim = ckpt["aligned_dim"]
    hdim = ckpt["hidden_dim"]
    ns = ckpt["num_sources"]

    data, num_classes = _load_graph(args.data_root, args.dataset)
    x_np = data.x.cpu().numpy().astype(float)
    x_aligned = _aligned_features_from_ckpt_dir(args.ckpt, x_np, adim, args.dataset)
    x = torch.from_numpy(x_aligned.astype(np.float32)).to(device)
    ei = data.edge_index.to(device)
    y = data.y.to(device)

    sp = args.splits_path or os.path.join(
        args.downstream_root, args.dataset, f"{args.k_shot}shot_graph_batch", "splits.pt"
    )
    if not os.path.isfile(sp):
        raise FileNotFoundError(
            f"Missing {sp}\nRun: python scripts/bridge/generate_downstream.py graph_batch "
            f"--dataset {args.dataset} --k_shot {args.k_shot}"
        )
    down = torch.load(sp, map_location="cpu")
    splits = down["splits"]
    nsp = len(splits)
    sids = list(range(min(args.task_num, nsp))) if args.task_num > 0 else [args.split_id]

    test_start = max(0, y.size(0) - args.test_reserve)
    test_indices = list(range(test_start, y.size(0)))
    test_y = y[test_start:].to(device)
    tlist, tbatch = build_test_subgraphs(
        ei.cpu(), test_indices, args.max_one_hop, args.max_two_hop, args.seed
    )
    tlist, tbatch = tlist.to(device), tbatch.to(device)

    eivec, eival = compute_spectral_components(ei, y.size(0), device=device, k=args.spectral_k)

    accs = []
    for sid in sids:
        split = splits[sid]
        sidx = split["idx"].to(device)
        sb = split["batch"].to(device)
        slabels = split["labels"].to(device)

        model = BridgeDownPromptGraphModel(
            aligned_dim=adim,
            hidden_dim=hdim,
            num_sources=ns,
            num_classes=num_classes,
            domain_name=args.dataset,
            device=device,
        )
        model.load_preprompt_checkpoint(ckpt, strict=False)
        model.freeze_preprompt_parts()

        model.eval()
        with torch.no_grad():
            h0 = model.embed_backbone_unmasked(x, ei)
            from torch_geometric.utils import scatter

            ns_g = int(sb.max().item()) + 1
            sg0 = scatter(h0[sidx], sb, dim=0, dim_size=ns_g, reduce="mean")
            protos = []
            for c in range(num_classes):
                m = slabels == c
                protos.append(sg0[m].mean(0) if m.any() else torch.zeros(hdim, device=device))
            model.prototypes.data.copy_(torch.stack(protos))

        opt = torch.optim.Adam(
            [
                {"params": model.routing_net.parameters(), "lr": args.lr_router},
                {"params": [model.graph_prompt], "lr": args.lr_router},
                {"params": [model.prototypes], "lr": args.lr_proto},
            ],
            lr=args.lr_router,
            weight_decay=1e-5,
        )

        for _ in range(args.max_epochs):
            model.train()
            opt.zero_grad()
            ls, _, reg, ent = model.forward_pyg_fewshot(
                x, ei, sidx, sb, sidx, sb, eivec=eivec, eival=eival
            )
            loss = F.cross_entropy(ls, slabels) + args.weight_ent * ent + args.weight_spec * reg
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            _, lq, _, _ = model.forward_pyg_fewshot(
                x, ei, sidx, sb, tlist, tbatch, eivec=eivec, eival=eival
            )
            pred = lq.argmax(1)
            acc = (pred == test_y).float().mean().item()
        accs.append(acc)
        print(f"[{args.dataset}] graph {args.k_shot}-shot split {sid} test acc: {acc:.4f}")

    if len(accs) > 1:
        t = torch.tensor(accs)
        print(f"mean {t.mean():.4f} std {t.std():.4f}")


if __name__ == "__main__":
    main()
