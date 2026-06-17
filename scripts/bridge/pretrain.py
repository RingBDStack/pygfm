#!/usr/bin/env python
"""
BRIDGE PrePrompt: leave-one-out sources, per-domain subgraphs, domain-mask contrastive + variance regularizer.
Saves ckpt for scripts/bridge/finetune.py / finetune_graph.py.

Examples:
  # Weights and aligners.pkl -> ckpts/bridge/{datasets}/ (--datasets names output subdir)
  python scripts/bridge/pretrain.py --target Cora --datasets cora_run
  python scripts/bridge/pretrain.py --domains Cora,Citeseer,Pubmed,Photo --datasets my_exp
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from pygfm.public.repo_paths import driver_script_repo_root

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import subgraph

ROOT = driver_script_repo_root(__file__)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore")

from pygfm.private.utlis.domain_alignment import DomainAlignment
from pygfm.baseline_models.bridge import BridgePrePromptModel
from pygfm.public.utils.runtime import (
    set_seed,
    load_all_datasets,
    bridge_preprompt_negative_tuples,
)
from pygfm.public.cli.yaml_config import parse_args_with_optional_yaml
from pygfm.public.cli.export_yaml import add_export_yaml_arguments, handle_export_args


def _parse():
    p = argparse.ArgumentParser(
        description="BRIDGE PrePrompt pretraining",
        epilog="YAML: -c PATH; export: --export-default-yaml / --export-run-yaml PATH (needs pyyaml)",
    )
    p.add_argument("--data_root", type=str, default=None, help="Default: GFM_DATA_ROOT or datasets/bridge")
    p.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Override default save root; default is always ckpts/bridge/{datasets}/",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Output subdir name: weights go to ckpts/bridge/{datasets}/; "
        "if omitted: lowercase --target when set, else default",
    )
    p.add_argument("--save_name", type=str, default="preprompt.pth")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unify_dim", type=int, default=50)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_neg", type=int, default=50)
    p.add_argument("--num_domains", type=int, default=4)
    p.add_argument(
        "--domains",
        type=str,
        default=None,
        help="Source domains for pretrain, comma-separated (use with or without --target; if neither, first num_domains)",
    )
    p.add_argument("--target", type=str, default=None, help="Leave-one-out target domain; others are sources")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max_epochs", type=int, default=10001)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--variance_weight", type=float, default=0.01)
    p.add_argument("--n_variance_samples", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--row_norm", action="store_true", help="L1 row-normalize features before PCA (typical BRIDGE)")
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--swanlab_project", type=str, default="gfm-toolbox-bridge")
    p.add_argument("--swanlab_run_name", type=str, default=None)
    p.add_argument("--log_interval", type=int, default=10)
    add_export_yaml_arguments(p)
    return p, parse_args_with_optional_yaml(p)


def main():
    p, args = _parse()
    handle_export_args(p, args)
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root or os.environ.get("GFM_DATA_ROOT", "datasets/bridge")

    def _safe_subdir(name: str) -> str:
        s = name.strip().replace(os.sep, "_").replace("/", "_").replace("\\", "_")
        return s if s else "default"

    if args.datasets:
        sub = _safe_subdir(args.datasets)
    elif args.target:
        sub = args.target.lower()
    else:
        sub = "default"
    if args.save_dir is None:
        args.save_dir = os.path.join("ckpts", "bridge", sub.lower())
    if args.target:
        args.save_name = f"preprompt_{args.target.lower()}.pth"

    use_swanlab = not args.no_swanlab
    if use_swanlab:
        try:
            import swanlab
            swanlab.init(
                project=args.swanlab_project,
                experiment_name=args.swanlab_run_name or f"bridge_pretrain_{args.target or 'all'}",
                config=vars(args),
            )
        except ImportError:
            use_swanlab = False

    all_raw = load_all_datasets(data_root=data_root)
    name2item = {d["name"]: d for d in all_raw}

    if args.target:
        if args.target not in name2item:
            raise ValueError(f"Unknown target {args.target!r}; available: {list(name2item)}")
        sources = [d for d in all_raw if d["name"] != args.target]
        ordered_names = [s["name"] for s in sources]
    elif args.domains:
        wanted = [x.strip() for x in args.domains.split(",") if x.strip()]
        sources = [name2item[n] for n in wanted]
        ordered_names = wanted
    else:
        sources = all_raw[: args.num_domains]
        ordered_names = [s["name"] for s in sources]

    print(f"BRIDGE PrePrompt | source domains: {ordered_names}")

    source_list = []
    all_rows_neg = []
    node_offset = 0
    pre_dir = Path("pre_data_bridge")
    pre_dir.mkdir(parents=True, exist_ok=True)
    aligners: list = []

    for idx, s in enumerate(sources):
        s_name = s["name"]
        feat_raw = s["ds"][0].x
        if torch.is_tensor(feat_raw):
            feat_np = feat_raw.cpu().numpy().astype(np.float64)
        else:
            feat_np = np.asarray(feat_raw, dtype=np.float64)
        if args.row_norm:
            rs = feat_np.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1.0
            feat_np = feat_np / rs

        aligner = DomainAlignment(n_components=args.unify_dim)
        aligner.fit(feat_np)
        aligners.append(aligner)
        aligned = aligner.transform(feat_np)
        pca_path = pre_dir / f"{s_name}_pca_{args.unify_dim}.npy"
        if not pca_path.exists():
            np.save(pca_path, aligned)

        x_t = torch.from_numpy(aligned.astype(np.float32)).to(device)
        e_ind = s["ds"][0].edge_index.numpy()
        adj = sp.coo_matrix(
            (np.ones(e_ind.shape[1]), (e_ind[0], e_ind[1])),
            shape=(x_t.size(0), x_t.size(0)),
        )
        local_neg = bridge_preprompt_negative_tuples(adj, args.num_neg)
        global_neg = local_neg.copy()
        global_neg[:, :] += node_offset
        all_rows_neg.append(global_neg)
        node_offset += x_t.size(0)
        source_list.append({"x": x_t, "domain_id": idx, "adj_raw": adj, "name": s_name})

    negative_samples = np.vstack(all_rows_neg)
    negative_samples_t = torch.from_numpy(negative_samples).long()

    processed = []
    neg_dict = {}
    curr = 0
    for s_dict in source_list:
        adj_coo = s_dict["adj_raw"]
        ei = torch.from_numpy(np.vstack((adj_coo.row, adj_coo.col))).long().to(device)
        d = Data(x=s_dict["x"], edge_index=ei)
        d.domain_id = torch.full((s_dict["x"].size(0),), s_dict["domain_id"], dtype=torch.long)
        processed.append(d)
        n = s_dict["x"].size(0)
        dom_neg = negative_samples_t[curr : curr + n] - curr
        dom_neg = torch.clamp(dom_neg, 0, n - 1)
        neg_dict[s_dict["name"]] = dom_neg.to(device)
        curr += n

    big = Batch.from_data_list(processed).to(device)

    model = BridgePrePromptModel(
        aligned_dim=args.unify_dim,
        hidden_dim=args.hidden_dim,
        num_sources=len(sources),
        dropout=args.dropout,
        variance_weight=args.variance_weight,
        n_samples=args.n_variance_samples,
        device=device,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    from tqdm import tqdm

    pbar = tqdm(range(args.max_epochs))
    for epoch in pbar:
        model.train()
        opt.zero_grad()
        epoch_loss = 0.0
        for idx, name in enumerate(ordered_names):
            mask = big.domain_id == idx
            if not mask.any():
                continue
            node_idx = mask.nonzero(as_tuple=False).view(-1)
            x_curr = big.x[mask]
            e_curr, _ = subgraph(node_idx, big.edge_index, relabel_nodes=True)
            loss_d = model(x_curr, e_curr, domain_id=idx, negative_samples=neg_dict[name])
            (loss_d / len(ordered_names)).backward()
            epoch_loss += (loss_d / len(ordered_names)).item()

        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if torch.isnan(gn) or torch.isinf(gn):
            opt.zero_grad()
        else:
            opt.step()

        if epoch % args.log_interval == 0:
            pbar.set_description(f"loss={epoch_loss:.4f}")
            if use_swanlab:
                try:
                    import swanlab
                    swanlab.log({"pretrain/loss": epoch_loss, "pretrain/grad_norm": float(gn)}, step=epoch)
                except Exception:
                    pass

    if not os.path.isabs(args.save_dir):
        args.save_dir = str(Path.cwd() / args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    out = os.path.join(args.save_dir, args.save_name)
    payload = {
        "model": model.state_dict(),
        "aligned_dim": args.unify_dim,
        "hidden_dim": args.hidden_dim,
        "num_sources": len(sources),
        "ordered_names": ordered_names,
        "variance_weight": args.variance_weight,
        "n_variance_samples": args.n_variance_samples,
    }
    if args.target:
        payload["target"] = args.target
    torch.save(payload, out)
    print(f"Saved: {out}")

    try:
        import joblib

        ap = os.path.join(args.save_dir, "aligners.pkl")
        joblib.dump({"aligners": aligners, "ordered_names": ordered_names}, ap)
        print(f"Saved aligners: {ap}")
    except Exception as e:
        print(f"!! aligners.pkl not saved: {e}")
    if use_swanlab:
        try:
            import swanlab
            swanlab.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
