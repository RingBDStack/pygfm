#!/usr/bin/env python
"""Original-style SAMGPT downstream node finetuning, packaged for pygfm CLI."""
from __future__ import annotations

import argparse
import os
import random
import sys
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from pygfm.public.repo_paths import driver_script_repo_root

warnings.filterwarnings("ignore")
ROOT = driver_script_repo_root(__file__)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pygfm.baseline_models import SAMGPTDownPromptModel, SAMGPTPrePromptModel
from pygfm.public.cli.default_ckpt import resolve_preprompt_ckpt
from pygfm.public.cli.export_yaml import add_export_yaml_arguments, handle_export_args
from pygfm.public.cli.yaml_config import parse_args_with_optional_yaml
from pygfm.public.utils.runtime import load_single_graph_dataset_or_reddit


def pca_compression(seq, k: int) -> np.ndarray:
    pca = PCA(n_components=k)
    seq = pca.fit_transform(seq)
    print(pca.explained_variance_ratio_.sum())
    return seq


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def _safe_torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _graph_to_feature_adj(data) -> tuple[np.ndarray, sp.csr_matrix]:
    features = data.x.detach().cpu().numpy()
    edge_index = data.edge_index.detach().cpu().numpy()
    adj = sp.coo_matrix(
        (np.ones(edge_index.shape[1], dtype=np.float32), (edge_index[0], edge_index[1])),
        shape=(features.shape[0], features.shape[0]),
    ).tocsr()
    return features, adj


def _seed_like_original(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _parse_args():
    p = argparse.ArgumentParser(
        description="Original-style SAMGPT DownPrompt node finetune",
        epilog="YAML: -c PATH; export: --export-default-yaml / --export-run-yaml PATH (needs pyyaml)",
    )
    p.add_argument("--dataset", type=str, default="Cora")
    p.add_argument("--k_shot", type=int, default=1, choices=[1, 5])
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--downstream_root", type=str, default="downstream_data/samgpt")
    p.add_argument("--splits_path", type=str, default=None)
    p.add_argument("--split_id", type=int, default=0)
    p.add_argument("--task_num", type=int, default=0)
    p.add_argument("--run_all_splits", action="store_true")
    p.add_argument("--data_root", type=str, default="datasets/samgpt")
    p.add_argument("--seed", type=int, default=39)
    p.add_argument("--unify_dim", type=int, default=50)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--test_reserve", type=int, default=100)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--prompt_mode", type=str, default="mul", choices=["add", "mul"])
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--ablation_down", type=str, default="all")
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--swanlab_project", type=str, default="gfm-toolbox-samgpt")
    p.add_argument("--swanlab_run_name", type=str, default=None)
    add_export_yaml_arguments(p)
    return p, parse_args_with_optional_yaml(p)


def main():
    p, args = _parse_args()
    handle_export_args(p, args)
    args.ckpt = resolve_preprompt_ckpt(ROOT, "samgpt", args.dataset, args.ckpt)
    torch.cuda.empty_cache()
    _seed_like_original(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    xent = nn.CrossEntropyLoss()

    use_swanlab = not args.no_swanlab
    if use_swanlab:
        try:
            import swanlab

            run_name = args.swanlab_run_name or f"finetune_{args.dataset}_{args.k_shot}shot"
            swanlab.init(project=args.swanlab_project, experiment_name=run_name, config=vars(args))
        except ImportError:
            use_swanlab = False

    ckpt = _safe_torch_load(args.ckpt, map_location=device)
    if not (isinstance(ckpt, dict) and "model" in ckpt):
        raise TypeError(
            "SAMGPT finetune expects a pygfm-style checkpoint dict with a 'model' field."
        )

    state_dict = ckpt["model"]
    num_domains = int(ckpt.get("num_domains", 1))
    num_layers = int(ckpt.get("num_layers", args.num_layers))
    unify_dim = int(ckpt.get("unify_dim", args.unify_dim))
    hidden_dim = int(ckpt.get("hidden_dim", args.hidden_dim))
    prompt_mode = str(ckpt.get("prompt_mode", args.prompt_mode))
    alpha = float(ckpt.get("alpha", args.alpha))
    pretrain_method = str(ckpt.get("pretrain_method", "GRAPHCL"))
    lp_mode = pretrain_method == "LP"

    preprompt = SAMGPTPrePromptModel(
        input_dim=unify_dim,
        hidden_dim=hidden_dim,
        num_domains=num_domains,
        num_layers=num_layers,
        prompt_mode=prompt_mode,
        temperature=1.0,
        alpha=alpha,
        device=device,
    )
    missing, unexpected = preprompt.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f">> Checkpoint load note: missing={len(missing)}, unexpected={len(unexpected)}")
    preprompt = preprompt.to(device)
    preprompt.eval()

    fea_weights, str_weights, combines = preprompt.get_weights()
    combines.append(args.beta)

    data, num_classes = load_single_graph_dataset_or_reddit(
        args.data_root,
        args.dataset,
    )
    feature_np, adj_sp = _graph_to_feature_adj(data)
    features = torch.FloatTensor(pca_compression(feature_np, k=unify_dim)).to(device)
    adj = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj_sp + sp.eye(adj_sp.shape[0]))).to(device)
    labels = data.y.to(device).long()

    preprompt.embed(features, adj, True, None, lp_mode)

    if args.splits_path is not None:
        splits_path = args.splits_path
    else:
        splits_path = os.path.join(args.downstream_root, args.dataset, f"{args.k_shot}shot", "splits.pt")
    down_data = _safe_torch_load(splits_path, map_location="cpu")
    splits = down_data["splits"]

    if args.run_all_splits:
        split_ids = list(range(len(splits)))
    elif args.task_num and args.task_num > 0:
        split_ids = list(range(min(args.task_num, len(splits))))
    else:
        if not (0 <= args.split_id < len(splits)):
            raise IndexError(f"split_id {args.split_id} out of range")
        split_ids = [args.split_id]

    test_start = max(0, int(labels.shape[0] - args.test_reserve))
    idx_test = range(test_start, labels.shape[0])
    test_lbls = labels[idx_test]
    acc_list = []
    cnt_wait = 0

    for sid in split_ids:
        split = splits[sid]
        idx_train = torch.as_tensor(split["indices"], dtype=torch.long, device=device)
        lbls_train = torch.as_tensor(split["labels"], dtype=torch.long, device=device).view(-1)

        log = SAMGPTDownPromptModel(
            gcn=preprompt.gcn,
            input_dim=unify_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            fea_pretext_weights=fea_weights,
            str_pretext_weights=str_weights,
            combines=combines,
            prompt_mode=prompt_mode,
            ablation=args.ablation_down,
            device=device,
        ).to(device)
        opt = torch.optim.Adam(log.parameters(), lr=args.lr)
        best_loss = float("inf")

        print(
            f">> SAMGPT Finetune {args.dataset} | {args.k_shot}-shot | split {sid} | "
            f"support={len(idx_train)}, test={len(test_lbls)}"
        )
        for step in range(args.max_steps):
            log.train()
            opt.zero_grad()
            logits = log(features, adj, True, preprompt.gcn, idx_train, lbls_train, 1).float()
            loss = xent(logits, lbls_train)
            loss_value = float(loss.detach().item())
            if loss_value < best_loss:
                best_loss = loss_value
                cnt_wait = 0
            else:
                cnt_wait += 1
            if cnt_wait == args.patience:
                print(f"  Early stopping at step {step}, best loss={best_loss:.4f}")
                break
            loss.backward()
            opt.step()

        log.eval()
        with torch.inference_mode():
            logits = log(features, adj, True, preprompt.gcn, torch.as_tensor(list(idx_test), device=device))
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        acc_value = float(acc.item())
        acc_list.append(acc_value)
        print(f"[{args.dataset}] {args.k_shot}-shot split {sid} test accuracy: {acc_value:.4f}")

    if len(acc_list) > 1:
        acc_tensor = torch.tensor(acc_list)
        print(
            f"[{args.dataset}] {args.k_shot}-shot {len(acc_list)} splits "
            f"mean acc: {acc_tensor.mean():.4f}, std: {acc_tensor.std():.4f}"
        )


if __name__ == "__main__":
    main()
