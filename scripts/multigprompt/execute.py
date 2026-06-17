#!/usr/bin/env python3
"""
MultiGPrompt training / few-shot experiment entrypoint (scripts layer).

Run from repository root:

  python scripts/multigprompt/execute.py --dataset cora --gpu 0

Core model code lives under ``pygfm/baseline_models/multigprompt/``; this file only
orchestrates the experiment.
"""
from __future__ import annotations

import csv
import os
import random
import sys
from pathlib import Path
from pygfm.public.repo_paths import driver_script_repo_root

# Repo root on sys.path so ``pygfm`` imports work without a prior editable install.
_ROOT = driver_script_repo_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from pygfm.baseline_models.multigprompt import aug, preprompt
from pygfm.baseline_models.multigprompt.downprompt import downprompt, featureprompt
from pygfm.baseline_models.multigprompt.paths import (
    get_default_pretrain_ckpt_path,
    get_downstream_data_dir,
    get_repo_root,
)
from pygfm.baseline_models.multigprompt.preprompt import PrePrompt
from pygfm.baseline_models.multigprompt.utils import process
from pygfm.public.cli.multigprompt_config import parse_args


def fewshot_task_paths(
    data_dir: Path, dataset: str, shotnum: int, task_idx: int
) -> tuple[Path, Path]:
    """
    Few-shot task files (same layout as upstream, parameterized by ``dataset``).

    ``<data_dir>/fewshot_<dataset>/<shotnum>-shot_<dataset>/<task_idx>/idx.pt``
    ``<data_dir>/fewshot_<dataset>/<shotnum>-shot_<dataset>/<task_idx>/labels.pt``
    """
    base = (
        data_dir / f"fewshot_{dataset}" / f"{shotnum}-shot_{dataset}" / str(task_idx)
    )
    return base / "idx.pt", base / "labels.pt"


def _torch_load_split_file(path: Path, *, map_location=None):
    kw = {}
    if map_location is not None:
        kw["map_location"] = map_location
    try:
        return torch.load(path, weights_only=False, **kw)
    except TypeError:
        return torch.load(path, **kw)


def _dedupe_path_list(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _collect_splits_pt_candidates(
    downstream_dir: Path,
    dataset: str,
    shotnum: int,
    *,
    splits_path: str | None,
) -> list[Path]:
    """
    Try ``splits.pt`` in order: explicit arg, current downstream dir, multigprompt case variants,
    shared MDGPT splits, ``downstream_data/<dataset>/{k}shot``.
    """
    ds = dataset.lower()
    shot = f"{shotnum}shot"
    cands: list[Path] = []

    if splits_path:
        raw = Path(splits_path).expanduser()
        if not raw.is_absolute():
            raw = (_ROOT / raw).resolve()
        else:
            raw = raw.resolve()
        if raw.is_file():
            cands.append(raw)
        else:
            cands.append(raw / "splits.pt")

    cands.append(downstream_dir / shot / "splits.pt")

    rr = get_repo_root()
    mgp = rr / "downstream_data" / "multigprompt"
    if mgp.is_dir():
        for child in sorted(mgp.iterdir()):
            if child.is_dir() and child.name.casefold() == ds:
                cands.append(child / shot / "splits.pt")

    cands.append(rr / "downstream_data" / "mdgpt" / ds / shot / "splits.pt")
    cands.append(rr / "downstream_data" / ds / shot / "splits.pt")

    return _dedupe_path_list(cands)


def resolve_fewshot_mode(
    downstream_dir: Path,
    dataset: str,
    shotnum: int,
    *,
    splits_path: str | None = None,
) -> tuple[str, list | None]:
    """
    Return (\"files\", None) for fewshot_<ds>/<k>-shot_<ds>/<i>/idx.pt + labels.pt;
    or (\"splits\", splits_list) for toolbox ``splits.pt`` (list entries with indices, labels).
    """
    idx0, lbl0 = fewshot_task_paths(downstream_dir, dataset, shotnum, 0)
    if idx0.is_file() and lbl0.is_file():
        return "files", None

    candidates = _collect_splits_pt_candidates(
        downstream_dir, dataset, shotnum, splits_path=splits_path
    )
    for splits_pt in candidates:
        if not splits_pt.is_file():
            continue
        blob = _torch_load_split_file(splits_pt, map_location="cpu")
        splits = blob.get("splits")
        if not isinstance(splits, list):
            raise ValueError(
                f"{splits_pt} must contain a list under key 'splits' (toolbox few-shot format)."
            )
        return "splits", splits

    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "MultiGPrompt few-shot data not found. Expected legacy idx+labels under fewshot_* "
        "or toolbox splits.pt. Searched:\n"
        f"{tried}\n"
        f"Legacy reference: {idx0} + {lbl0}\n"
        "Tip: generate with downstream_data_gen, or pass --splits_path / YAML splits_path "
        "to your splits.pt or its .../1shot directory."
    )


def load_fewshot_task_tensors(
    downstream_dir: Path,
    dataset: str,
    shotnum: int,
    task_idx: int,
    *,
    mode: str,
    splits_list: list | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "splits":
        assert splits_list is not None
        if task_idx < 0 or task_idx >= len(splits_list):
            raise IndexError(
                f"few-shot task_idx={task_idx} out of range for splits.pt (len={len(splits_list)})."
            )
        s = splits_list[task_idx]
        indices = s.get("indices", s.get("idx"))
        if indices is None:
            raise KeyError(f"split {task_idx} needs 'indices' or 'idx'")
        labels = s.get("labels")
        if labels is None:
            raise KeyError(f"split {task_idx} needs 'labels'")
        idx_train = torch.as_tensor(indices, dtype=torch.long, device=device)
        train_lbls = torch.as_tensor(labels, dtype=torch.long, device=device).squeeze()
        return idx_train, train_lbls

    idx_path, labels_path = fewshot_task_paths(
        downstream_dir, dataset, shotnum, task_idx
    )
    idx_train = (
        _torch_load_split_file(idx_path, map_location=device).type(torch.long).to(device)
    )
    train_lbls = (
        _torch_load_split_file(labels_path, map_location=device)
        .type(torch.long)
        .squeeze()
        .to(device)
    )
    return idx_train, train_lbls


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv, script_file=Path(__file__))

    print("-" * 100)
    print(args)
    print("-" * 100)

    dataset = args.dataset
    downstream_dir = get_downstream_data_dir(dataset)
    drop_percent = args.drop_percent
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 1
    nb_epochs = args.epochs
    patience = 20
    l2_coef = 0.0
    sparse = True

    nonlinearity = "prelu"  # special name to separate parameters
    adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)

    features, _ = process.preprocess_features(features)
    negetive_sample = preprompt.prompt_pretrain_sample(adj, 200)

    nb_nodes = features.shape[0]  # node number
    ft_size = features.shape[1]  # node features dim
    nb_classes = labels.shape[1]  # classes = 6

    features = torch.FloatTensor(features[np.newaxis])

    aug_features1edge = features
    aug_features2edge = features

    aug_adj1edge = aug.aug_random_edge(adj, drop_percent=drop_percent)
    aug_adj2edge = aug.aug_random_edge(adj, drop_percent=drop_percent)

    aug_features1mask = aug.aug_random_mask(features, drop_percent=drop_percent)
    aug_features2mask = aug.aug_random_mask(features, drop_percent=drop_percent)

    aug_adj1mask = adj
    aug_adj2mask = adj

    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    aug_adj1edge = process.normalize_adj(aug_adj1edge + sp.eye(aug_adj1edge.shape[0]))
    aug_adj2edge = process.normalize_adj(aug_adj2edge + sp.eye(aug_adj2edge.shape[0]))

    aug_adj1mask = process.normalize_adj(aug_adj1mask + sp.eye(aug_adj1mask.shape[0]))
    aug_adj2mask = process.normalize_adj(aug_adj2mask + sp.eye(aug_adj2mask.shape[0]))

    if sparse:
        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
        sp_aug_adj1edge = process.sparse_mx_to_torch_sparse_tensor(aug_adj1edge)
        sp_aug_adj2edge = process.sparse_mx_to_torch_sparse_tensor(aug_adj2edge)

        sp_aug_adj1mask = process.sparse_mx_to_torch_sparse_tensor(aug_adj1mask)
        sp_aug_adj2mask = process.sparse_mx_to_torch_sparse_tensor(aug_adj2mask)

    else:
        adj = (adj + sp.eye(adj.shape[0])).todense()
        aug_adj1edge = (aug_adj1edge + sp.eye(aug_adj1edge.shape[0])).todense()
        aug_adj2edge = (aug_adj2edge + sp.eye(aug_adj2edge.shape[0])).todense()

        aug_adj1mask = (aug_adj1mask + sp.eye(aug_adj1mask.shape[0])).todense()
        aug_adj2mask = (aug_adj2mask + sp.eye(aug_adj2mask.shape[0])).todense()

    if not sparse:
        adj = torch.FloatTensor(adj[np.newaxis])
        aug_adj1edge = torch.FloatTensor(aug_adj1edge[np.newaxis])
        aug_adj2edge = torch.FloatTensor(aug_adj2edge[np.newaxis])
        aug_adj1mask = torch.FloatTensor(aug_adj1mask[np.newaxis])
        aug_adj2mask = torch.FloatTensor(aug_adj2mask[np.newaxis])

    labels = torch.FloatTensor(labels[np.newaxis])
    idx_train = torch.LongTensor(idx_train)
    print("adj", sp_adj.shape)
    print("feature", features.shape)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    LP = False

    print("")

    lista4 = [0.0001]

    list1 = [256]
    list2 = [0.0001]
    save_name = args.save_name
    if save_name is None:
        save_name = str(get_default_pretrain_ckpt_path(dataset))
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    num_tasks = args.tasks
    inner_steps = args.inner_steps
    for lr in list2:
        for hid_units in list1:
            for a4 in lista4:
                a1 = 0.9
                a2 = 0.9
                a3 = 0.1
                model = PrePrompt(
                    ft_size,
                    hid_units,
                    nonlinearity,
                    negetive_sample,
                    a1,
                    a2,
                    a3,
                    a4,
                    1,
                    0.3,
                )
                optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
                if torch.cuda.is_available():
                    print("Using CUDA")
                model = model.to(device)
                features = features.to(device)
                aug_features1edge = aug_features1edge.to(device)
                aug_features2edge = aug_features2edge.to(device)
                aug_features1mask = aug_features1mask.to(device)
                aug_features2mask = aug_features2mask.to(device)
                if sparse:
                    sp_adj = sp_adj.to(device)
                    sp_aug_adj1edge = sp_aug_adj1edge.to(device)
                    sp_aug_adj2edge = sp_aug_adj2edge.to(device)
                    sp_aug_adj1mask = sp_aug_adj1mask.to(device)
                    sp_aug_adj2mask = sp_aug_adj2mask.to(device)
                else:
                    adj = adj.to(device)
                    aug_adj1edge = aug_adj1edge.to(device)
                    aug_adj2edge = aug_adj2edge.to(device)
                    aug_adj1mask = aug_adj1mask.to(device)
                    aug_adj2mask = aug_adj2mask.to(device)
                labels = labels.to(device)
                idx_train = idx_train.to(device)
                idx_val = idx_val.to(device)
                idx_test = idx_test.to(device)
                b_xent = nn.BCEWithLogitsLoss()
                xent = nn.CrossEntropyLoss()
                cnt_wait = 0
                best = 1e9
                best_t = 0
                for epoch in range(nb_epochs):
                    model.train()
                    optimiser.zero_grad()
                    idx = np.random.permutation(nb_nodes)
                    shuf_fts = features[:, idx, :]
                    lbl_1 = torch.ones(batch_size, nb_nodes)
                    lbl_2 = torch.zeros(batch_size, nb_nodes)
                    lbl = torch.cat((lbl_1, lbl_2), 1)
                    shuf_fts = shuf_fts.to(device)
                    lbl = lbl.to(device)
                    loss = model(
                        features,
                        shuf_fts,
                        aug_features1edge,
                        aug_features2edge,
                        aug_features1mask,
                        aug_features2mask,
                        sp_adj if sparse else adj,
                        sp_aug_adj1edge if sparse else aug_adj1edge,
                        sp_aug_adj2edge if sparse else aug_adj2edge,
                        sp_aug_adj1mask if sparse else aug_adj1mask,
                        sp_aug_adj2mask if sparse else aug_adj2mask,
                        sparse,
                        None,
                        None,
                        None,
                        lbl=lbl,
                    )
                    print("Loss:[{:.4f}]".format(loss.item()))
                    if loss < best:
                        best = loss
                        best_t = epoch
                        cnt_wait = 0
                        torch.save(model.state_dict(), save_name)
                    else:
                        cnt_wait += 1
                    if cnt_wait == patience:
                        print("Early stopping!")
                        break
                    loss.backward()
                    optimiser.step()
                print("Loading {}th epoch".format(best_t))
                model.load_state_dict(torch.load(save_name, map_location=device))
                model.eval()
                embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None, LP)
                dgiprompt = model.dgi.prompt
                graphcledgeprompt = model.graphcledge.prompt
                lpprompt = model.lp.prompt
                preval_embs = embeds[0, idx_val]
                test_embs = embeds[0, idx_test]
                val_lbls = torch.argmax(labels[0, idx_val], dim=1)
                test_lbls = torch.argmax(labels[0, idx_test], dim=1)
                tot = torch.zeros(1)
                tot = tot.to(device)
                accs = []

                print("-" * 100)
                cnt_wait = 0
                best = 1e9
                best_t = 0
                for shotnum in range(1, 2):
                    tot = torch.zeros(1)
                    tot = tot.to(device)
                    accs = []
                    print("shotnum", shotnum)
                    fs_mode, fs_splits = resolve_fewshot_mode(
                        downstream_dir,
                        dataset,
                        shotnum,
                        splits_path=getattr(args, "splits_path", None),
                    )
                    for i in range(num_tasks):
                        idx_train, train_lbls = load_fewshot_task_tensors(
                            downstream_dir,
                            dataset,
                            shotnum,
                            i,
                            mode=fs_mode,
                            splits_list=fs_splits,
                            device=device,
                        )
                        pretrain_embs = embeds[0, idx_train]
                        print("true", i, train_lbls)
                        feature_prompt = featureprompt(
                            model.dgiprompt.prompt,
                            model.graphcledgeprompt.prompt,
                            model.lpprompt.prompt,
                        ).to(device)
                        log = downprompt(
                            dgiprompt,
                            graphcledgeprompt,
                            lpprompt,
                            a4,
                            hid_units,
                            nb_classes,
                            embeds,
                            train_lbls,
                        )
                        opt = torch.optim.Adam(
                            [*log.parameters(), *feature_prompt.parameters()],
                            lr=0.001,
                        )
                        log.to(device)
                        best = 1e9
                        pat_steps = 0
                        best_acc = torch.zeros(1)
                        best_acc = best_acc.to(device)
                        cnt_wait = 0
                        for _ in range(inner_steps):
                            log.train()
                            opt.zero_grad()
                            prompt_feature = feature_prompt(features)
                            embeds1 = model.gcn(
                                prompt_feature,
                                sp_adj if sparse else adj,
                                sparse,
                                LP,
                            )
                            pretrain_embs1 = embeds1[0, idx_train]
                            logits = log(pretrain_embs, pretrain_embs1, 1).float().to(device)
                            loss = xent(logits, train_lbls)
                            if loss < best:
                                best = loss
                                cnt_wait = 0
                            else:
                                cnt_wait += 1
                            if cnt_wait == patience:
                                print("Early stopping!")
                                break

                            loss.backward(retain_graph=True)
                            opt.step()
                        prompt_feature = feature_prompt(features)
                        embeds1, _ = model.embed(
                            prompt_feature,
                            sp_adj if sparse else adj,
                            sparse,
                            None,
                            LP,
                        )
                        test_embs1 = embeds1[0, idx_test]
                        logits = log(test_embs, test_embs1)
                        preds = torch.argmax(logits, dim=1)
                        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
                        accs.append(acc * 100)
                        tot += acc

                    print("-" * 100)
                    denom = max(1, num_tasks)
                    print("Average accuracy:[{:.4f}]".format(tot.item() / denom))
                    accs_tensor = torch.stack(accs)
                    mean_acc = accs_tensor.mean().item()
                    # With one task, sample std has no dof; torch may warn and return nan
                    acc_std = (
                        0.0
                        if accs_tensor.numel() <= 1
                        else accs_tensor.std(unbiased=False).item()
                    )
                    print("Mean:[{:.4f}]".format(mean_acc))
                    print("Std :[{:.4f}]".format(acc_std))
                    print("-" * 100)
                    row = [
                        shotnum,
                        lr,
                        LP,
                        hid_units,
                        a1,
                        a2,
                        a3,
                        a4,
                        mean_acc,
                        acc_std,
                    ]
                    csv_path = downstream_dir / f"{dataset}_fewshot.csv"
                    downstream_dir.mkdir(parents=True, exist_ok=True)
                    with open(csv_path, "a", newline="") as out:
                        csv_writer = csv.writer(out, dialect="excel")
                        csv_writer.writerow(row)


if __name__ == "__main__":
    main()
