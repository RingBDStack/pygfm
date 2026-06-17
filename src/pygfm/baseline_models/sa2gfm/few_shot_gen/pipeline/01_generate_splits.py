#!/usr/bin/env python3
"""
Generate few-shot train splits for downstream (same format as legacy generate_few_data).

Writes: {SA2GFM_DATA_ROOT}/few_shot/{dataset}/{k}shot/split_{i}.pt
Each file: {"indices": [...], "labels": [...]} — matches down_all_sparse_multi load_few_shot_data.

Reads labels from ori/{dataset}.pt field `y` (excludes last test_reserve nodes from sampling pool).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pygfm.baseline_models.sa2gfm.paths import paths
from pygfm.public.utils import set_seed
from pygfm.public.cli.yaml_config import parse_args_with_config
from pygfm.public.utils.runtime import _maybe_hetero_to_homogeneous, _torch_load_to_single_data


def _load_graph(data_path: Path):
    try:
        raw = torch.load(data_path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(data_path, map_location="cpu")
    raw = _maybe_hetero_to_homogeneous(raw)
    data = _torch_load_to_single_data(raw)
    data = _maybe_hetero_to_homogeneous(data)
    if getattr(data, "y", None) is None:
        raise ValueError(
            f"No node labels `y` after load ({data_path!r}); few-shot split generation requires `y`."
        )
    return data


def get_label_indices(labels, n_way: int, exclude_last_n: int):
    valid_indices = list(range(len(labels) - exclude_last_n))
    label_to_indices: defaultdict = defaultdict(list)
    for idx in valid_indices:
        label = int(labels[idx].item())
        label_to_indices[label].append(idx)

    unique_labels = sorted(label_to_indices.keys())
    if n_way > 0 and n_way < len(unique_labels):
        selected_labels = unique_labels[:n_way]
        label_to_indices = {k: v for k, v in label_to_indices.items() if k in selected_labels}
    else:
        selected_labels = unique_labels

    return label_to_indices, selected_labels


def generate_few_shot_split(label_to_indices, selected_labels, k_shot: int):
    support_indices = []
    for label in selected_labels:
        pool = label_to_indices[label]
        if len(pool) < k_shot:
            print(f"Warning: class {label} has only {len(pool)} samples, fewer than requested {k_shot}")
            selected = pool
        else:
            selected = np.random.choice(pool, k_shot, replace=False).tolist()
        support_indices.extend(selected)
    return support_indices


def parse_args():
    p = argparse.ArgumentParser(description="Generate few-shot splits under data/few_shot/")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--k-shot", type=int, dest="k_shot", choices=[1, 5], required=True)
    p.add_argument("--n-splits", type=int, default=20, help="Number of split_*.pt files (>= downstream --num-splits)")
    p.add_argument("--n-way", type=int, default=0, help="0 = all classes; else first n-way labels only")
    p.add_argument("--test-reserve", type=int, default=1000, help="Exclude last N nodes from few-shot pool (test band)")
    p.add_argument("--seed", type=int, default=39)
    p.add_argument(
        "--write-example",
        action="store_true",
        help="Write example_structure.txt under dataset few_shot folder",
    )
    return parse_args_with_config(p, script_file=Path(__file__))


def main():
    args = parse_args()
    set_seed(args.seed)

    data_path = paths.resolve_ori_graph_pt(args.dataset)

    data = _load_graph(data_path)
    labels = data.y
    num_nodes = int(labels.shape[0])

    label_to_indices, selected_labels = get_label_indices(
        labels, n_way=args.n_way, exclude_last_n=args.test_reserve
    )

    train_nodes = num_nodes - args.test_reserve
    print(f"Dataset: {args.dataset}  path={data_path}")
    print(f"Total nodes: {num_nodes}  pool size (excluding last {args.test_reserve}): {train_nodes}")
    print(f"Num classes: {len(selected_labels)}  labels={selected_labels}")
    for lb in selected_labels:
        print(f"  - label {lb}: {len(label_to_indices[lb])} samples")

    out_root = paths.few_shot_dir / args.dataset
    shot_dir = out_root / f"{args.k_shot}shot"
    shot_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {args.n_splits} {args.k_shot}-shot splits -> {shot_dir}")

    for split_idx in tqdm(range(args.n_splits)):
        support_indices = generate_few_shot_split(label_to_indices, selected_labels, args.k_shot)
        support_labels = [int(labels[i].item()) for i in support_indices]
        split_data = {"indices": support_indices, "labels": support_labels}
        torch.save(split_data, shot_dir / f"split_{split_idx}.pt")

    print(f"Done: {args.n_splits} files")

    if args.write_example:
        example_file = out_root / "example_structure.txt"
        example_file.write_text(
            f"Dataset: {args.dataset}\n"
            f"Few-shot: {args.k_shot}-shot, {len(selected_labels)}-way\n\n"
            "Each split_*.pt:\n"
            "  'indices': training node indices\n"
            "  'labels':  integer labels aligned with indices\n\n"
            f"Example load: torch.load('{shot_dir}/split_0.pt')\n",
            encoding="utf-8",
        )
        print(f"Wrote: {example_file}")


if __name__ == "__main__":
    main()
