#!/usr/bin/env python3
"""
Analyze `{dataset}_communities.pt`: size distribution, coverage vs `ori/{dataset}.pt`.
Portable version of `analyze/analyze_communities.py` (no hard-coded home paths).
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from pygfm.baseline_models.sa2gfm.attack_data_gen.lib.data_utils import load_graph
from pygfm.baseline_models.sa2gfm.paths import paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    cf = paths.communities_dir / f"{args.dataset}_communities.pt"
    if not cf.is_file():
        raise FileNotFoundError(cf)
    try:
        comm_data = torch.load(cf, map_location="cpu", weights_only=False)
    except TypeError:
        comm_data = torch.load(cf, map_location="cpu")
    all_communities = comm_data["communities"]
    print(f"Loaded {cf}: {len(all_communities)} communities")

    graph_data = load_graph(args.dataset)
    total_nodes = graph_data.x.shape[0]

    valid = [c for c in all_communities if len(c) > 1]
    singletons = [c for c in all_communities if len(c) == 1]
    sizes = np.array([len(c) for c in valid], dtype=int)

    print(f"nodes in graph: {total_nodes}")
    print(f"communities size>1: {len(valid)}, singleton communities: {len(singletons)}")
    if sizes.size:
        print(f"size>1 stats: min={sizes.min()} max={sizes.max()} mean={sizes.mean():.2f} median={np.median(sizes):.2f}")

    flat = [node for c in all_communities for node in c]
    uniq = set(flat)
    print(f"unique nodes covered: {len(uniq)}")
    if len(flat) != len(uniq):
        print("WARNING: duplicate node assignment across communities")
    if len(uniq) == total_nodes and len(flat) == len(uniq):
        print("COVERAGE: full partition OK")
    else:
        print("COVERAGE: FAIL (missing or duplicate nodes)")

    if args.plot and sizes.size:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.hist(sizes, bins=50, alpha=0.75)
        plt.title(f"Community sizes (>1) — {args.dataset}")
        plt.xlabel("size")
        plt.ylabel("count")
        plt.yscale("log")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()
