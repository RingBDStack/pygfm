#!/usr/bin/env python3
"""
Detect disjoint communities on an undirected graph and save
`{SA2GFM_DATA_ROOT}/communities/{dataset}_communities.pt`
compatible with `down_all_sparse_multi` (key `communities`: list of node-id lists).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import torch

from pygfm.baseline_models.sa2gfm.attack_data_gen.lib.data_utils import load_graph
from pygfm.baseline_models.sa2gfm.paths import paths
from pygfm.public.cli.yaml_config import parse_args_with_config


def edge_index_to_graph(edge_index: torch.Tensor, num_nodes: int) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    ei = edge_index.cpu().numpy()
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        if u != v:
            g.add_edge(u, v)
    return g


def partition_to_communities(partition: list) -> list[list[int]]:
    """Stable list-of-lists: each community sorted, communities ordered by min node id."""
    comms = [sorted(int(x) for x in block) for block in partition]
    comms.sort(key=lambda c: c[0] if c else 0)
    return comms


def detect_communities(g: nx.Graph, method: str, seed: int, resolution: float):
    if method == "louvain":
        try:
            part = nx.community.louvain_communities(g, resolution=resolution, seed=seed)
        except AttributeError as e:
            raise RuntimeError(
                "Louvain requires networkx>=3.2 with nx.community.louvain_communities. "
                "Upgrade networkx or choose --method greedy_modularity."
            ) from e
        return partition_to_communities(part)

    if method == "greedy_modularity":
        part = nx.community.greedy_modularity_communities(g, resolution=resolution)
        return partition_to_communities(part)

    if method == "label_propagation":
        part = nx.community.label_propagation_communities(g)
        return partition_to_communities(part)

    raise ValueError(f"unknown method: {method}")


def validate_partition(comms: list[list[int]], num_nodes: int) -> None:
    seen = []
    for c in comms:
        seen.extend(c)
    if len(seen) != len(set(seen)):
        raise ValueError("partition has duplicate node ids")
    if set(seen) != set(range(num_nodes)):
        missing = set(range(num_nodes)) - set(seen)
        extra = set(seen) - set(range(num_nodes))
        raise ValueError(f"bad cover: missing={len(missing)} extra={len(extra)}")


def main():
    parser = argparse.ArgumentParser(description="SA2GFM community detection")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--method",
        type=str,
        default="louvain",
        choices=["louvain", "greedy_modularity", "label_propagation"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Louvain / greedy_modularity resolution (NetworkX semantics).",
    )
    args = parse_args_with_config(parser, script_file=Path(__file__))

    data = load_graph(args.dataset)
    n = int(data.num_nodes) if hasattr(data, "num_nodes") else data.x.shape[0]
    g = edge_index_to_graph(data.edge_index, n)
    comms = detect_communities(g, args.method, args.seed, args.resolution)
    validate_partition(comms, n)

    paths.communities_dir.mkdir(parents=True, exist_ok=True)
    out = paths.communities_dir / f"{args.dataset}_communities.pt"
    payload = {
        "communities": comms,
        "meta": {
            "dataset": args.dataset,
            "method": args.method,
            "seed": args.seed,
            "resolution": args.resolution,
            "num_nodes": n,
            "num_communities": len(comms),
        },
    }
    torch.save(payload, out)
    print(f"Saved {out} ({len(comms)} communities)")


if __name__ == "__main__":
    main()
