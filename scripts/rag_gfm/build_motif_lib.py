#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG-GFM step 2 (optional): build motif library.

Train a subgraph encoder per dataset and build motif nano-vectordb for downstream motif retrieval.
Graph data: .pt or processed/data.pt with x and edge_index under data_root.

Usage:
  python scripts/rag_gfm/build_motif_lib.py
  python scripts/rag_gfm/build_motif_lib.py --data_root datasets/rag_gfm --datasets Cora,Citeseer,Pubmed
  python scripts/rag_gfm/build_motif_lib.py --motif_lib_path downstream_data/rag_gfm/motif_lib --epochs 200
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

from pygfm.private.utlis.rag_gfm import MotifLibBuilderConfig, build_motif_lib
from pygfm.public.cli.yaml_config import parse_args_with_optional_yaml


def main():
    p = argparse.ArgumentParser(
        description="RAG-GFM motif library: train subgraph encoder and build motif_vectordb"
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="datasets/rag_gfm",
        help="Graph data root",
    )
    p.add_argument(
        "--motif_lib_path",
        type=str,
        default="downstream_data/rag_gfm/motif_lib",
        help="Motif library root (per-dataset subdirs: encoder.pth, config.pth, motif_vectordb.json)",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="Cora,Citeseer,Pubmed",
        help="Dataset names, comma-separated",
    )
    p.add_argument("--top_k", type=int, default=200, help="Top-K central nodes")
    p.add_argument("--epochs", type=int, default=200, help="Encoder training epochs")
    p.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--hidden_dim", type=int, default=64, help="Encoder hidden dim")
    p.add_argument("--output_dim", type=int, default=32, help="Encoder output dim")
    p.add_argument("--device", type=str, default="cuda", help="Device")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--no_cse_cache", action="store_true", help="Disable CSE cache")
    args = parse_args_with_optional_yaml(p)

    # Relative paths resolved from repo root
    data_root = args.data_root
    if not os.path.isabs(data_root):
        data_root = str(ROOT / data_root)
    motif_lib_path = args.motif_lib_path
    if not os.path.isabs(motif_lib_path):
        motif_lib_path = str(ROOT / motif_lib_path)

    config = MotifLibBuilderConfig(
        data_root=data_root,
        dataset_names=[s.strip() for s in args.datasets.split(",") if s.strip()],
        motif_lib_path=motif_lib_path,
        top_k=args.top_k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        device=args.device,
        seed=args.seed,
        use_cse_cache=not args.no_cse_cache,
    )
    paths = build_motif_lib(config)
    print(f"RAG-GFM motif library written for {len(paths)} dataset(s):")
    for path in paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
