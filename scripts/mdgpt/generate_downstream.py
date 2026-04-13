#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI: generate downstream few-shot splits.

Usage:
  python scripts/mdgpt/generate_downstream.py few_shot --dataset Cora --k_shot 5
  python scripts/mdgpt/generate_downstream.py -c scripts/mdgpt/generate_downstream.yaml
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_SCRIPTS_MDGPT = Path(__file__).resolve().parent
if str(_SCRIPTS_MDGPT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_MDGPT))

from pygfm.private.utlis.downstream_data_gen import (
    generate_few_shot_splits,
    generate_graph_batch_splits,
    DownstreamGeneratorConfig,
)

from config_utils import parse_args_with_config


def _parse():
    p = argparse.ArgumentParser(description="Generate downstream few-shot splits")
    p.add_argument(
        "mode",
        nargs="?",
        default=None,
        choices=["few_shot", "graph_batch"],
        help="Mode; optional if set in YAML",
    )
    p.add_argument("--dataset", type=str, default="Cora", help="Dataset name")
    p.add_argument("--data_root", type=str, default="datasets/mdgpt", help="Data root (baseline subfolder layout)")
    p.add_argument("--downstream_root", type=str, default="downstream_data/mdgpt", help="Downstream output root (baseline subfolder layout)")
    p.add_argument("--output_dir", type=str, default=None, help="Output dir; default downstream_root/dataset")
    p.add_argument("--data_path", type=str, default=None, help="Path to .pt graph; overrides default if set")
    p.add_argument("--k_shot", type=int, default=5, choices=[1, 5], help="k-shot")
    p.add_argument("--n_splits", type=int, default=100, help="Number of splits")
    p.add_argument("--n_way", type=int, default=0, help="n-way; 0 means all classes")
    p.add_argument("--test_reserve", type=int, default=1000, help="Reserve this many tail nodes for test")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--max_one_hop", type=int, default=10, help="Max 1-hop neighbors (graph_batch only)")
    p.add_argument("--max_two_hop", type=int, default=4, help="Max 2-hop neighbors (graph_batch only)")
    p.add_argument("--n_jobs", type=int, default=4, help="Parallel workers (graph_batch only)")
    return parse_args_with_config(p, script_file=Path(__file__))


def main():
    args = _parse()
    if args.mode is None:
        raise SystemExit("Set mode: few_shot or graph_batch (first CLI arg or YAML `mode`)")
    cfg = DownstreamGeneratorConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        downstream_root=args.downstream_root,
        output_dir=args.output_dir,
        data_path=args.data_path,
        k_shot=args.k_shot,
        n_splits=args.n_splits,
        n_way=args.n_way,
        test_reserve=args.test_reserve,
        seed=args.seed,
        max_one_hop=args.max_one_hop,
        max_two_hop=args.max_two_hop,
        n_jobs=args.n_jobs,
    )

    if args.mode == "few_shot":
        generate_few_shot_splits(
            config=cfg,
            k_shot=args.k_shot,
            n_splits=args.n_splits,
        )
    else:
        generate_graph_batch_splits(
            config=cfg,
            k_shot=args.k_shot,
            n_splits=args.n_splits,
        )


if __name__ == "__main__":
    main()
