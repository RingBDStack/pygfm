#!/usr/bin/env python
"""Generate GraphKeeper downstream few-shot splits (default downstream_data/graphkeeper)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pygfm.public.repo_paths import driver_script_repo_root

ROOT = driver_script_repo_root(__file__)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pygfm.private.utlis.downstream_data_gen import (
    DownstreamGeneratorConfig,
    generate_few_shot_splits,
    generate_graph_batch_splits,
)


def _parse():
    p = argparse.ArgumentParser(description="Generate GraphKeeper downstream splits")
    p.add_argument("mode", choices=["few_shot", "graph_batch"])
    p.add_argument("--dataset", type=str, default="Cora")
    p.add_argument("--data_root", type=str, default="datasets/graphkeeper")
    p.add_argument("--downstream_root", type=str, default="../../downstream_data/graphkeeper")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--k_shot", type=int, default=5, choices=[1, 5])
    p.add_argument("--n_splits", type=int, default=100)
    p.add_argument("--n_way", type=int, default=0)
    p.add_argument("--test_reserve", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_one_hop", type=int, default=10)
    p.add_argument("--max_two_hop", type=int, default=4)
    p.add_argument("--n_jobs", type=int, default=4)
    return p.parse_args()


def main():
    args = _parse()
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
        generate_few_shot_splits(cfg, k_shot=args.k_shot, n_splits=args.n_splits)
    else:
        generate_graph_batch_splits(cfg, k_shot=args.k_shot, n_splits=args.n_splits)


if __name__ == "__main__":
    main()
