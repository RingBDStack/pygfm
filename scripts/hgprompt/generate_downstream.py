#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate HGPrompt downstream few-shot splits (node-level; same format as GraphPrompt/MDGPT).

Usage:
  python scripts/hgprompt/generate_downstream.py few_shot --dataset Cora --k_shot 5
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pygfm.private.utlis.downstream_data_gen import (
    generate_few_shot_splits,
    DownstreamGeneratorConfig,
)


def _parse():
    p = argparse.ArgumentParser(description="Generate HGPrompt downstream few-shot data")
    p.add_argument("mode", choices=["few_shot"], help="Only few_shot (node-level) is supported")
    p.add_argument("--dataset", type=str, default="Cora")
    p.add_argument("--data_root", type=str, default="datasets/hgprompt")
    p.add_argument("--downstream_root", type=str, default="../../downstream_data/hgprompt")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--k_shot", type=int, default=5, choices=[1, 5])
    p.add_argument("--n_splits", type=int, default=100)
    p.add_argument("--n_way", type=int, default=0)
    p.add_argument("--test_reserve", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
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
    )
    generate_few_shot_splits(
        config=cfg,
        k_shot=args.k_shot,
        n_splits=args.n_splits,
    )


if __name__ == "__main__":
    main()
