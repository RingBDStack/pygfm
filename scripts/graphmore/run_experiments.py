#!/usr/bin/env python
"""
GraphMoRE batch experiments: run LP pretrain then NC finetune; multiple datasets x backbones.

Settings aligned with paper Table 2 (LP) and Table 3 (NC).

Examples:
  python scripts/graphmore/run_experiments.py --datasets Cora,Citeseer
  python scripts/graphmore/run_experiments.py --datasets Cora --backbones gcn,gat,sage --task NC
  python scripts/graphmore/run_experiments.py --task LP --datasets Cora,Citeseer,airport,Pubmed,photo
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from pygfm.public.repo_paths import driver_script_repo_root

ROOT = driver_script_repo_root(__file__)


def _parse():
    p = argparse.ArgumentParser(description="GraphMoRE batch experiments")
    p.add_argument(
        "--datasets",
        type=str,
        default="Cora,Citeseer",
        help="comma-separated dataset names",
    )
    p.add_argument(
        "--backbones",
        type=str,
        default="gcn",
        help="comma-separated backbone names for NC (gcn, gat, sage)",
    )
    p.add_argument(
        "--task",
        type=str,
        default="both",
        choices=["LP", "NC", "both"],
        help="which task(s) to run",
    )
    p.add_argument("--exp_iters", type=int, default=10)
    p.add_argument(
        "--data_root",
        type=str,
        default="datasets/graphmore",
        help="Same as load_all_datasets: flat Cora.pt or TU subdirs",
    )
    p.add_argument("--embed_features", type=int, default=32)
    p.add_argument("--no_swanlab", action="store_true")
    return p.parse_args()


def run_cmd(cmd: list[str], desc: str):
    print(f"\n{'='*60}")
    print(f"Running: {desc}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"WARNING: {desc} exited with code {result.returncode}")
    return result.returncode


def main():
    args = _parse()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    python = sys.executable

    swanlab_flag = ["--no_swanlab"] if args.no_swanlab else []

    for ds in datasets:
        if args.task in ("LP", "both"):
            cmd = [
                python,
                "scripts/graphmore/pretrain.py",
                "--dataset", ds,
                "--data_root", args.data_root,
                "--embed_features", str(args.embed_features),
                *swanlab_flag,
            ]
            run_cmd(cmd, f"LP pretrain on {ds}")

        if args.task in ("NC", "both"):
            for bb in backbones:
                cmd = [
                    python,
                    "scripts/graphmore/finetune.py",
                    "--dataset", ds,
                    "--data_root", args.data_root,
                    "--backbone", bb,
                    "--embed_features", str(args.embed_features),
                    "--exp_iters", str(args.exp_iters),
                    *swanlab_flag,
                ]
                run_cmd(cmd, f"NC finetune on {ds} with {bb}")

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
