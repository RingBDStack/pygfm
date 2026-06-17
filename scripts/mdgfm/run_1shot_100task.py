#!/usr/bin/env python
"""
Batch MDGFM 1-shot 100-task finetune (node + graph), same tier as MDGPT/SAMGPT.
Split layout like downstream_data/mdgpt unless --downstream_root overrides.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from pygfm.public.repo_paths import driver_script_repo_root

ROOT = driver_script_repo_root(__file__)
DATASETS = ["Cora", "Citeseer", "Pubmed", "Photo", "Computers"]


def main():
    p = argparse.ArgumentParser(description="Batch MDGFM 1-shot 100-task finetune")
    p.add_argument("--datasets", type=str, default=None, help="Comma-separated datasets; default all")
    p.add_argument("--task_num", type=int, default=100)
    p.add_argument("--ckpt_dir", type=str, default="ckpts/mdgfm")
    p.add_argument("--downstream_root", type=str, default="../../downstream_data/mdgfm")
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--node_only", action="store_true")
    p.add_argument("--graph_only", action="store_true")
    p.add_argument("--beta", type=float, default=1.0)
    args = p.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",")] if args.datasets else DATASETS
    run_node = not args.graph_only
    run_graph = not args.node_only
    swanlab_flag = ["--no_swanlab"] if args.no_swanlab else []

    for ds in datasets:
        ckpt = str(ROOT / args.ckpt_dir / ds.lower() / f"preprompt_{ds.lower()}.pth")
        if not Path(ckpt).exists():
            print(f"[SKIP] {ds}: missing ckpt {ckpt}")
            continue

        if run_node:
            node_splits = ROOT / args.downstream_root / ds / "1shot" / "splits.pt"
            if not node_splits.exists():
                print(f"[SKIP] {ds} node NC: missing {node_splits}")
            else:
                print(f"\n{'='*60}\n>> MDGFM {ds} node NC 1-shot {args.task_num}-task\n{'='*60}")
                cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "mdgfm" / "finetune.py"),
                    "--dataset", ds, "--k_shot", "1", "--ckpt", ckpt,
                    "--task_num", str(args.task_num), "--downstream_root", args.downstream_root,
                    "--beta", str(args.beta),
                ] + swanlab_flag
                subprocess.run(cmd, cwd=str(ROOT))

        if run_graph:
            graph_splits = ROOT / args.downstream_root / ds / "1shot_graph_batch" / "splits.pt"
            if not graph_splits.exists():
                print(f"[SKIP] {ds} graph: missing {graph_splits}")
            else:
                print(f"\n{'='*60}\n>> MDGFM {ds} graph 1-shot {args.task_num}-task\n{'='*60}")
                cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "mdgfm" / "finetune_graph.py"),
                    "--dataset", ds, "--k_shot", "1", "--ckpt", ckpt,
                    "--task_num", str(args.task_num), "--downstream_root", args.downstream_root,
                    "--beta", str(args.beta),
                ] + swanlab_flag
                subprocess.run(cmd, cwd=str(ROOT))

    print("\n" + "=" * 60)
    print("All done")
    print("=" * 60)


if __name__ == "__main__":
    main()
