#!/usr/bin/env python
"""
Batch GraphPrompt 1-shot 100-task (node + graph classification).

Usage:
    python scripts/graphprompt/run_1shot_100task.py
    python scripts/graphprompt/run_1shot_100task.py --datasets Cora,Pubmed --no_swanlab
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["Cora", "Citeseer", "Pubmed", "Photo", "Computers"]


def main():
    p = argparse.ArgumentParser(description="GraphPrompt batch 1-shot 100-task")
    p.add_argument("--datasets", type=str, default=None)
    p.add_argument("--task_num", type=int, default=100)
    p.add_argument("--ckpt_dir", type=str, default="ckpts/graphprompt")
    p.add_argument("--downstream_root", type=str, default="../../downstream_data/graphprompt")
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--node_only", action="store_true")
    p.add_argument("--graph_only", action="store_true")
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
                print(f"\n{'='*60}\n>> {ds} node NC 1-shot {args.task_num}-task\n{'='*60}")
                cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "graphprompt" / "finetune.py"),
                    "--dataset", ds,
                    "--k_shot", "1",
                    "--ckpt", ckpt,
                    "--task_num", str(args.task_num),
                    "--downstream_root", args.downstream_root,
                ] + swanlab_flag
                subprocess.run(cmd, cwd=str(ROOT))

        if run_graph:
            graph_splits = ROOT / args.downstream_root / ds / "1shot_graph_batch" / "splits.pt"
            if not graph_splits.exists():
                print(f"[SKIP] {ds} graph: missing {graph_splits}")
            else:
                print(f"\n{'='*60}\n>> {ds} graph 1-shot {args.task_num}-task\n{'='*60}")
                cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "graphprompt" / "finetune_graph.py"),
                    "--dataset", ds,
                    "--k_shot", "1",
                    "--ckpt", ckpt,
                    "--task_num", str(args.task_num),
                    "--downstream_root", args.downstream_root,
                ] + swanlab_flag
                subprocess.run(cmd, cwd=str(ROOT))

    print("\n" + "=" * 60)
    print("All done")
    print("=" * 60)


if __name__ == "__main__":
    main()
