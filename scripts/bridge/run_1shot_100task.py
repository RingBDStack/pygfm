#!/usr/bin/env python
"""
Batch BRIDGE 1-shot 100-task node classification (optional graph batch; prepare ckpt and data).

  python scripts/bridge/run_1shot_100task.py
  python scripts/bridge/run_1shot_100task.py --datasets Cora,Pubmed --no_swanlab
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["Cora", "Citeseer", "Pubmed", "Photo", "Computers"]


def main():
    p = argparse.ArgumentParser(description="BRIDGE batch 1-shot finetuning")
    p.add_argument("--datasets", type=str, default=None)
    p.add_argument("--task_num", type=int, default=100)
    p.add_argument("--ckpt_dir", type=str, default="ckpts/bridge")
    p.add_argument("--downstream_root", type=str, default="downstream_data/bridge")
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--graph_only", action="store_true")
    p.add_argument("--node_only", action="store_true")
    args = p.parse_args()

    dss = [x.strip() for x in args.datasets.split(",")] if args.datasets else DATASETS
    sw = ["--no_swanlab"] if args.no_swanlab else []
    run_node = not args.graph_only
    run_graph = not args.node_only

    for ds in dss:
        ckpt = ROOT / args.ckpt_dir / ds.lower() / f"preprompt_{ds.lower()}.pth"
        if not ckpt.exists():
            print(f"[SKIP] {ds}: missing {ckpt}")
            continue

        if run_node:
            sp = ROOT / args.downstream_root / ds / "1shot" / "splits.pt"
            if not sp.exists():
                print(f"[SKIP] {ds} node: {sp} missing")
            else:
                print(f"\n>> {ds} node 1-shot x{args.task_num}")
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "bridge" / "finetune.py"),
                        "--dataset",
                        ds,
                        "--k_shot",
                        "1",
                        "--ckpt",
                        str(ckpt),
                        "--task_num",
                        str(args.task_num),
                        "--downstream_root",
                        args.downstream_root,
                    ]
                    + sw,
                    cwd=str(ROOT),
                )

        if run_graph:
            gp = ROOT / args.downstream_root / ds / "1shot_graph_batch" / "splits.pt"
            if not gp.exists():
                print(f"[SKIP] {ds} graph: {gp} missing (run generate_downstream graph_batch)")
            else:
                print(f"\n>> {ds} graph 1-shot x{args.task_num}")
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "bridge" / "finetune_graph.py"),
                        "--dataset",
                        ds,
                        "--k_shot",
                        "1",
                        "--ckpt",
                        str(ckpt),
                        "--task_num",
                        str(args.task_num),
                        "--downstream_root",
                        args.downstream_root,
                    ]
                    + sw,
                    cwd=str(ROOT),
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
