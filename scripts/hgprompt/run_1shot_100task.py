#!/usr/bin/env python
"""
Batch HGPrompt 1-shot 100-task node classification.

No graph-level HGPrompt DownPrompt yet; only finetune.py.

Usage:
    python scripts/hgprompt/run_1shot_100task.py
    python scripts/hgprompt/run_1shot_100task.py --datasets Cora,Pubmed --no_swanlab
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
    p = argparse.ArgumentParser(description="HGPrompt batch 1-shot 100-task (node)")
    p.add_argument("--datasets", type=str, default=None)
    p.add_argument("--task_num", type=int, default=100)
    p.add_argument("--ckpt_dir", type=str, default="ckpts/hgprompt")
    p.add_argument("--downstream_root", type=str, default="../../downstream_data/hgprompt")
    p.add_argument("--no_swanlab", action="store_true")
    args = p.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",")] if args.datasets else DATASETS
    swanlab_flag = ["--no_swanlab"] if args.no_swanlab else []

    for ds in datasets:
        ckpt = str(ROOT / args.ckpt_dir / ds.lower() / f"preprompt_{ds.lower()}.pth")
        if not Path(ckpt).exists():
            print(f"[SKIP] {ds}: missing ckpt {ckpt}")
            continue

        node_splits = ROOT / args.downstream_root / ds / "1shot" / "splits.pt"
        if not node_splits.exists():
            print(f"[SKIP] {ds}: missing {node_splits}")
            continue

        print(f"\n{'='*60}\n>> {ds} node NC 1-shot {args.task_num}-task\n{'='*60}")
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "hgprompt" / "finetune.py"),
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
