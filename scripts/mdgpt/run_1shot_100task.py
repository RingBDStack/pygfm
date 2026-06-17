#!/usr/bin/env python
"""
Batch 1-shot 100-task finetune (node classification + graph classification).

Usage:
    python scripts/mdgpt/run_1shot_100task.py
    python scripts/mdgpt/run_1shot_100task.py -c scripts/mdgpt/run_1shot_100task.yaml
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from pygfm.public.repo_paths import driver_script_repo_root

ROOT = driver_script_repo_root(__file__)
_SCRIPTS_MDGPT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(_SCRIPTS_MDGPT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_MDGPT))

from config_utils import parse_args_with_config

DATASETS = ["Cora", "Citeseer", "Pubmed", "Photo", "Computers"]


def _resolve_path(p: str | None) -> str | None:
    if not p:
        return None
    path = Path(p)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return str(path)


def main():
    p = argparse.ArgumentParser(description="Batch 1-shot 100-task finetune")
    p.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated datasets; default all",
    )
    p.add_argument("--task_num", type=int, default=100)
    p.add_argument("--ckpt_dir", type=str, default="ckpts/mdgpt", help="Pretrain ckpt root (baseline subfolder layout)")
    p.add_argument("--downstream_root", type=str, default="downstream_data/mdgpt", help="Downstream splits root (baseline subfolder layout)")
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--node_only", action="store_true", help="Node classification only")
    p.add_argument("--graph_only", action="store_true", help="Graph classification only")
    p.add_argument(
        "--finetune_config",
        type=str,
        default=None,
        help="YAML for finetune.py (subprocess can still override via CLI)",
    )
    p.add_argument(
        "--finetune_graph_config",
        type=str,
        default=None,
        help="YAML for finetune_graph.py",
    )
    args = parse_args_with_config(p, script_file=Path(__file__))

    datasets = [x.strip() for x in args.datasets.split(",")] if args.datasets else DATASETS
    run_node = not args.graph_only
    run_graph = not args.node_only

    swanlab_flag = ["--no_swanlab"] if args.no_swanlab else []
    ft_cfg = _resolve_path(args.finetune_config)
    ftg_cfg = _resolve_path(args.finetune_graph_config)

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
                    str(ROOT / "scripts" / "mdgpt" / "finetune.py"),
                    "--dataset", ds,
                    "--k_shot", "1",
                    "--ckpt", ckpt,
                    "--task_num", str(args.task_num),
                    "--downstream_root", args.downstream_root,
                ] + swanlab_flag
                if ft_cfg:
                    cmd += ["--config", ft_cfg]
                subprocess.run(cmd, cwd=str(ROOT))

        if run_graph:
            graph_splits = ROOT / args.downstream_root / ds / "1shot_graph_batch" / "splits.pt"
            if not graph_splits.exists():
                print(f"[SKIP] {ds} graph: missing {graph_splits}")
            else:
                print(f"\n{'='*60}\n>> {ds} graph 1-shot {args.task_num}-task\n{'='*60}")
                cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "mdgpt" / "finetune_graph.py"),
                    "--dataset", ds,
                    "--k_shot", "1",
                    "--ckpt", ckpt,
                    "--task_num", str(args.task_num),
                    "--downstream_root", args.downstream_root,
                ] + swanlab_flag
                if ftg_cfg:
                    cmd += ["--config", ftg_cfg]
                subprocess.run(cmd, cwd=str(ROOT))

    print("\n" + "=" * 60)
    print("All done")
    print("=" * 60)


if __name__ == "__main__":
    main()
