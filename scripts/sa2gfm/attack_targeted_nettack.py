#!/usr/bin/env python3
"""Targeted Nettack pipeline: (1) surrogate -> (2) reports -> (3) assemble (replaces run_attack_targeted_nettack.sh)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _setup_repo import sa2gfm_baseline_models_root, setup_repo

setup_repo()

from pygfm.public.cli.yaml_config import parse_args_with_config

AG = sa2gfm_baseline_models_root() / "attack_data_gen" / "pipeline"


def main():
    p = argparse.ArgumentParser(
        description="SA2GFM targeted Nettack chain (01→02→03). "
        "Env equivalent: SKIP_SURROGATE_TRAIN=1 skips step (1)."
    )
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--p", type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument(
        "--skip-surrogate-train",
        action="store_true",
        help="Skip step (1); reuse checkpoints/gcn_{dataset}.pth",
    )
    p.add_argument("--device", type=str, default="cuda", help="Device for surrogate training in step (1)")
    args = parse_args_with_config(p, script_file=Path(__file__))

    skip = args.skip_surrogate_train or os.environ.get("SKIP_SURROGATE_TRAIN", "").strip() in (
        "1",
        "true",
        "yes",
    )
    py = sys.executable
    if not skip:
        subprocess.check_call(
            [py, str(AG / "01_train_gcn_surrogate.py"), "--datasets", args.dataset, "--device", args.device]
        )
    else:
        ckpt_dir = AG.parents[1] / "checkpoints"
        print(f"SKIP surrogate train -> expect {ckpt_dir / f'gcn_{args.dataset}.pth'}")

    subprocess.check_call([py, str(AG / "02_nettack_reports.py"), "--dataset", args.dataset, "--p", str(args.p)])
    subprocess.check_call(
        [py, str(AG / "03_assemble_final.py"), "--dataset", args.dataset, "--p-values", str(args.p)]
    )
    print(f"Final graphs: {AG.parents[1]}/outputs/attack_post/{args.dataset}_p{args.p}_final/")


if __name__ == "__main__":
    main()
