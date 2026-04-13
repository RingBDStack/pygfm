#!/usr/bin/env python3
"""Metattack: surrogate -> batch (replaces run_attack_metattack.sh). Needs: pip install -e \".[sa2gfm-attack]\"."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _setup_repo import setup_repo

REPO = setup_repo()

from pygfm.public.cli.yaml_config import parse_args_with_config

AG = REPO / "pygfm" / "baseline_models" / "sa2gfm" / "attack_data_gen" / "pipeline"


def main():
    p = argparse.ArgumentParser(description="SA2GFM Metattack chain (05→06)")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda/cpu; default ATTACK_DEVICE env or cuda",
    )
    args = parse_args_with_config(p, script_file=Path(__file__))

    dev = args.device or os.environ.get("ATTACK_DEVICE", "cuda")
    py = sys.executable
    subprocess.check_call([py, str(AG / "05_metattack_surrogate.py"), "--dataset", args.dataset, "--device", dev])
    subprocess.check_call([py, str(AG / "06_metattack_batch.py"), "--dataset", args.dataset])
    print(f"Batch pickles: {AG.parents[1]}/outputs/metattack_batch/{args.dataset}/")


if __name__ == "__main__":
    main()
