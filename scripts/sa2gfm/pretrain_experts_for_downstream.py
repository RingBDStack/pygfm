#!/usr/bin/env python3
"""
For each downstream target graph, call ``pretrain.py`` to train MoE experts on the other graphs (pure Python, no shell).

From repo root::

    python scripts/sa2gfm/pretrain_experts_for_downstream.py --target cora \\
        -- -c scripts/sa2gfm/configs/pretrain_smoke.yaml

Arguments after ``--`` are forwarded to each ``pretrain.py`` run (except ``--dataset``, set per call).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _setup_repo import setup_repo

setup_repo()

from pygfm.baseline_models.sa2gfm.downstream.lib.config import get_pretrain_datasets


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sequential pretrain for MoE experts required by downstream --target dataset.",
    )
    p.add_argument("--target", type=str, required=True, help="Downstream dataset, e.g. cora / Cora")
    p.add_argument("--dry_run", action="store_true", help="Print commands only")
    args, rest = p.parse_known_args()
    if rest[:1] == ["--"]:
        rest = rest[1:]
    experts = get_pretrain_datasets(args.target)
    if not experts:
        raise SystemExit(f"No expert list for target={args.target!r}")
    pretrain_py = _SCRIPT_DIR / "pretrain.py"
    print(f"MoE experts for downstream target {args.target!r}: {experts}")
    for name in experts:
        cmd = [sys.executable, str(pretrain_py), "--dataset", name, *rest]
        print(">>", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=_REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
