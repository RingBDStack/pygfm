#!/usr/bin/env python3
"""SA²GFM single-graph pretraining (legacy run_pretrain.sh). Run from the repository root."""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _setup_repo import setup_repo

setup_repo()

if __name__ == "__main__":
    from pygfm.baseline_models.sa2gfm.pretrain.pipeline.train_single import train

    train()
