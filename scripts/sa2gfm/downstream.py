#!/usr/bin/env python3
"""SA²GFM MoE downstream finetuning (legacy run_downstream.sh). Run from the repository root."""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _setup_repo import setup_repo

setup_repo()

if __name__ == "__main__":
    from pygfm.baseline_models.sa2gfm.downstream.pipeline.train_downstream import main

    main()
