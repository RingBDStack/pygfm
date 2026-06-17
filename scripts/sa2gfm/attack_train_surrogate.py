#!/usr/bin/env python3
"""Attack step 1: train GCN surrogate for Nettack (replaces run_attack_train_surrogate.sh)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _setup_repo import sa2gfm_baseline_models_root, setup_repo

setup_repo()


def _main():
    path = (
        sa2gfm_baseline_models_root()
        / "attack_data_gen"
        / "pipeline"
        / "01_train_gcn_surrogate.py"
    )
    spec = importlib.util.spec_from_file_location("sa2gfm_01_surrogate", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    _main()
