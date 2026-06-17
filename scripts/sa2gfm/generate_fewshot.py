#!/usr/bin/env python3
"""Generate few-shot splits(legacy run_generate_fewshot.sh)。Usage: python scripts/sa2gfm/generate_fewshot.py --dataset cora --k-shot 1"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _setup_repo import sa2gfm_baseline_models_root, setup_repo

setup_repo()


def _main():
    path = sa2gfm_baseline_models_root() / "few_shot_gen" / "pipeline" / "01_generate_splits.py"
    spec = importlib.util.spec_from_file_location("sa2gfm_generate_splits", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    _main()
