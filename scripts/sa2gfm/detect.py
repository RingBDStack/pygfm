#!/usr/bin/env python3
"""Community detection(legacy run_detect.sh)。Usage: python scripts/sa2gfm/detect.py --dataset cora [louvain|greedy_modularity|label_propagation]"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _setup_repo import setup_repo

REPO = setup_repo()


def _main():
    path = REPO / "pygfm/baseline_models/sa2gfm/community_detection/pipeline/01_detect_communities.py"
    spec = importlib.util.spec_from_file_location("sa2gfm_detect_communities", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    _main()
