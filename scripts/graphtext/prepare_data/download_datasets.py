#!/usr/bin/env python3
"""
Download the six ICL datasets into the project data/ directory (PyG path).
No DGL needed; only torch + torch_geometric.

Usage:
  cd /path/to/GraphText-main
  python download_datasets.py
"""
import sys
from pathlib import Path
from pygfm.public.repo_paths import driver_script_repo_root

# Resolve target dataset directory inside gfm-toolbox-main.
PROJECT_ROOT = driver_script_repo_root(__file__, parents_up_to_repo=3)  # .../gfm-toolbox-main
DATA_DIR = PROJECT_ROOT / "datasets" / "graphtext" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Allow `from download_datasets_pyg import main` when called from any cwd.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

def main():
    # Reuse the dedicated PyG download script to keep behavior consistent.
    from download_datasets_pyg import main as pyg_main
    pyg_main()

if __name__ == "__main__":
    main()
