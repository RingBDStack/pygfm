#!/usr/bin/env python3
"""
Download the six ICL datasets into project data/ using PyTorch Geometric (PyG).
No DGL; requires: pip install torch torch_geometric

Usage:
  cd /path/to/GraphText-main
  python download_datasets_pyg.py
"""
import os
import sys
from pathlib import Path
from pygfm.public.repo_paths import driver_script_repo_root

PROJECT_ROOT = driver_script_repo_root(__file__, parents_up_to_repo=3)  # .../gfm-toolbox-main
DATA_DIR = PROJECT_ROOT / "datasets" / "graphtext" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main():
    root = os.path.abspath(str(DATA_DIR))
    print(f"Data directory: {root}\n")

    # Planetoid: Cora, CiteSeer, PubMed
    from torch_geometric.datasets import Planetoid
    for name in ["Cora", "CiteSeer", "PubMed"]:
        print(f"  Downloading {name.lower()} (Planetoid) ...")
        try:
            ds = Planetoid(root=root, name=name)
            d = ds[0]
            print(f"  Done: {name.lower()} (nodes={d.num_nodes}, edges={d.edge_index.size(1)})")
        except Exception as e:
            print(f"  Failed: {name.lower()} - {e}")

    # WebKB: Texas, Wisconsin, Cornell
    from torch_geometric.datasets import WebKB
    for name in ["Texas", "Wisconsin", "Cornell"]:
        print(f"  Downloading {name.lower()} (WebKB) ...")
        try:
            ds = WebKB(root=root, name=name)
            d = ds[0]
            print(f"  Done: {name.lower()} (nodes={d.num_nodes}, edges={d.edge_index.size(1)})")
        except Exception as e:
            print(f"  Failed: {name.lower()} - {e}")

    print("\nDone. Data saved under:", root)
    print("  Planetoid ->", os.path.join(root, "Planetoid"))
    print("  WebKB    ->", os.path.join(root, "WebKB"))


if __name__ == "__main__":
    main()
