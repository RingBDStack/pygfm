#!/usr/bin/env python3
"""
No DGL: HTTP-download official DGL dataset zips and extract to data/.
Works in any Python env with requests. Use this if the DGL stack fails.

Usage:
  cd /path/to/GraphText-main
  pip install requests   # if missing
  python download_datasets_standalone.py
"""
import os
import zipfile
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../gfm-toolbox-main
DATA_DIR = str(PROJECT_ROOT / "datasets" / "graphtext" / "data")
BASE_URL = os.environ.get("DGL_REPO", "https://data.dgl.ai/").rstrip("/") + "/"

# Official DGL citation zips (same sources as CoraGraphDataset, etc.)
DATASETS = [
    ("cora_v2", "dataset/cora_v2.zip"),   # Cora uses cora_v2
    ("citeseer", "dataset/citeseer.zip"),
    ("pubmed", "dataset/pubmed.zip"),
]

def download_and_extract(name, path_suffix):
    out_dir = os.path.join(DATA_DIR, name.replace(".zip", ""))
    zip_path = os.path.join(DATA_DIR, os.path.basename(path_suffix))
    url = BASE_URL + path_suffix
    try:
        import requests
    except ImportError:
        print("Install first: pip install requests")
        sys.exit(1)
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        print(f"  Already present, skip: {name}")
        return
    print(f"  Downloading {name} ...")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
    print(f"  Extracting {name} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    try:
        os.remove(zip_path)
    except OSError:
        pass
    print(f"  Done: {name}")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Data directory: {DATA_DIR}\n")
    for name, path_suffix in DATASETS:
        try:
            download_and_extract(name, path_suffix)
        except Exception as e:
            print(f"  Failed {name}: {e}")
    print(f"\nDone. Cora / Citeseer / PubMed saved under: {DATA_DIR}.")
    print("Texas / Wisconsin / Cornell need a DGL env, or fetch when your main env loads DGL.")

if __name__ == "__main__":
    main()
