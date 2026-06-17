#!/usr/bin/env python3
"""
GraphGPT model weights: download Vicuna-7B and the graph encoder from HuggingFace for Stage1/Stage2.

Usage (from GraphGPT root):
  pip install huggingface_hub tqdm
  python scripts/prepare_models.py

Downloads to:
  - ckpts/graphgpt/checkpoints/vicuna-7b-v1.5/     (Vicuna 7B v1.5, ~14GB, --model_name_or_path)
  - ckpts/graphgpt/checkpoints/clip_gt_arxiv/      (Arxiv-PubMed-GraphCLIP-GT, --graph_tower / pretra_gnn)

China-friendly default: HF_ENDPOINT=https://hf-mirror.com; expect ~15-20GB disk.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from pygfm.public.repo_paths import driver_script_repo_root

ROOT = driver_script_repo_root(__file__) / "scripts"
os.chdir(ROOT)


# Prefer China mirror in code so env overrides do not silently break downloads
HF_MIRROR = "https://hf-mirror.com"

# Completeness: Vicuna needs two .bin shards totaling ~12.5GB; encoder needs pkl + config
VICUNA_BIN_NAMES = ("pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin")
VICUNA_MIN_BYTES = 12 * (1 << 30)  # 12GB
GRAPH_ENCODER_PKL = "clip_gt_arxiv_pub.pkl"
GRAPH_ENCODER_MIN_BYTES = 1 << 20   # 1MB


def _vicuna_complete(path: Path) -> bool:
    if not path.is_dir():
        return False
    total = 0
    for name in VICUNA_BIN_NAMES:
        f = path / name
        if not f.is_file():
            return False
        total += f.stat().st_size
    return total >= VICUNA_MIN_BYTES


def _graph_encoder_complete(path: Path) -> bool:
    if not path.is_dir():
        return False
    pkl = path / GRAPH_ENCODER_PKL
    cfg = path / "config.json"
    return pkl.is_file() and pkl.stat().st_size >= GRAPH_ENCODER_MIN_BYTES and cfg.is_file()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download GraphGPT weights (Vicuna + graph encoder)")
    parser.add_argument("--check-only", action="store_true", help="Only verify local weights, no download")
    args = parser.parse_args()

    ckpt_dir = ROOT / "ckpts" / "graphgpt" / "checkpoints"
    vicuna_path = ckpt_dir / "vicuna-7b-v1.5"
    graph_enc_path = ckpt_dir / "clip_gt_arxiv"

    if args.check_only:
        print("=== Weight integrity check ===\n")
        v_ok = _vicuna_complete(vicuna_path)
        print("Vicuna-7B-v1.5:  ", "OK" if v_ok else "INCOMPLETE (missing .bin or total < ~12GB)")
        print("  path:", vicuna_path)
        if vicuna_path.is_dir():
            bin_total = sum((vicuna_path / n).stat().st_size for n in VICUNA_BIN_NAMES if (vicuna_path / n).is_file())
            print("  .bin total:", f"{bin_total / (1<<30):.2f} GB")
        g_ok = _graph_encoder_complete(graph_enc_path)
        print("\nGraph encoder (clip_gt_arxiv):", "OK" if g_ok else "INCOMPLETE (missing clip_gt_arxiv_pub.pkl or too small)")
        print("  path:", graph_enc_path)
        if graph_enc_path.is_dir():
            pkl = graph_enc_path / GRAPH_ENCODER_PKL
            print("  clip_gt_arxiv_pub.pkl:", f"{pkl.stat().st_size / (1<<20):.2f} MB" if pkl.is_file() else "missing")
        if not v_ok or not g_ok:
            print("\nRemove incomplete dirs before re-download:")
            if not v_ok:
                print("  rm -rf", vicuna_path)
            if not g_ok:
                print("  rm -rf", graph_enc_path)
        sys.exit(0 if (v_ok and g_ok) else 1)

    os.environ["HF_ENDPOINT"] = HF_MIRROR
    print("Using mirror:", HF_MIRROR, "\n")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Install first: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi(endpoint=HF_MIRROR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1) Vicuna 7B v1.5 (~14GB)
    if _vicuna_complete(vicuna_path):
        print("[skip] Vicuna complete:", vicuna_path)
    else:
        if vicuna_path.exists():
            print("[incomplete] Re-downloading Vicuna-7B-v1.5 (~14GB) ...")
        else:
            print("Downloading Vicuna-7B-v1.5 (~14GB) ...")
        api.snapshot_download(
            repo_id="lmsys/vicuna-7b-v1.5",
            local_dir=str(vicuna_path),
            force_download=False,
            resume_download=True,
        )
        print("  ->", vicuna_path)

    # 2) Graph encoder Arxiv-PubMed-GraphCLIP-GT
    if _graph_encoder_complete(graph_enc_path):
        print("[skip] Graph encoder complete:", graph_enc_path)
    else:
        if graph_enc_path.exists():
            print("[incomplete] Re-downloading Arxiv-PubMed-GraphCLIP-GT (encoder) ...")
        else:
            print("Downloading Arxiv-PubMed-GraphCLIP-GT (encoder) ...")
        api.snapshot_download(
            repo_id="Jiabin99/Arxiv-PubMed-GraphCLIP-GT",
            local_dir=str(graph_enc_path),
            force_download=False,
            resume_download=True,
        )
        print("  ->", graph_enc_path)

    print("\nDone. Training paths:")
    print("  model_path   =", vicuna_path)
    print("  pretra_gnn   =", graph_enc_path)


if __name__ == "__main__":
    main()
