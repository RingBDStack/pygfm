#!/usr/bin/env python3
"""
Fill in / verify Stage1 finetuning assets (defaults to hf-mirror in China).

Usage (from GraphGPT project root):
  pip install huggingface_hub tqdm
  python scripts/ensure_stage1_files.py

This will:
  - Download graph_data_all.pt and graph-matching data from HuggingFace if missing/too small
  - Copy graph_data_all.pt -> all_graph_data.pt
  - Download graph encoder weights to ckpts/graphgpt/checkpoints/clip_gt_arxiv/
  - arxiv_ti_ab.json: no full public bundle; write {} placeholder if missing
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from pygfm.public.repo_paths import driver_script_repo_root

ROOT = driver_script_repo_root(__file__) / "scripts"
os.chdir(ROOT)

if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def _need(path: Path, min_bytes: int) -> bool:
    if not path.is_file():
        return True
    return path.stat().st_size < min_bytes


def main() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install: pip install huggingface_hub tqdm")
        sys.exit(1)

    graph_dir = ROOT / "graph_data"
    graph_dir.mkdir(parents=True, exist_ok=True)
    gd = graph_dir / "graph_data_all.pt"
    if _need(gd, 800_000_000):
        print("Downloading graph_data_all.pt ...")
        hf_hub_download(
            repo_id="Jiabin99/All_pyg_graph_data",
            filename="graph_data_all.pt",
            repo_type="dataset",
            local_dir=str(graph_dir),
        )
    ag = graph_dir / "all_graph_data.pt"
    if not ag.exists() or ag.resolve() != gd.resolve():
        shutil.copy2(gd, ag)
        print(f"Copied to {ag}")

    stage1 = ROOT / "data" / "stage_1"
    stage1.mkdir(parents=True, exist_ok=True)
    gm = stage1 / "graph_matching.json"
    if _need(gm, 900_000_000):
        print("Downloading graph-matching instruction data ...")
        raw = stage1 / "train_instruct_graphmatch.json"
        hf_hub_download(
            repo_id="Jiabin99/graph-matching",
            filename="train_instruct_graphmatch.json",
            repo_type="dataset",
            local_dir=str(stage1),
        )
        shutil.copy2(raw, gm)
        print(f"Copied to {gm}")

    enc = ROOT / "ckpts" / "graphgpt" / "checkpoints" / "clip_gt_arxiv"
    for name, repo, fn in (
        ("clip_gt_arxiv_pub.pkl", "Jiabin99/Arxiv-PubMed-GraphCLIP-GT", "clip_gt_arxiv_pub.pkl"),
        ("config.json", "Jiabin99/Arxiv-PubMed-GraphCLIP-GT", "config.json"),
    ):
        p = enc / name
        if not p.is_file() or (name.endswith(".pkl") and p.stat().st_size < 1_000_000):
            enc.mkdir(parents=True, exist_ok=True)
            print(f"Downloading graph encoder {name} ...")
            hf_hub_download(
                repo_id=repo,
                filename=fn,
                local_dir=str(enc),
            )

    # NOTE: We intentionally do NOT create a repo-root symlink (e.g. ./clip_gt_arxiv).
    # All code and scripts should reference ckpts/graphgpt/checkpoints directly.

    ti_ab = ROOT / "datasets" / "graphgpt" / "arxiv_ti_ab.json"
    if not ti_ab.exists():
        ti_ab.parent.mkdir(parents=True, exist_ok=True)
        ti_ab.write_text("{}\n", encoding="utf-8")
        print(f"Wrote placeholder {ti_ab} (no full public bundle; Stage1 graph_matching often skips this)")

    readme = ROOT / "datasets" / "graphgpt" / "arxiv_ti_ab.README.txt"
    if not readme.exists():
        readme.parent.mkdir(parents=True, exist_ok=True)
        readme.write_text(
            "arxiv_ti_ab.json is not published separately on HuggingFace.\n"
            "Stage1 (graph_matching) does not read this file; empty {} is enough to train.\n"
            "If a later stage needs title/abstract, build from text-graph-grounding or paper supplements.\n",
            encoding="utf-8",
        )
        print(f"Wrote notes {readme}")

    print(
        "\nDone. Ensure Vicuna is under ckpts/graphgpt/checkpoints/vicuna-7b-v1.5/ (see scripts/graphgpt/model_download_links.md)."
    )


if __name__ == "__main__":
    main()
