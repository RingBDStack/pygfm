#!/usr/bin/env python3
"""
GraphGPT data prep: download graph + instruction assets from HuggingFace into expected local paths.

Usage (from GraphGPT root):
  pip install huggingface_hub tqdm
  python scripts/prepare_data.py

If huggingface.co is unreachable, this sets HF_ENDPOINT=https://hf-mirror.com by default.
For the official hub: unset HF_ENDPOINT first.

Creates/writes:
  - graph_data/graph_data_all.pt     (All_pyg_graph_data, --graph_data_path)
  - data/stage_1/graph_matching.json  (graph-matching, Stage1)
  - data/stage_2/arxiv_pub_node_st_cot_link_mix.json (Arxiv-PubMed-mix-NC-LP, Stage2)
  - arxiv_ti_ab.json: download if available on HF; else empty placeholder (--graph_content)

Skips download when files exist and meet minimum sizes; re-downloads truncated files.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

# Minimum sizes: treat smaller files as incomplete and re-fetch
MIN_BYTES = {
    "graph_data/graph_data_all.pt": 800_000_000,   # ~840MB
    "data/stage_1/graph_matching.json": 900_000_000,  # ~882MB
    "data/stage_2/arxiv_pub_node_st_cot_link_mix.json": 900_000_000,  # ~1GB, 900MB threshold
}


def _is_complete(rel_path: str) -> bool:
    """True if file exists and size >= expected minimum."""
    p = ROOT / rel_path
    if not p.is_file():
        return False
    min_b = MIN_BYTES.get(rel_path, 0)
    return p.stat().st_size >= min_b


def _download_with_progress(repo_id: str, filename: str, local_dir: Path, repo_type: str = "dataset") -> Path:
    """Download one file from HuggingFace with tqdm; return local path."""
    try:
        from huggingface_hub import hf_hub_url, hf_hub_download
        from tqdm import tqdm
        import urllib.request
    except ImportError as e:
        if "tqdm" in str(e):
            print("Install first: pip install tqdm")
        else:
            print("Install first: pip install huggingface_hub")
        raise SystemExit(1) from e

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    dest = local_dir / filename

    # Public datasets need no token; gated repos: set HF_TOKEN and use fallback below
    headers = {"User-Agent": "GraphGPT-prepare_data"}
    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type=repo_type)
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            with open(dest, "wb") as f:
                with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=filename) as pbar:
                    while True:
                        chunk = resp.read(64 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
    except Exception:
        # Stream failed: fall back to hub download (HF_TOKEN, etc.)
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=str(local_dir),
        )
        return Path(path)
    return dest


def main():
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("Using HuggingFace mirror: https://hf-mirror.com (override with HF_ENDPOINT)\n")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install first: pip install huggingface_hub tqdm")
        sys.exit(1)

    try:
        from tqdm import tqdm
    except ImportError:
        print("Install tqdm for progress bars: pip install tqdm")
        use_tqdm = False
    else:
        use_tqdm = True

    graph_data_dir = ROOT / "graph_data"
    stage1_dir = ROOT / "data" / "stage_1"
    stage2_dir = ROOT / "data" / "stage_2"
    for d in (graph_data_dir, stage1_dir, stage2_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1) Graph: All_pyg_graph_data -> graph_data/graph_data_all.pt
    if _is_complete("graph_data/graph_data_all.pt"):
        dest = graph_data_dir / "graph_data_all.pt"
        print("  [skip] graph_data_all.pt already complete")
    else:
        if use_tqdm:
            dest = _download_with_progress(
                "Jiabin99/All_pyg_graph_data",
                "graph_data_all.pt",
                graph_data_dir,
                repo_type="dataset",
            )
        else:
            path = hf_hub_download(
                repo_id="Jiabin99/All_pyg_graph_data",
                filename="graph_data_all.pt",
                repo_type="dataset",
                local_dir=str(graph_data_dir),
            )
            dest = Path(path).resolve()
        print(f"  -> {dest}")
    all_graph = graph_data_dir / "all_graph_data.pt"
    if not all_graph.exists() or all_graph.resolve() != dest.resolve():
        import shutil
        shutil.copy(str(dest), str(all_graph))
        print(f"  (also copied to {all_graph})")

    # 2) Stage1: graph-matching -> data/stage_1/graph_matching.json
    graph_matching = stage1_dir / "graph_matching.json"
    if _is_complete("data/stage_1/graph_matching.json"):
        print("  [skip] graph_matching.json already complete")
    else:
        if use_tqdm:
            path = _download_with_progress(
                "Jiabin99/graph-matching",
                "train_instruct_graphmatch.json",
                stage1_dir,
                repo_type="dataset",
            )
        else:
            path = hf_hub_download(
                repo_id="Jiabin99/graph-matching",
                filename="train_instruct_graphmatch.json",
                repo_type="dataset",
                local_dir=str(stage1_dir),
            )
            path = Path(path)
        if graph_matching.resolve() != Path(path).resolve():
            import shutil
            shutil.copy(str(path), str(graph_matching))
        print(f"  -> {graph_matching}")

    # 3) Stage2: Arxiv-PubMed-mix-NC-LP
    stage2_file = stage2_dir / "arxiv_pub_node_st_cot_link_mix.json"
    if _is_complete("data/stage_2/arxiv_pub_node_st_cot_link_mix.json"):
        print("  [skip] arxiv_pub_node_st_cot_link_mix.json already complete")
    else:
        if use_tqdm:
            path = _download_with_progress(
                "Jiabin99/Arxiv-PubMed-mix-NC-LP",
                "arxiv_pub_node_st_cot_link_mix.json",
                stage2_dir,
                repo_type="dataset",
            )
        else:
            path = hf_hub_download(
                repo_id="Jiabin99/Arxiv-PubMed-mix-NC-LP",
                filename="arxiv_pub_node_st_cot_link_mix.json",
                repo_type="dataset",
                local_dir=str(stage2_dir),
            )
            path = Path(path)
        print(f"  -> {path}")

    # 4) arxiv_ti_ab.json (--graph_content); not on HF as standalone -> placeholder
    arxiv_ti_ab = ROOT / "datasets" / "graphgpt" / "arxiv_ti_ab.json"
    if not arxiv_ti_ab.exists():
        arxiv_ti_ab.parent.mkdir(parents=True, exist_ok=True)
        arxiv_ti_ab.write_text("{}")
        print("  arxiv_ti_ab.json not on HF; wrote empty {}; add node title/abstract map if Stage1 errors.")
    else:
        print("  arxiv_ti_ab.json exists; left unchanged.")

    print("\nDone. Point tune_script paths at Stage1/Stage2 assets.")
    print("  graph_data_path e.g.: ./datasets/graphgpt/graph_data/graph_data_all.pt")
    print("  Stage1 data_path:     ./datasets/graphgpt/stage_1/graph_matching.json")
    print("  Stage2 data_path:     ./datasets/graphgpt/stage_2/arxiv_pub_node_st_cot_link_mix.json")


if __name__ == "__main__":
    main()
