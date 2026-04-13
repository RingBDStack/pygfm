#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert fewshot_* and fewshot_*_graph under tjc_www/model_node_rag/data into
gfm-toolbox downstream_data/rag_gfm/{dataset}/{k}shot/splits.pt and
{k}shot_graph_batch/splits.pt layouts.
"""
from __future__ import annotations

import os
import re
import torch
from pathlib import Path

SRC_DATA = Path("/root/autodl-tmp/tjc_www/model_node_rag/data")
DST_ROOT = Path("/root/autodl-tmp/gfm-toolbox-main/downstream_data/rag_gfm")

def _count_n_way(splits: list, key: str) -> int:
    seen = set()
    for s in splits:
        for x in s[key].tolist() if hasattr(s[key], "tolist") else s[key]:
            if isinstance(x, (list, tuple)):
                seen.update(int(v) for v in x)
            else:
                seen.add(int(x))
    return len(seen) if seen else 0


# fewshot_cora -> Cora, fewshot_wikics -> Wikics
def to_dataset_name(folder_name: str) -> str:
    name = folder_name.replace("fewshot_", "").replace("_graph", "").replace("_new", "")
    return name.capitalize() if len(name) <= 6 else name[0].upper() + name[1:]


def migrate_node_fewshot():
    """Node: fewshot_<name>/<k>-shot_<name>/0,1,2,.../idx.pt, labels.pt -> splits.pt"""
    for d in sorted(SRC_DATA.iterdir()):
        if not d.is_dir() or not d.name.startswith("fewshot_") or "_graph" in d.name or "_new" in d.name:
            continue
        base_name = d.name.replace("fewshot_", "")
        dataset = to_dataset_name(d.name)
        # subdirs: 1-shot_cora, 5-shot_cora, ...
        for shot_dir in d.iterdir():
            if not shot_dir.is_dir():
                continue
            m = re.match(r"(\d+)-shot_" + re.escape(base_name), shot_dir.name, re.I)
            if not m:
                continue
            k = int(m.group(1))
            out_dir = DST_ROOT / dataset / f"{k}shot"
            out_dir.mkdir(parents=True, exist_ok=True)
            splits = []
            for split_dir in sorted(shot_dir.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else -1):
                if not split_dir.is_dir() or not split_dir.name.isdigit():
                    continue
                idx_path = split_dir / "idx.pt"
                labels_path = split_dir / "labels.pt"
                if not idx_path.exists() or not labels_path.exists():
                    continue
                idx = torch.load(idx_path, weights_only=True)
                labels = torch.load(labels_path, weights_only=True)
                if hasattr(idx, "tolist"):
                    idx = idx.tolist()
                if hasattr(labels, "tolist"):
                    labels = labels.tolist()
                splits.append({"indices": idx, "labels": labels})
            if not splits:
                continue
            out_file = out_dir / "splits.pt"
            torch.save(
                {
                    "splits": splits,
                    "meta": {
                        "dataset": dataset,
                        "k_shot": k,
                        "n_splits": len(splits),
                        "n_way": _count_n_way(splits, "labels"),
                    },
                },
                out_file,
            )
            print(f"  {dataset} {k}shot -> {out_file} ({len(splits)} splits)")


def migrate_graph_fewshot():
    """Graph: fewshot_<name>_graph/<k>-shot_<name>_graph/0,1,.../idx.pt, batch.pt, labels.pt -> splits.pt"""
    for d in sorted(SRC_DATA.iterdir()):
        if not d.is_dir() or not d.name.startswith("fewshot_") or "_graph" not in d.name or "_new" in d.name:
            continue
        base_name = d.name.replace("fewshot_", "").replace("_graph", "")
        dataset = to_dataset_name(d.name)
        for shot_dir in d.iterdir():
            if not shot_dir.is_dir():
                continue
            m = re.match(r"(\d+)-shot_" + re.escape(base_name) + r"_graph", shot_dir.name, re.I)
            if not m:
                continue
            k = int(m.group(1))
            out_dir = DST_ROOT / dataset / f"{k}shot_graph_batch"
            out_dir.mkdir(parents=True, exist_ok=True)
            splits = []
            for split_dir in sorted(shot_dir.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else -1):
                if not split_dir.is_dir() or not split_dir.name.isdigit():
                    continue
                idx_path = split_dir / "idx.pt"
                batch_path = split_dir / "batch.pt"
                labels_path = split_dir / "labels.pt"
                if not idx_path.exists() or not batch_path.exists() or not labels_path.exists():
                    continue
                idx = torch.load(idx_path, weights_only=False)
                batch = torch.load(batch_path, weights_only=False)
                labels = torch.load(labels_path, weights_only=False)
                splits.append({"idx": idx, "batch": batch, "labels": labels})
            if not splits:
                continue
            out_file = out_dir / "splits.pt"
            torch.save(
                {
                    "splits": splits,
                    "meta": {
                        "dataset": dataset,
                        "k_shot": k,
                        "n_splits": len(splits),
                        "n_way": _count_n_way(splits, "labels"),
                    },
                },
                out_file,
            )
            print(f"  {dataset} {k}shot_graph_batch -> {out_file} ({len(splits)} splits)")


def main():
    print("Migrate node few-shot splits...")
    migrate_node_fewshot()
    print("Migrate graph few-shot splits...")
    migrate_graph_fewshot()
    print("Done.")


if __name__ == "__main__":
    main()
