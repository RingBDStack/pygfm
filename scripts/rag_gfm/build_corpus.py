#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG-GFM step 1: build corpus (RAG-GFM-specific).

Embed node text and class descriptions from multiple datasets into nano-vectordb for RAG.
Graph data under data_root must include raw_texts.

Usage:
  python scripts/rag_gfm/build_corpus.py
  python scripts/rag_gfm/build_corpus.py --data_root datasets/rag_gfm --datasets Cora,Citeseer,Pubmed
  python scripts/rag_gfm/build_corpus.py --corpus_output downstream_data/rag_gfm/corpus/unified_database.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pygfm.private.utlis.rag_gfm import RAGCorpusBuilderConfig, build_rag_corpus


def main():
    p = argparse.ArgumentParser(
        description="RAG-GFM corpus: build nano-vectordb from graph data for RAG"
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="datasets/rag_gfm",
        help="Graph data root (GFM-Toolbox baseline layout)",
    )
    p.add_argument(
        "--corpus_output",
        type=str,
        default="downstream_data/rag_gfm/corpus/unified_database.json",
        help="Corpus output path (nano-vectordb JSON)",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="Cora,Citeseer,Pubmed,Photo,Computers",
        help="Dataset names to include, comma-separated",
    )
    p.add_argument(
        "--text_encoder",
        type=str,
        default="SentenceBert",
        choices=["SentenceBert", "bert", "Bert"],
        help="Text encoder",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (e.g. cuda/cpu)",
    )
    args = p.parse_args()

    # Relative paths resolved from repo root
    corpus_output = args.corpus_output
    if not os.path.isabs(corpus_output):
        corpus_output = str(ROOT / corpus_output)
    data_root = args.data_root
    if not os.path.isabs(data_root):
        data_root = str(ROOT / data_root)

    config = RAGCorpusBuilderConfig(
        data_root=data_root,
        dataset_names=[s.strip() for s in args.datasets.split(",") if s.strip()],
        corpus_output_path=corpus_output,
        text_encoder=args.text_encoder,
        device=args.device,
    )
    out_path = build_rag_corpus(config)
    print(f"RAG-GFM corpus written: {out_path}")


if __name__ == "__main__":
    main()
