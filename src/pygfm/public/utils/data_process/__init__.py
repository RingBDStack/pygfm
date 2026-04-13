"""Data processing: PyG/OGB, BERT text features; domain alignment / subgraph splits live in ``private.utlis``."""

from __future__ import annotations

from pygfm.private.utlis.domain_alignment import DomainAlignment
from pygfm.private.utlis.downstream_data_gen import build_test_subgraphs

from .ogb_preprocess import load_ogb_graph_structure_only
from .pyg_graph import PyGGraph, to_bidirected_and_self_loop
from .text_embed import BertTextEncoder, encode_texts_with_bert

__all__ = [
    "BertTextEncoder",
    "DomainAlignment",
    "PyGGraph",
    "build_test_subgraphs",
    "encode_texts_with_bert",
    "load_ogb_graph_structure_only",
    "to_bidirected_and_self_loop",
]
