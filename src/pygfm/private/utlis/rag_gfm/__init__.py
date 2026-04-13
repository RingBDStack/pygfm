"""
RAG-GFM private data helpers: corpus build + motif library build.

- Corpus: embed node text and class descriptions from multiple datasets into nano-vectordb for RAG retrieval.
- Motif lib: train a subgraph encoder per dataset and build motif_vectordb for motif retrieval.
"""

from .corpus_builder import (
    RAGCorpusBuilderConfig,
    build_rag_corpus,
)
from .motif_builder import (
    MotifLibBuilderConfig,
    build_motif_lib,
)

__all__ = [
    "RAGCorpusBuilderConfig",
    "build_rag_corpus",
    "MotifLibBuilderConfig",
    "build_motif_lib",
]
