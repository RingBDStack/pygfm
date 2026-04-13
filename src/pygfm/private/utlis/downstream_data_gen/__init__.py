"""
Downstream data generation: 1-shot / 5-shot splits and subgraph batch data.

Precompute few-shot node-classification support/query splits instead of sampling at runtime,
for reproducibility and a fixed eval protocol.
"""

from .generator import (
    generate_few_shot_splits,
    generate_graph_batch_splits,
    build_test_subgraphs,
    DownstreamGeneratorConfig,
)

__all__ = [
    "generate_few_shot_splits",
    "generate_graph_batch_splits",
    "build_test_subgraphs",
    "DownstreamGeneratorConfig",
]
