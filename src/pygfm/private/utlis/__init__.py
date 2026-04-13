"""
Data Module for GFM-Toolbox

This module provides data processing and feature engineering capabilities for Graph Foundation Models.
"""

from .graph_construction import GraphConstruction
from .feature_handling import (
    RawNodeFeatures,
    TextEncodedNodeAttributes,
    PreExtractedEmbeddings,
    FeatureEngineeringMLP,
)
from .graph_type_variants import (
    StaticGraph,
    MultiDomainGraphCollection,
    FeatureProjectionMLP,
)
from pygfm.public.utils.loss_func import sample_negative_pairs
from .loss_calculation import (
    ContrastiveLossModule,
    TaskHead,
    DomainRegularizer,
    NodeNodeContrastiveLoss,
)
from .domain_alignment import (
    DomainAlignment,
    TaskAdapter,
    NodeLevelPrompt,
    ComposedNodeLevelPrompt,
)
from pygfm.private.core import (
    GNNBackboneEncoder,
    GCNEncoder,
    GATEncoder,
    GraphAttentionLayer,
    GINEncoder,
    GINLayer,
)
from .downstream_data_gen import (
    generate_few_shot_splits,
    generate_graph_batch_splits,
    DownstreamGeneratorConfig,
)
from .rag_gfm import (
    build_rag_corpus,
    RAGCorpusBuilderConfig,
    build_motif_lib,
    MotifLibBuilderConfig,
)


__all__ = [
    "GraphConstruction",
    "RawNodeFeatures",
    "TextEncodedNodeAttributes",
    "FeatureEngineeringMLP",
    "DomainAlignment",
    "PreExtractedEmbeddings",
    "StaticGraph",
    "MultiDomainGraphCollection",
    "FeatureProjectionMLP",
    "ContrastiveLossModule",
    "TaskAdapter",
    "TaskHead",
    "DomainRegularizer",
    "DomainAlignment",
    "GNNBackboneEncoder",
    "GCNEncoder",
    "GATEncoder",
    "GraphAttentionLayer",
    "GINEncoder",
    "GINLayer",
    "generate_few_shot_splits",
    "generate_graph_batch_splits",
    "DownstreamGeneratorConfig",
    "build_rag_corpus",
    "RAGCorpusBuilderConfig",
    "build_motif_lib",
    "MotifLibBuilderConfig",
    "sample_negative_pairs",
    "NodeNodeContrastiveLoss",
    "NodeLevelPrompt",
    "ComposedNodeLevelPrompt",
]

