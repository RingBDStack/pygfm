# `pygfm.private.utlis`

Baseline-specific **data-side** helpers live here: graph construction, features and domain alignment, RAG corpus / motif building, downstream split generation, etc. Implementations evolve with each baseline (RAG-GFM, MDGPT, …); this is **not** a single global read-only dataset layer.

## How this relates to “public” code (avoid duplicating mental models)

| Topic | Canonical location | Role of this package |
|-------|--------------------|----------------------|
| Shared GNN backbones (GCN/GAT/GIN, …) | `pygfm.private.core` | `utlis.__init__` re-exports some classes |
| Losses / GRL / task heads, etc. | `pygfm.public.utils.loss_func` | `loss_calculation.py` is a compatibility re-export |
| Negative sampling and similar | `pygfm.public.utils.others` (via `loss_func`, etc.) | Same as above |

Prefer **`private.core`** / **`public.utils`** in new code; for **baseline data scripts**, you may import from **`pygfm.private.utlis`** as a single entry point.

## Directory overview

| Path | Purpose |
|------|---------|
| `graph_construction.py` | Graph construction entry points |
| `feature_handling.py` | Raw / text-encoded / precomputed embedding features |
| `graph_type_variants.py` | Static graphs, multi-domain graph sets, projection MLP variants |
| `domain_alignment.py` | Domain alignment, task adaptation, NodeLevelPrompt, etc. |
| `loss_calculation.py` | Compatibility re-export (see `public.utils.loss_func`) |
| `downstream_data_gen/` | Few-shot / graph-batch split generation for downstream |
| `rag_gfm/` | RAG-GFM: corpus build, motif library, loaders and model helpers |

For finer API notes see repo root `docs/MODULE_SPEC.md` and `baseline_docs/`.
