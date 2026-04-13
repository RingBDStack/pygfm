# `pygfm.public.utils.loss_func`

This directory holds **losses and components tied directly to training objectives**: `nn.Module` losses, functional contrastive objectives, task heads, domain regularizers, GRL, etc.

**Sampling, tensor gather, thin few-shot CE wrappers**, and similar “around the loss” utilities live in **`pygfm.public.utils.loss_func.loss_support`** (`sample_negative_pairs` is still implemented under `public.utils.others` and re-exported through `loss_support`).

## Design principles

1. **Single implementation**; baselines may re-export.
2. **Clear names**; keep baseline-compat aliases when needed (e.g. `InfoNCEloss`).
3. **Light dependencies**; tensor shapes belong in docstrings.

## Module map

| Module | Contents |
|--------|----------|
| **`node_node_contrastive.py`** | `NodeNodeContrastiveLoss` |
| **`motif_contrastive_loss.py`** | `motif_subgraph_contrastive_loss` |
| **`domain_contrastive.py`** | `ContrastiveLossModule` |
| **`gradient_reversal.py`** | `GradientReversal` |
| **`task_head.py`** | `TaskHead` |
| **`domain_regularizer.py`** | `DomainRegularizer` |
| **`info_nce_mi_matrix.py`** | `InfoNCEMIMatrixLoss`, `InfoNCEloss` |
| **`pairwise_ranking.py`** | `NegLogLoss`, `FirstPosNegLoss`, `MRRLoss`, `IDLoss` |
| **`cca_loss.py`** | `CCALoss` |
| **`spectral_loss.py`** | `spectral_loss`, `SpectralRegularizationLoss` (graph Laplacian spectral regularization, BRIDGE DownPrompt, etc.) |

## Import examples

```python
from pygfm.public.utils.loss_func import (
    NodeNodeContrastiveLoss,
    motif_subgraph_contrastive_loss,
    spectral_loss,
    SpectralRegularizationLoss,
    InfoNCEloss,
    CCALoss,
)
from pygfm.public.utils.loss_func.loss_support import (
    gather_rows,
    sample_negative_pairs,
    few_shot_cross_entropy,
)
```

## vs. `pygfm.private.utlis.loss_calculation`

`private.utlis.loss_calculation` is a compatibility re-export; new code should use **`loss_func`** / **`loss_support`** directly.

## `gather_rows` and legacy `compareloss`

See the docstring on **`loss_support.gather_rows`** (MultigPrompt / SA2GFM temperature-scaling differences).
